/**
 * @file deviceArrays.cu
 * @brief Templated classes for managing 1D and 2D arrays on a CUDA device in column-major order.
 */

#include <vector>
#include <numeric>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fstream>

/**
 * @brief Base class for file transfer helpers. Manages chunking logic.
 */
template <typename T>
class CuFileHelper {
protected:    
    
    size_t _colsProcessed; // Number of bytes already processed
    
    std::vector<T> _hostBuffer; // Host buffer to hold the current chunk of data

public:

    const size_t _totalCols; // Total number of bytes to process
    const size_t _maxColsPerChunk; 
    const size_t _rows;

    /**
     * @brief Constructor for CuFileHelper.
     * @param rows Number of rows in the array.
     * @param cols Number of columns in the array.
     */
    CuFileHelper(size_t rows, size_t cols)
        : _totalCols(cols), 
        _colsProcessed(0), 
        _maxColsPerChunk(std::max(size_t(1), 1024*1024/(rows * sizeof(T)))),
        _hostBuffer(_maxColsPerChunk * rows),
        _rows(rows) {}

    virtual ~CuFileHelper() = default;

    /**
     * @brief Checks if there are more chunks to process.
     * @return True if more data can be transferred, false otherwise.
     */
    bool hasNext() const {
        return _colsProcessed < _totalCols;
    }

    /**
     * @brief Gets the number of bytes in the next chunk.
     * @return The size of the next chunk in bytes.
     */
    size_t getNextChunkColNumber() const {
        return std::min(_maxColsPerChunk, _totalCols - _colsProcessed);
    }
    
    /**
     * @brief Gets a pointer to the host buffer to be used for the current chunk.
     * @return Pointer to the host buffer.
     */
    T* getHostBuffer() {
        return _hostBuffer.data();
    }

    /**
     * @brief Updates the processed byte count after a successful chunk transfer.
     * @param bytes_transferred The number of bytes transferred in the last chunk.
     */
    void updateProgress() {
        _colsProcessed += getNextChunkColNumber();
    }

    /**
     * @brief Gets the number of columns processed so far.
     * @return The number of columns processed.
     */
    size_t getColsProcessed() const {
        return _colsProcessed;
    }
};

/**
 * @brief Helper class to read chunks from a file and provide them for device transfer.
 */
template <typename T>
class SetFromFile : public CuFileHelper<T> {
private:
    std::istream& _input_stream;

public:
    /**
     * @brief Constructor for CuSetFromFileHelper.
     * @param array The CudaArray to set data into.
     * @param input_stream The input stream to read from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    SetFromFile(size_t rows, size_t cols, std::istream& input_stream)
        : CuFileHelper<T>(rows, cols), _input_stream(input_stream) {}

    /**
     * @brief Reads the next chunk of data from the file into the internal host buffer.
     */
    void readNextChunk() {
        size_t current_chunk_bytes = this->getNextChunkColNumber() * this->_rows * sizeof(T);
        if (current_chunk_bytes > 0) {
            this->_input_stream.read(reinterpret_cast<char*>(this->_hostBuffer.data()), current_chunk_bytes);
            if (!this->_input_stream) throw std::runtime_error("Stream read error or premature end of stream.");
            
        }
    }
};

/**
 * @brief Helper class to get chunks from the device and write them to a file.
 */
template <typename T>
class GetToFile : public CuFileHelper<T> {
private:
    std::ostream& _output_stream;

public:
    /**
     * @brief Constructor for GetToFile.
     * @param array The CudaArray to read from.
     * @param output_stream The output stream to write to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    GetToFile(size_t rows, size_t cols, std::ostream& output_stream)
        : CuFileHelper<T>(rows, cols), _output_stream(output_stream) {}

    /**
     * @brief Writes the next chunk of data from the internal host buffer to the file.
     * @param current_chunk_bytes The size of the chunk to write.
     */
    void writeNextChunkToFile() {
        size_t current_chunk_bytes = this->getNextChunkColNumber() * this->_rows * sizeof(T);
        if (current_chunk_bytes > 0) {
            this->_output_stream.write(reinterpret_cast<const char*>(this->_hostBuffer.data()), current_chunk_bytes);
            if (!this->_output_stream) throw std::runtime_error("Stream write error.");
        }
    }
};


/**
 * @enum IndexType
 * @brief Indicates whether to index by row or by column.
 */
enum class IndexType {
    Row,    /**< Row index */
    Column  /**< Column index */
};

/**
 * @brief CUDA device memory deleter function for std::shared_ptr.
 * @param ptr Pointer to CUDA device memory to free.
 */
inline void cudaFreeDeleter(void* ptr) {
    if (ptr) cudaFree(ptr);
}

/**
 * @brief Abstract base template for CUDA array types.
 */
template <typename T>
class CuArray {

public:
    /** Number of rows (const). */
    const size_t _rows;

    /** Number of columns (const). */
    const size_t _cols;


protected:
    /** Pointer to the device memory (shared pointer). */
    std::shared_ptr<void> _ptr;
    /** Leading dimension (stride) in elements (const). */
    size_t _ld;

    /**
     * @brief Protected constructor for CudaArray.
     * @param rows Number of rows in the array.
     * @param cols Number of columns in the array.
     * @param ld Leading dimension (stride) in elements.
     */
    CuArray(size_t rows, size_t cols, size_t ld)
        : _rows(rows), _cols(cols), _ld(ld){}

public:
    /**
     * @brief Default destructor for CudaArray.
     * Cleans up the device memory automatically.
     */
    virtual ~CuArray() = default;

    /**
     * @brief Get the number of elements in the array.
     * @return Total number of elements in the array.
     * @note This is a pure virtual function, must be implemented by derived classes.
     * @return size_t Total number of elements in the array.
     */
    virtual size_t size() const = 0;
    /**
     * @brief Get the size of the array in bytes.
     * @return Total size in bytes of the array.
     * @note This is a pure virtual function, must be implemented by derived classes.
     */
    virtual size_t bytes() const = 0;  

    /**
     * @brief Set the array data from host memory.
     * @param hostData Pointer to the host data to copy from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     * @note This is a pure virtual function, must be implemented by derived classes.
     */
    virtual void set(const T* hostData, cudaStream_t stream = 0) = 0;

    /**
     * @brief Get the array data to host memory.
     * @param hostData Pointer to the host memory to copy to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     * @note This is a pure virtual function, must be implemented by derived classes.
     */
    virtual void get(T* hostData, cudaStream_t stream = 0) const = 0;
    
    /**
     * @brief Set the array data from another CudaArray.
     * @param src Source CudaArray to copy from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     * @note This is a pure virtual function, must be implemented by derived classes.
     */
    virtual void set(const CuArray<T>& src, cudaStream_t stream = 0) = 0;

    /**
     * @brief Get the array data to another CudaArray.
     * @param dst Destination CudaArray to copy to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     * @note This is a pure virtual function, must be implemented by derived classes.
     */
    virtual void get(CuArray<T>& dst, cudaStream_t stream = 0) const = 0;

        /**
     * @brief Sets the array data from an input stream.
     * @param input_stream The input stream to read from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    virtual void set(std::istream& input_stream, cudaStream_t stream = 0) = 0;

    /**
     * @brief Gets the array data to an output stream.
     * @param output_stream The output stream to write to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    virtual void get(std::ostream& output_stream, cudaStream_t stream = 0) const = 0;


    /**
     * @brief Get the raw pointer to the device memory.
     * @return Pointer to the device memory.
     */
    T* data() { return static_cast<T*>(_ptr.get()); }

    /**
     * @brief Get the raw pointer to the device memory (const version).
     * @return Pointer to the device memory.
     */
    const T* data() const { return static_cast<const T*>(_ptr.get()); }

    /**
     * @brief Get the leading dimension (stride) in elements.
     * @return Leading dimension in elements.
     */
    size_t getLD() const { return _ld; }

    /**
     * @brief Get the shared pointer to the device memory.
     * @return Shared pointer to the device memory.
     */
    std::shared_ptr<void> getPtr() const{ return _ptr; }
        
};

/**
 * @brief CUDA 2D array view, column-major storage.
 *
 * Storage layout: columns are contiguous with stride _ld.
 */
template <typename T>
class CuArray2D : public CuArray<T> {
public:
    /**
     * @brief Constructor for CudaArray2D.
     * Allocates device memory for a 2D array with given rows and columns.
     * @param rows Number of rows in the array.
     * @param cols Number of columns in the array.
     * @throws std::runtime_error if cudaMallocPitch fails.
     * @note The pitch (leading dimension) is automatically calculated based on the column size.
     */
    CuArray2D(size_t rows, size_t cols): CuArray<T>(rows, cols, 0) {
        void* rawPtr = nullptr;
        size_t pitch = 0;
        cudaError_t err = cudaMallocPitch(&rawPtr, &pitch, rows * sizeof(T), cols);
        if (err != cudaSuccess) 
            throw std::runtime_error("cudaMallocPitch failed");
        
        this->_ptr = std::shared_ptr<void>(rawPtr, cudaFreeDeleter);
        this->_ld = pitch / sizeof(T);  // leading dimension in elements
    }

    /**
     * @brief Constructor for CudaArray2D that creates a subarray view.
     * @param superArray The parent CudaArray2D to create a subarray from.
     * @param startRow Starting row index in the parent array.
     * @param startCol Starting column index in the parent array.
     * @param height Height of the subarray.
     * @param width Width of the subarray.
     */
    CuArray2D(const CuArray2D<T>& superArray, size_t startRow, size_t startCol, size_t height, size_t width)
        : CuArray<T>(height, width, superArray.getLD()) {
        size_t offset = startCol * superArray.getLD() + startRow; // column-major: col offset first, then row offset
        this->_ptr = std::shared_ptr<void>(
            superArray._ptr,
            static_cast<void*>(static_cast<char*>(superArray._ptr.get()) + offset * sizeof(T))
        );
    }

    /**
     * @brief The numver of elements in the array.
     * @return Total number of elements in the array (rows * cols).
     */
    size_t size() const override {
        return this->_rows * this->_cols;
    }

    /**
     * @brief Get the size of the array in bytes.
     * @return Total size in bytes of the array (rows * ld * sizeof(T)).
     */
    size_t bytes() const override {
        return this->_cols * this->_ld * sizeof(T);
    }

    /**
     * @brief Set the array data from host memory.
     * @param src Pointer to the host data to copy from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void set(const T* src, cudaStream_t stream = 0) override {
        cudaMemcpy2DAsync(
            this->_ptr.get(), this->_ld * sizeof(T),
            src, this->_rows * sizeof(T),
            this->_rows * sizeof(T), this->_cols,
            cudaMemcpyHostToDevice, stream
        );
    }

    /**
     * @brief Get the array data to host memory.
     * @param dst Pointer to the host memory to copy to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void get(T* dst, cudaStream_t stream = 0) const override {
        cudaMemcpy2DAsync(
            dst, this->_rows * sizeof(T),
            this->_ptr.get(), this->_ld * sizeof(T),
            this->_rows * sizeof(T), this->_cols,
            cudaMemcpyDeviceToHost, stream
        );
    }

    /**
     * @brief Set the array data from another CudaArray.
     * @param src Source CudaArray to copy from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void set(const CuArray<T>& src, cudaStream_t stream = 0) override {
        cudaMemcpy2DAsync(
            this->_ptr.get(), this->_ld * sizeof(T),
            src.data(), src.getLD() * sizeof(T),
            this->_rows * sizeof(T), this->_cols,
            cudaMemcpyDeviceToDevice, stream
        );
    }

    /**
     * @brief Get the array data to another CudaArray.
     * @param dst Destination CudaArray to copy to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void get(CuArray<T>& dst, cudaStream_t stream = 0) const override {
        cudaMemcpy2DAsync(
            dst.data(), dst.getLD() * sizeof(T),
            this->_ptr.get(), this->_ld * sizeof(T),
            this->_rows * sizeof(T), this->_cols,
            cudaMemcpyDeviceToDevice, stream
        );
    }
/**
     * @brief Sets the array data from an input stream.
     * @param input_stream The input stream to read from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void set(std::istream& input_stream, cudaStream_t stream = 0) override {
        SetFromFile<T> helper(this->_rows, this->_cols, input_stream);
        while (helper.hasNext()) {
            helper.readNextChunk();
            CuArray2D<T> subArray(
                *this, 
                0, 
                helper.getColsProcessed(),
                helper.getNextChunkColNumber(),
                this->_rows
            );
            
            subArray.set(helper.getHostBuffer(), stream);
            helper.updateProgress();
        }
    }

    /**
     * @brief Gets the array data to an output stream.
     * @param output_stream The output stream to write to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void get(std::ostream& output_stream, cudaStream_t stream = 0) const override {
        GetToFile<T> helper(this->_rows, this->_cols, output_stream);
        while (helper.hasNext()) {
            CuArray2D<T> subArray(
                *this, 
                0, 
                helper.getColsProcessed(),
                helper.getNextChunkColNumber(),
                this->_rows
            );            
            subArray.get(helper.getHostBuffer(), stream);
            helper.writeNextChunkToFile();
            helper.updateProgress();
        }
    }
};

/**
 * @brief CUDA 1D array view, representing either a vector or a single column/row slice.
 *
 * Note: For column-major data,
 *   - _rows = 1
 *   - _cols = length of vector
 *   - _ld = stride between elements (in elements)
 */
template <typename T>
class CuArray1D : public CuArray<T> {
public:
    /**
     * @brief Constructor for CudaArray1D.
     * Allocates device memory for a 1D array with given length.
     * @param length Length of the 1D array.
     * @throws std::runtime_error if cudaMalloc fails.
     */
    explicit CuArray1D(size_t length)
        : CuArray<T>(1, length, 1) {
        void* rawPtr = nullptr;
        cudaMalloc(&rawPtr, length * sizeof(T));
        this->_ptr = std::shared_ptr<void>(rawPtr, cudaFreeDeleter);
    }

    /**
     * @brief Constructor for CudaArray1D that creates a subarray view.
     * @param superArray The parent CudaArray1D to create a subarray from.
     * @param offset Starting index in the parent array.
     * @param length Length of the subarray.
     * @param ld Leading dimension (stride) in elements for the subarray.
     * @throws std::out_of_range if offset + length exceeds the parent array size.
     * @note The leading dimension is used to calculate the offset correctly.
     */ 
    CuArray1D(const CuArray1D<T>& superArray, size_t offset, size_t length, size_t stride = 1)
        : CuArray<T>(1, length, stride * superArray.getLD()) {
        this->_ptr = std::shared_ptr<void>(
            superArray._ptr,
            static_cast<void*>(static_cast<char*>(superArray._ptr.get()) + offset * this->_ld * sizeof(T))
        );
    }

    /**
     * @brief Constructor for CudaArray1D that extracts a row or column from a CudaArray2D.
     * @param extractFrom The parent CudaArray2D to extract from.
     * @param index The row or column index to extract.
     * @param indexType Specify whether to extract a row or a column (IndexType::Row or IndexType::Column).
     * @throws std::out_of_range if index is out of bounds.
     */
    CuArray1D(const CuArray2D<T>& extractFrom, int index, IndexType indexType):  
    CuArray<T>(
        1, 
        indexType == IndexType::Row ? extractFrom._cols : extractFrom._rows, 
        indexType == IndexType::Row ? extractFrom.getLD() : 1
    ) {
        
        if ((indexType == IndexType::Column && static_cast<size_t>(index) >= extractFrom._cols) || (indexType == IndexType::Row && static_cast<size_t>(index) >= extractFrom._rows))
            throw std::out_of_range("Out of range");
        
        size_t offset = indexType == IndexType::Row ? static_cast<size_t>(index) : static_cast<size_t>(index) * extractFrom.getLD();
       
        this->_ptr = std::shared_ptr<void>(
            extractFrom.getPtr(),
            const_cast<void*>(reinterpret_cast<const void*>(reinterpret_cast<const char*>(extractFrom.data()) + offset * sizeof(T)))
        );
    }

    /**
     * @brief Get the number of elements in the array.
     * @return Total number of elements in the array (cols).
     */
    size_t size() const override {
        return this->_cols;
    }

    /**
     * @brief Get the size of the array in bytes.
     * @return Total size in bytes of the array (cols * ld * sizeof(T)).
     */
    size_t bytes() const override {        
        return this->_cols * this->_ld * sizeof(T);
    }

    /**
     * @brief Set the array data from host memory.
     * @param hostData Pointer to the host data to copy from.
     * @param stream CUDA strethis->bytes()am for asynchronous operations (default is 0).
     */
    void set(const T* hostData, cudaStream_t stream = 0) override {
        
        if (this->_ld == 1) cudaMemcpyAsync(this->_ptr.get(), hostData, bytes(), cudaMemcpyHostToDevice, stream);
        else cudaMemcpy2DAsync(
                this->_ptr.get(), this->_ld * sizeof(T),
                hostData, sizeof(T),
                sizeof(T), this->_cols,
                cudaMemcpyHostToDevice, stream
            );
    }

    /**
     * @brief Get the array data to host memory.
     * @param hostData Pointer to the host memory to copy to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void get(T* hostData, cudaStream_t stream = 0) const override {
        if (this->_ld == 1)
            cudaMemcpyAsync(hostData, this->_ptr.get(), bytes(), cudaMemcpyDeviceToHost, stream);
        else cudaMemcpy2DAsync(
                hostData, sizeof(T),
                this->_ptr.get(), this->_ld * sizeof(T),
                sizeof(T), this->_cols,
                cudaMemcpyDeviceToHost, stream
            );
    }

    /**
     * @brief Set the array data from another CudaArray.
     * @param src Source CudaArray to copy from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void set(const CuArray<T>& src, cudaStream_t stream = 0) override {
        if (this->_ld == 1 && src.getLD() == 1) {
            cudaMemcpyAsync(this->_ptr.get(), src.data(), bytes(), cudaMemcpyDeviceToDevice, stream);
        } else {
            cudaMemcpy2DAsync(
                this->_ptr.get(), this->_ld * sizeof(T),
                src.data(), src.getLD() * sizeof(T),
                sizeof(T), this->_cols,
                cudaMemcpyDeviceToDevice, stream
            );
        }
    }

    /**
     * @brief Get the array data to another CudaArray.
     * @param dst Destination CudaArray to copy to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void get(CuArray<T>& dst, cudaStream_t stream = 0) const override {
        if (this->_ld == 1 && dst.getLD() == 1) {
            cudaMemcpyAsync(dst.data(), this->_ptr.get(), bytes(), cudaMemcpyDeviceToDevice, stream);
        } else {
            cudaMemcpy2DAsync(
                dst.data(), dst.getLD() * sizeof(T),
                this->_ptr.get(), this->_ld * sizeof(T),
                sizeof(T), this->_cols,
                cudaMemcpyDeviceToDevice, stream
            );
        }
    }

    
    /**
     * @brief Sets the array data from an input stream.
     * @param input_stream The input stream to read from.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void set(std::istream& input_stream, cudaStream_t stream = 0) override {
        SetFromFile<T> helper(this->_rows, this->_cols, input_stream);
        while (helper.hasNext()) {
            helper.readNextChunk();
            CuArray1D<T> subArray(
                *this,
                helper.getColsProcessed(),
                helper.getNextChunkColNumber()
            );
            
            subArray.set(helper.getHostBuffer(), stream);
            helper.updateProgress();
        }
    }

    /**
     * @brief Gets the array data to an output stream.
     * @param output_stream The output stream to write to.
     * @param stream CUDA stream for asynchronous operations (default is 0).
     */
    void get(std::ostream& output_stream, cudaStream_t stream = 0) const override {
        GetToFile<T> helper(this->_rows, this->_cols, output_stream);
        while (helper.hasNext()) {
            CuArray1D<T> subArray(
                *this, 
                helper.getColsProcessed(),
                helper.getNextChunkColNumber()
            );            
            subArray.get(helper.getHostBuffer(), stream);
            helper.writeNextChunkToFile();
            helper.updateProgress();
        }
    }
};

// --- Helper Functions and Macros for Testing ---

/**
 * @brief Helper function to check for CUDA errors and exit on failure.
 * @param err The cudaError_t value to check.
 * @param file The file name where the error occurred.
 * @param line The line number where the error occurred.
 */
void checkCudaErrors(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Macro to wrap CUDA function calls for easy error checking.
 */
#define CHECK_CUDA_ERROR(err) checkCudaErrors(err, __FILE__, __LINE__)

/**
 * @brief A helper function to verify if two vectors are identical.
 * @tparam T The data type of the vectors.
 * @param expected The vector containing the expected values.
 * @param result The vector containing the test results.
 * @param test_name A string to identify the test.
 */
template <typename T>
void verifyVectors(const std::vector<T>& expected, const std::vector<T>& result, const std::string& test_name) {
    if (expected == result) {
        std::cout << "✅ " << test_name << " successful." << std::endl;
    } else {
        std::cout << "❌ " << test_name << " failed." << std::endl;
        std::cerr << "Expected size: " << expected.size() << ", Result size: " << result.size() << std::endl;
    }
}

/**
 * @brief Runs all tests for a specific data type.
 * @tparam T The data type to test (e.g., int, float).
 */
template <typename T>
void runTests() {
    std::cout << "--- Running tests for type " << typeid(T).name() << " ---" << std::endl;

    // --- 2D Array Tests ---
    std::cout << "\n## Testing CudaArray2D" << std::endl;
    const size_t rows_2d = 4;
    const size_t cols_2d = 3;
    std::vector<T> host_data_2d(rows_2d * cols_2d);
    std::iota(host_data_2d.begin(), host_data_2d.end(), static_cast<T>(1));

    try {
        CuArray2D<T> device_array_2d(rows_2d, cols_2d);
        std::cout << "CudaArray2D created with dimensions " << device_array_2d._rows << "x" << device_array_2d._cols << ", LD: " << device_array_2d.getLD() << std::endl;
        device_array_2d.set(host_data_2d.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        std::vector<T> host_result_2d(rows_2d * cols_2d);
        device_array_2d.get(host_result_2d.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        verifyVectors(host_data_2d, host_result_2d, "CudaArray2D set/get");
    } catch (const std::runtime_error& e) {
        std::cerr << "CudaArray2D Test Failed: " << e.what() << std::endl;
    }

    // --- 2D Subarray Test ---
    std::cout << "\n## Testing CudaArray2D Subarray View" << std::endl;
    try {
        CuArray2D<T> parent_array(rows_2d, cols_2d);
        parent_array.set(host_data_2d.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        const size_t startRow = 1;
        const size_t startCol = 1;
        const size_t subHeight = 2;
        const size_t subWidth = 2;
        CuArray2D<T> subArray(parent_array, startRow, startCol, subHeight, subWidth);
        
        std::vector<T> retrievedSubArray(subHeight * subWidth);
        subArray.get(retrievedSubArray.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        std::vector<T> expected_subarray;
        for (size_t c = startCol; c < startCol + subWidth; ++c) {
            for (size_t r = startRow; r < startRow + subHeight; ++r) {
                expected_subarray.push_back(host_data_2d[c * rows_2d + r]);
            }
        }
        verifyVectors(expected_subarray, retrievedSubArray, "CudaArray2D Subarray");
    } catch (const std::runtime_error& e) {
        std::cerr << "CudaArray2D Subarray Test Failed: " << e.what() << std::endl;
    }

    // --- 1D Array Tests ---
    std::cout << "\n## Testing CudaArray1D" << std::endl;
    const size_t length_1d = 8;
    std::vector<T> host_data_1d(length_1d);
    std::iota(host_data_1d.begin(), host_data_1d.end(), static_cast<T>(100));

    try {
        CuArray1D<T> device_array_1d(length_1d);
        device_array_1d.set(host_data_1d.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        std::vector<T> host_result_1d(length_1d);
        device_array_1d.get(host_result_1d.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        verifyVectors(host_data_1d, host_result_1d, "CudaArray1D set/get");
    } catch (const std::runtime_error& e) {
        std::cerr << "CudaArray1D Test Failed: " << e.what() << std::endl;
    }

    // --- 1D Extraction from 2D Array Test ---
    std::cout << "\n## Testing CudaArray1D Extraction from CudaArray2D" << std::endl;
    try {
        CuArray2D<T> parent_2d(rows_2d, cols_2d);
        parent_2d.set(host_data_2d.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Test column extraction
        int col_index = 1;
        CuArray1D<T> extracted_col(parent_2d, col_index, IndexType::Column);
        std::vector<T> host_col_result(extracted_col.size());
        extracted_col.get(host_col_result.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        std::vector<T> expected_col;
        for (size_t r = 0; r < rows_2d; ++r) {
            expected_col.push_back(host_data_2d[col_index * rows_2d + r]);
        }
        verifyVectors(expected_col, host_col_result, "CudaArray1D extraction of a column");
        
        // Test row extraction
        int row_index = 1;
        CuArray1D<T> extracted_row(parent_2d, row_index, IndexType::Row);
        std::vector<T> host_row_result(extracted_row.size());
        extracted_row.get(host_row_result.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        std::vector<T> expected_row;
        for (size_t c = 0; c < cols_2d; ++c) {
            expected_row.push_back(host_data_2d[c * rows_2d + row_index]);
        }
        verifyVectors(expected_row, host_row_result, "CudaArray1D extraction of a row");
    } catch (const std::runtime_error& e) {
        std::cerr << "CudaArray1D Extraction Test Failed: " << e.what() << std::endl;
    }
}

/**
 * @brief Runs tests for file I/O operations using streams.
 * @tparam T The data type to test.
 */
template <typename T>
void runFileIOTests() {
    std::cout << "\n--- Running File I/O tests for type " << typeid(T).name() << " ---" << std::endl;
    
    const size_t rows = 1000;
    const size_t cols = 2000;
    const size_t large_array_size = rows * cols;
    
    std::string test_filename = "test_array.bin";
    
    try {
        // Create a large host array
        std::vector<T> host_source_data(large_array_size);
        std::iota(host_source_data.begin(), host_source_data.end(), static_cast<T>(1));
        
        // Create a device array and copy data to it
        CuArray2D<T> device_source_array(rows, cols);
        device_source_array.set(host_source_data.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Write the device array to a file using the stream method
        std::cout << "Writing device array to file: " << test_filename << std::endl;
        std::ofstream outfile(test_filename, std::ios::binary);
        if (!outfile) throw std::runtime_error("Could not open file for writing.");
        device_source_array.get(outfile);
        outfile.close();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Create a second device array and read data from the file
        std::cout << "Reading from file into new device array..." << std::endl;
        CuArray2D<T> device_dest_array(rows, cols);
        std::ifstream infile(test_filename, std::ios::binary);
        if (!infile) throw std::runtime_error("Could not open file for reading.");
        device_dest_array.set(infile);
        infile.close();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Copy the second device array back to host memory for verification
        std::vector<T> host_dest_data(large_array_size);
        device_dest_array.get(host_dest_data.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Verify that the data is identical
        verifyVectors(host_source_data, host_dest_data, "File I/O Test");
        
        // Clean up the test file
        std::remove(test_filename.c_str());

    } catch (const std::runtime_error& e) {
        std::cerr << "File I/O Test Failed: " << e.what() << std::endl;
        std::remove(test_filename.c_str());
    }
}

int main() {
    // Check for CUDA device
    int deviceCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Starting thorough testing of CudaArray classes..." << std::endl;

    // Run tests for different data types
    runTests<int>();
    std::cout << "\n========================================\n" << std::endl;
    runTests<float>();
    std::cout << "\n========================================\n" << std::endl;
    runTests<double>();


    // Run file I/O tests
    runFileIOTests<int>();
    std::cout << "\n========================================\n" << std::endl;
    runFileIOTests<float>();
    std::cout << "\n========================================\n" << std::endl;
    runFileIOTests<double>();
    
    std::cout << "\nAll tests complete." << std::endl;
    
    CHECK_CUDA_ERROR(cudaDeviceReset());
    return 0;
}




