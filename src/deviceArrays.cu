/**
 * @file deviceArrays.cu
 * @brief Templated classes for managing 1D and 2D arrays on a CUDA device in column-major order.
 */
#include "deviceArrays.h"
#include <vector>
#include <numeric>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <algorithm>

// --- CuFileHelper Definitions ---
template <typename T>
CuFileHelper<T>::CuFileHelper(size_t rows, size_t cols)
    : _totalCols(cols),
    _colsProcessed(0),
    _maxColsPerChunk(std::clamp(size_t((32ull * 1024ull * 1024ull) / (rows * sizeof(T))), size_t(1), size_t(cols))),
    _hostBuffer(_maxColsPerChunk * rows),    
    _rows(rows) {}

template <typename T>
CuFileHelper<T>::~CuFileHelper() = default;

template <typename T>
bool CuFileHelper<T>::hasNext() const {
    return _colsProcessed < _totalCols;
}

template <typename T>
size_t CuFileHelper<T>::getNextChunkColNumber() const {
    return std::min(_maxColsPerChunk, _totalCols - _colsProcessed);
}

template <typename T>
T* CuFileHelper<T>::getHostBuffer() {
    return _hostBuffer.data();
}

template <typename T>
void CuFileHelper<T>::updateProgress() {
    _colsProcessed += getNextChunkColNumber();
}

template <typename T>
size_t CuFileHelper<T>::getColsProcessed() const {
    return _colsProcessed;
}

// --- SetFromFile Definitions ---
template <typename T>
SetFromFile<T>::SetFromFile(size_t rows, size_t cols, std::istream& input_stream)
    : CuFileHelper<T>(rows, cols), _input_stream(input_stream) {}

template <typename T>
void SetFromFile<T>::readNextChunk() {
    size_t current_chunk_bytes = this->getNextChunkColNumber() * this->_rows * sizeof(T);
    if (current_chunk_bytes > 0) {
        this->_input_stream.read(reinterpret_cast<char*>(this->_hostBuffer.data()), current_chunk_bytes);
        if (!this->_input_stream) throw std::runtime_error("Stream read error or premature end of stream.");
    }
}

// --- GetToFile Definitions ---
template <typename T>
GetToFile<T>::GetToFile(size_t rows, size_t cols, std::ostream& output_stream)
    : CuFileHelper<T>(rows, cols), _output_stream(output_stream) {}

template <typename T>
void GetToFile<T>::writeNextChunkToFile() {
    size_t current_chunk_bytes = this->getNextChunkColNumber() * this->_rows * sizeof(T);
    if (current_chunk_bytes > 0) {
        this->_output_stream.write(reinterpret_cast<const char*>(this->_hostBuffer.data()), current_chunk_bytes);
        if (!this->_output_stream) throw std::runtime_error("Stream write error.");
    }
}

// --- CuArray Definitions ---
void cudaFreeDeleter(void* ptr) {
    if (ptr) cudaFree(ptr);
}

template <typename T>
CuArray<T>::CuArray(size_t rows, size_t cols, size_t ld)
    : _rows(rows), _cols(cols), _ld(ld){}

template <typename T>
CuArray<T>::~CuArray() = default;

template <typename T>
T* CuArray<T>::data() { return static_cast<T*>(_ptr.get()); }

template <typename T>
const T* CuArray<T>::data() const { return static_cast<const T*>(_ptr.get()); }

template <typename T>
size_t CuArray<T>::getLD() const { return _ld; }

template <typename T>
std::shared_ptr<void> CuArray<T>::getPtr() const{ return _ptr; }

// --- CuArray2D Definitions ---
template <typename T>
CuArray2D<T>::CuArray2D(size_t rows, size_t cols): CuArray<T>(rows, cols, 0) {
    void* rawPtr = nullptr;
    size_t pitch = 0;
    cudaError_t err = cudaMallocPitch(&rawPtr, &pitch, rows * sizeof(T), cols);
    if (err != cudaSuccess)
        throw std::runtime_error("cudaMallocPitch failed");

    this->_ptr = std::shared_ptr<void>(rawPtr, cudaFreeDeleter);
    this->_ld = pitch / sizeof(T);
}

template <typename T>
CuArray2D<T>::CuArray2D(const CuArray2D<T>& superArray, size_t startRow, size_t startCol, size_t height, size_t width)
    : CuArray<T>(height, width, superArray.getLD()) {
    size_t offset = startCol * superArray.getLD() + startRow;
    this->_ptr = std::shared_ptr<void>(
        superArray._ptr,
        static_cast<void*>(static_cast<char*>(superArray._ptr.get()) + offset * sizeof(T))
    );
}

template <typename T>
size_t CuArray2D<T>::size() const {
    return this->_rows * this->_cols;
}

template <typename T>
size_t CuArray2D<T>::bytes() const {
    return this->_cols * this->_ld * sizeof(T);
}

template <typename T>
void CuArray2D<T>::set(const T* src, cudaStream_t stream) {
    cudaMemcpy2DAsync(
        this->_ptr.get(), this->_ld * sizeof(T),
        src, this->_rows * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyHostToDevice, stream
    );
}

template <typename T>
void CuArray2D<T>::get(T* dst, cudaStream_t stream) const {
    cudaMemcpy2DAsync(
        dst, this->_rows * sizeof(T),
        this->_ptr.get(), this->_ld * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToHost, stream
    );
}

template <typename T>
void CuArray2D<T>::set(const CuArray<T>& src, cudaStream_t stream) {
    cudaMemcpy2DAsync(
        this->_ptr.get(), this->_ld * sizeof(T),
        src.data(), src.getLD() * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToDevice, stream
    );
}

template <typename T>
void CuArray2D<T>::get(CuArray<T>& dst, cudaStream_t cuStream) const {
    cudaMemcpy2DAsync(
        dst.data(), dst.getLD() * sizeof(T),
        this->_ptr.get(), this->_ld * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToDevice, cuStream
    );
}

template <typename T>
void CuArray2D<T>::set(std::istream& input_stream, cudaStream_t cuStream) {

    SetFromFile<T> helper(this->_rows, this->_cols, input_stream);

    while (helper.hasNext()) {
        helper.readNextChunk();
        CuArray2D<T> subArray(
            *this,
            0,
            helper.getColsProcessed(),
            this->_rows,
            helper.getNextChunkColNumber()
        );

        subArray.set(helper.getHostBuffer(), cuStream);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());//TODO: this might be avoidable with multi threading

        helper.updateProgress();
    }
}

template <typename T>
void CuArray2D<T>::get(std::ostream& output_stream, cudaStream_t stream) const {

    GetToFile<T> helper(this->_rows, this->_cols, output_stream);

    while (helper.hasNext()) {
        CuArray2D<T> subArray(
            *this,
            0,
            helper.getColsProcessed(),
            this->_rows,
            helper.getNextChunkColNumber()
        );

        subArray.get(helper.getHostBuffer(), stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());//TODO: this might be avoidable with multi threading

        helper.writeNextChunkToFile();
        helper.updateProgress();
    }
}

// --- CuArray1D Definitions ---
template <typename T>
CuArray1D<T>::CuArray1D(size_t length)
    : CuArray<T>(1, length, 1) {
    void* rawPtr = nullptr;
    cudaMalloc(&rawPtr, length * sizeof(T));
    this->_ptr = std::shared_ptr<void>(rawPtr, cudaFreeDeleter);
}

template <typename T>
CuArray1D<T>::CuArray1D(const CuArray1D<T>& superArray, size_t offset, size_t length, size_t stride)
    : CuArray<T>(1, length, stride * superArray.getLD()) {
    this->_ptr = std::shared_ptr<void>(
        superArray._ptr,
        static_cast<void*>(static_cast<char*>(superArray._ptr.get()) + offset * superArray.getLD() * sizeof(T))
    );
}

template <typename T>
CuArray1D<T>::CuArray1D(const CuArray2D<T>& extractFrom, int index, IndexType indexType):
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

template <typename T>
size_t CuArray1D<T>::size() const {
    return this->_cols;
}

template <typename T>
size_t CuArray1D<T>::bytes() const {
    return this->_cols * this->_ld * sizeof(T);
}

template <typename T>
void CuArray1D<T>::set(const T* hostData, cudaStream_t stream) {
    if (this->_ld == 1) cudaMemcpyAsync(this->_ptr.get(), hostData, bytes(), cudaMemcpyHostToDevice, stream);
    else cudaMemcpy2DAsync(
            this->_ptr.get(), this->_ld * sizeof(T),
            hostData, sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyHostToDevice, stream
        );
}

template <typename T>
void CuArray1D<T>::get(T* hostData, cudaStream_t stream) const {
    if (this->_ld == 1)
        cudaMemcpyAsync(hostData, this->_ptr.get(), bytes(), cudaMemcpyDeviceToHost, stream);
    else cudaMemcpy2DAsync(
            hostData, sizeof(T),
            this->_ptr.get(), this->_ld * sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyDeviceToHost, stream
        );
}

template <typename T>
void CuArray1D<T>::set(const CuArray<T>& src, cudaStream_t stream) {
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

template <typename T>
void CuArray1D<T>::get(CuArray<T>& dst, cudaStream_t stream) const {
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

template <typename T>
void CuArray1D<T>::set(std::istream& input_stream, cudaStream_t stream) {
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

template <typename T>
void CuArray1D<T>::get(std::ostream& output_stream, cudaStream_t stream) const {
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


// --- Helper Functions and Macros Definitions ---
void checkCudaErrors(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void verifyVectors(const std::vector<T>& expected, const std::vector<T>& result, const std::string& test_name) {
    if (expected == result) {
        std::cout << "✅ " << test_name << " successful." << std::endl;
    } else {
        std::cout << "❌ " << test_name << " failed." << std::endl;
        std::cerr << "Expected size: " << expected.size() << ", Result size: " << result.size() << std::endl;
    }
}

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
        
        CuArray2D<T> device_array(rows, cols);
        device_array.set(host_source_data.data());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Write the device array to a file using the stream method
        std::cout << "Writing device array to file: " << test_filename << std::endl;
        std::ofstream outfile(test_filename, std::ios::binary);
        if (!outfile) throw std::runtime_error("Could not open file for writing.");

        device_array.get(outfile);
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

void checkForDevice(){
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "CUDA device count: " << deviceCount << std::endl;
}

void multiTest(){

    checkForDevice();

    std::cout << "Starting thorough testing of CudaArray classes..." << std::endl;
    // Run tests for different data types
    runTests<int>();
    std::cout << "\n========================================\n" << std::endl;
    runTests<float>();
    std::cout << "\n========================================\n" << std::endl;
    runFileIOTests<int>();
    std::cout << "\n========================================\n" << std::endl;
    runFileIOTests<float>();
}

// Explicit template instantiation to avoid linker errors.
template class CuArray<int>;
template class CuArray<float>;
template class CuArray2D<int>;
template class CuArray2D<float>;
template class CuArray1D<int>;
template class CuArray1D<float>;
template class CuFileHelper<int>;
template class CuFileHelper<float>;
template class SetFromFile<int>;
template class SetFromFile<float>;
template class GetToFile<int>;
template class GetToFile<float>;
template void verifyVectors<int>(const std::vector<int>&, const std::vector<int>&, const std::string&);
template void verifyVectors<float>(const std::vector<float>&, const std::vector<float>&, const std::string&);
template void runTests<int>();
template void runTests<float>();
template void runFileIOTests<int>();
template void runFileIOTests<float>();
template std::ostream& operator<< <int>(std::ostream&, const CuArray1D<int>&);
template std::istream& operator>> <int>(std::istream&, CuArray1D<int>&);
template std::ostream& operator<< <float>(std::ostream&, const CuArray1D<float>&);
template std::istream& operator>> <float>(std::istream&, CuArray1D<float>&);
template std::ostream& operator<< <int>(std::ostream&, const CuArray2D<int>&);
template std::istream& operator>> <int>(std::istream&, CuArray2D<int>&);
template std::ostream& operator<< <float>(std::ostream&, const CuArray2D<float>&);
template std::istream& operator>> <float>(std::istream&, CuArray2D<float>&);
