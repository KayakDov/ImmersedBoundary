// --- deviceArrays.h ---
// This file declares the classes and functions used in deviceArrays.cu.
// It is included by main.cu to let the compiler know what exists.
#ifndef DEVICEARRAYS_H
#define DEVICEARRAYS_H

#include <vector> // For std::vector
#include <memory> // For std::shared_ptr
#include <stdexcept> // For std::runtime_error
#include <cuda_runtime.h> // For CUDA runtime API
#include <fstream> // For file I/O
#include <string> // For std::string
#include <cublas_v2.h> // For cuBLAS

template <typename T> class GpuArray;
template <typename T> class Vec;
template <typename T> class Mat;
template <typename T> class StreamHelper;
template <typename T> class StreamSet;
template <typename T> class StreamGet;
template <typename T> class Singleton;
#include <curand_kernel.h>


void checkCudaErrors(cudaError_t err, const char* file, int line);
#define CHECK_CUDA_ERROR(err) checkCudaErrors(err, __FILE__, __LINE__)


inline void cudaFreeDeleter(void* ptr) {
    if (ptr) cudaFree(ptr);
}

template <typename T>
class StreamHelper {
public:
    const size_t _totalCols;
    const size_t _maxColsPerChunk; 
    const size_t _rows;
protected:    
    size_t _colsProcessed;
    std::vector<T> _hostBuffer;
public:
    
    StreamHelper(size_t rows, size_t cols);
    virtual ~StreamHelper();
    [[nodiscard]] bool hasNext() const;
    [[nodiscard]] size_t getChunkWidth() const;
    void updateProgress();
    [[nodiscard]] size_t getColsProcessed() const;
    std::vector<T>& getBuffer();
};

template <typename T>
class StreamSet : public StreamHelper<T> {
private:
    std::istream& _input_stream;
public:
    StreamSet(size_t rows, size_t cols, std::istream& input_stream);
    void readChunk(bool isText);
};

template <typename T>
class StreamGet : public StreamHelper<T> {
private:
    std::ostream& _output_stream;
public:
    StreamGet(size_t rows, size_t cols, std::ostream& output_stream);
    void writeChunk(bool isText);
};


enum class IndexType {
    Row,
    Column
};

class Handle {
public:
    cublasHandle_t handle;
    cudaStream_t stream;
    Handle();
    
    explicit Handle(cudaStream_t user_stream);

    static Handle* _get_or_create_handle(Handle* handle, std::unique_ptr<Handle>& out_ptr_unique);

    ~Handle();

private:
    bool isOwner = false; // Flag to indicate if the class owns the stream and should destroy it.
};



template <typename T>
class GpuArray {
public:
    const size_t _rows;
    const size_t _cols;
protected:
    std::shared_ptr<void> _ptr;
    size_t _ld;
    GpuArray(size_t rows, size_t cols, size_t ld);

    Mat<T>* _get_or_create_target(size_t rows, size_t cols, Mat<T>* result, std::unique_ptr<Mat<T>>& out_ptr_unique) const;
    Vec<T>* _get_or_create_target(size_t size, Vec<T>* result, std::unique_ptr<Vec<T>>& out_ptr_unique) const;
    Singleton<T>* _get_or_create_target(Singleton<T>* result, std::unique_ptr<Singleton<T>>& out_ptr_unique) const;

    virtual void mult(const GpuArray<T>& other, GpuArray<T>* result, Handle* handle, Singleton<T>* alpha, Singleton<T>* beta, bool transposeA, bool transposeB) const;
    
public:
    virtual ~GpuArray();
    [[nodiscard]] virtual size_t size() const = 0;
    [[nodiscard]] virtual size_t bytes() const = 0;
    virtual void set(const T* hostData, cudaStream_t stream) = 0;
    virtual void get(T* hostData, cudaStream_t stream) const = 0;
    virtual void set(const GpuArray<T>& src, cudaStream_t stream ) = 0;
    virtual void get(GpuArray<T>& dst, cudaStream_t stream) const = 0;
    virtual void set(std::istream& input_stream, bool isText, bool isColMjr, cudaStream_t stream) = 0;
    virtual void get(std::ostream& output_stream, bool isText, bool isColMjr, cudaStream_t stream) const = 0;
    T* data();
    const T* data() const;
    [[nodiscard]] size_t getLD() const;
    [[nodiscard]] std::shared_ptr<void> getPtr();

    GpuArray<T>& operator=(const GpuArray<T>& other) {
        if (this != &other) {
            if (_rows != other._rows || _cols != other._cols)
                throw std::runtime_error("gpuArray assignment: row/col dimensions do not match");
            
            _ptr = other._ptr;
            _ld = other._ld;
        }
        return *this;
    }


    GpuArray(const GpuArray<T>& other) = default;
    
};

template <typename T>
class Mat : public GpuArray<T> {
    using GpuArray<T>::mult;
    
public:
    Mat(size_t rows, size_t cols);
    Mat(const Mat<T>& superArray, size_t startRow, size_t startCol, size_t height, size_t width);
    [[nodiscard]] size_t size() const override;
    [[nodiscard]] size_t bytes() const override;
    void set(const T* src, cudaStream_t stream) override;
    void get(T* dst, cudaStream_t stream) const override;
    void set(const GpuArray<T>& src, cudaStream_t stream) override;
    void get(GpuArray<T>& dst, cudaStream_t stream) const override;
    void set(std::istream& input_stream, bool isText, bool readColMjr, cudaStream_t stream) override;
    void get(std::ostream& output_stream, bool isText, bool printColMjr, cudaStream_t stream) const override;
    
    Mat<T> mult(const Mat<T>& other, Mat<T>* result = nullptr, Handle* handle = nullptr, Singleton<T>* alpha = nullptr, Singleton<T>* beta = nullptr, bool transposeA = false, bool transposeB = false) const;

    Vec<T> mult(const Vec<T>& other, Vec<T>* result = nullptr, Handle* handle = nullptr, Singleton<T>* alpha = nullptr, Singleton<T>* beta = nullptr, bool transpose = false) const;    
    
    Vec<T> operator*(const Vec<T>& other) const;    

    Mat<T> operator*(const Mat<T>& other) const;    
    
    Mat<T> plus(const Mat<T>& x, Mat<T>* result = nullptr, Singleton<T>* alpha = nullptr, Singleton<T>* beta = nullptr, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr);
    
    Mat<T> minus(const Mat<T>& x, Mat<T>* result = nullptr, Singleton<T>* alpha = nullptr, Singleton<T>* beta = nullptr, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr);    

    void mult(const Singleton<T>& alpha, Handle* handle = nullptr);

    /**
     * Multiply a sparse diagonal matrix (packed diagonals) with a 1D vector.
     *
     * This matrix must have at most 64 rows, representing a sparse matrix with up to 64 non zero diagonals.
     *
     * this <- alpha * A * x + beta * this
     *
     * @param diags Array of diagonal indices (negative=sub-diagonal, 0=main, positive=super-diagonal).  Each row of this matrix is treated as a diagonal with the coresponding index.
     * @param x Input vector.
     * @param result Optional pointer to store the result. If nullptr, a new gpuArray1D is returned.
     * @param handle Optional Handle for managing CUDA streams/context.
     * @param alpha Optional scalar multiplier.
     * @param beta Optional scalar multiplier.
     * @return A new gpuArray1D containing the result of the multiplication.
     */
    Vec<T> diagMult(const Vec<int>& diags, const Vec<T>& x, Vec<T>* result = nullptr, Handle* handle = nullptr, const Singleton<T>* alpha = nullptr, const Singleton<T>* beta = nullptr) const;


    void transpose(Mat<T>& result, Handle* handle = nullptr) const;
    void transpose(Handle* handle = nullptr, Mat<T>* preAlocatedMem = nullptr);
    

};

template <typename T>
class Vec : public GpuArray<T> {
    using GpuArray<T>::mult;
protected:
    Vec(size_t cols, size_t rows, size_t ld);
public:
    explicit Vec(size_t length);
    Vec(const Vec<T>& superArray, size_t offset, size_t length, size_t stride = 1);
    Vec(const Mat<T>& extractFrom, int index, IndexType indexType);
    [[nodiscard]] size_t size() const override;
    [[nodiscard]] size_t bytes() const override;
    void set(const T* hostData, cudaStream_t stream) override;
    void get(T* hostData, cudaStream_t stream) const override;
    void set(const GpuArray<T>& src, cudaStream_t stream) override;
    void get(GpuArray<T>& dst, cudaStream_t stream) const override;
    void set(std::istream& input_stream, bool isText, bool isColMjr, cudaStream_t stream) override;
    void get(std::ostream& output_stream, bool isText, bool isColMjr, cudaStream_t stream) const override;

    Vec<T> mult(const Mat<T>& other, Vec<T>* result = nullptr, Handle* handle = nullptr, Singleton<T>* alpha = nullptr, Singleton<T>* beta = nullptr, bool transpose = false) const;

    T mult(const Vec<T>& other, Singleton<T>* result = nullptr, Handle* handle = nullptr) const;    

    Vec<T> operator*(const Mat<T>& other) const;    
    T operator*(const Vec<T>& other) const;
    

    void add(const Vec<T>& x, Singleton<T>* alpha, Handle* handle);
    void sub(const Vec<T>& x, Singleton<T>* alpha, Handle* handle);

    void mult(const Singleton<T>& alpha, Handle* handle = nullptr);

    void fillRandom(Handle* handle = nullptr);

    void EBEPow(const Singleton<T>& t, const Singleton<T>& n, cudaStream_t stream);

    void setSum(const Vec& a, const Vec& B, Singleton<T>* alpha, Singleton<T>* beta, Handle* handle);

};

template <typename T>
class Singleton : public Vec<T> {
    
public:
    static const Singleton<T> ONE, ZERO, MINUS_ONE;

    using Vec<T>::get;
    using Vec<T>::set;

    Singleton();
    Singleton(const Vec<T>& superVector, int index);
    Singleton(const Mat<T>& superMatrix, int row, int col);
    explicit Singleton(T val, Handle *hand = nullptr);

    T get(cudaStream_t stream = nullptr) const;
    void set(T val, cudaStream_t stream) override;
};
/**
 * @brief Input formatted data (space-separated values) into gpuArray1D<T> from a stream.
 * @tparam T Element type
 * @param is Input stream
 * @param arr Array to fill
 * @return Input stream
 */
template <typename T>
std::istream& operator>>(std::istream& is, Vec<T>& arr) {

    std::vector<T> hostData(arr.size());
    for (size_t i = 0; i < hostData.size(); ++i) {
        is >> hostData[i];
        if (!is) {
            is.setstate(std::ios::badbit);
            break;
        }
    }
    arr.set(hostData.data());
    return is;
}

/**
 * @brief Prints a gpuArray1D<T> to a stream.
 * @tparam T Element type
 * @param os Output stream
 * @param arr The gpuArray1D to print
 * @return Output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const Vec<T>& arr) {
    std::vector<T> hostData(arr.size());
    arr.get(hostData.data());

    for (size_t i = 0; i < hostData.size(); ++i) {
        os << hostData[i];
        if (i + 1 < hostData.size()) {
            os << " ";
        }
    }
    os << "\n";
    return os;
}

/**
 * @brief Prints a gpuArray2D<T> to a stream with improved formatting.
 *
 * This operator assumes a column-major memory layout and prints the matrix
 * row by row for a more readable output. It uses iomanip to format the
 * floating-point numbers to a fixed precision and set width.
 *
 * @tparam T Element type
 * @param os Output stream
 * @param arr The gpuArray2D to print
 * @return Output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const Mat<T>& arr) {
    cudaDeviceSynchronize();
    std::vector<T> hostData(arr.size());
    arr.get(hostData.data());

    for (size_t r = 0; r < arr._rows; ++r) {
        for (size_t c = 0; c < arr._cols; ++c) {
            // Contiguous column-major access
            os << hostData[c * arr._rows + r] << " ";
        }
        os << "\n";
    }
    return os;
}


/**
 * @brief Reads formatted data (column by column) into gpuArray2D<T> from a stream.
 *
 * This operator assumes the input stream is formatted in column-major order,
 * meaning it will read all elements of the first column, then the second, and so on.
 *
 * @tparam T Element type
 * @param is Input stream
 * @param arr Array to fill
 * @return Input stream
 */
template <typename T>
std::istream& operator>>(std::istream& is, Mat<T>& arr) {
    std::vector<T> hostData(arr.size());
    size_t ld = arr.getLD();

    for (size_t c = 0; c < arr._cols; ++c) {
        for (size_t r = 0; r < arr._rows; ++r) {
            // Reading data in column-major order from the stream
            is >> hostData[c * ld + r];
            if (!is) {
                is.setstate(std::ios::badbit);
                return is;
            }
        }
    }

    arr.set(hostData.data());
    return is;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const GpuArray<T>& arr) {
    
    if (auto ptr1d = dynamic_cast<const Vec<T>*>(&arr)) return os << *ptr1d;
    else if (auto ptr2d = dynamic_cast<const Mat<T>*>(&arr)) return os << *ptr2d;
    else throw std::runtime_error("Unable to detect the type of array, 1d or 2d.");
    return os;
}


#endif // DEVICEARRAYS_H