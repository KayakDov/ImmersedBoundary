// --- deviceArrays.h ---
// This file declares the classes and functions used in deviceArrays.cu.
// It is included by main.cu to let the compiler know what exists.
#ifndef DEVICEARRAYS_H
#define DEVICEARRAYS_H

#include <vector> // For std::vector
#include <iostream> // For std::cerr
#include <memory> // For std::shared_ptr
#include <stdexcept> // For std::runtime_error
#include <cuda_runtime.h> // For CUDA runtime API
#include <cuda_runtime_api.h> // For cudaFree
#include <fstream> // For file I/O
#include <string> // For std::string
#include <cublas_v2.h> // For cuBLAS
#include <curand_kernel.h> // For cuRAND
#include <iomanip> // For formatted output

template <typename T> class CuArray;
template <typename T> class CuArray1D;
template <typename T> class CuArray2D;
template <typename T> class CuFileHelper;
template <typename T> class SetFromFile;
template <typename T> class GetToFile;



void checkCudaErrors(cudaError_t err, const char* file, int line);
#define CHECK_CUDA_ERROR(err) checkCudaErrors(err, __FILE__, __LINE__)


inline void cudaFreeDeleter(void* ptr) {
    if (ptr) cudaFree(ptr);
}

template <typename T>
class CuFileHelper {
public:
    const size_t _totalCols;
    const size_t _maxColsPerChunk; 
    const size_t _rows;
protected:    
    size_t _colsProcessed;
    std::vector<T> _hostBuffer;
public:
    
    CuFileHelper(size_t rows, size_t cols);
    virtual ~CuFileHelper();
    bool hasNext() const;
    size_t getNextChunkColNumber() const;
    T* getHostBuffer();
    void updateProgress();
    size_t getColsProcessed() const;
};

template <typename T>
class SetFromFile : public CuFileHelper<T> {
private:
    std::istream& _input_stream;
public:
    SetFromFile(size_t rows, size_t cols, std::istream& input_stream);
    void readNextChunk();
};

template <typename T>
class GetToFile : public CuFileHelper<T> {
private:
    std::ostream& _output_stream;
public:
    GetToFile(size_t rows, size_t cols, std::ostream& output_stream);
    void writeNextChunkToFile();
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
    ~Handle();
};


template <typename T>
class CuArray {
public:
    const size_t _rows;
    const size_t _cols;
protected:
    std::shared_ptr<void> _ptr;
    size_t _ld;
    CuArray(size_t rows, size_t cols, size_t ld);

    void mult(const CuArray<float>& other, CuArray<float>* result, Handle* handle = nullptr, float alpha = 1.0f, float beta = 0.0f, bool transposeA = false, bool transposeB = false) const;
    void mult(const CuArray<double>& other, CuArray<double>* result, Handle* handle = nullptr, double alpha = 1.0, double beta = 0.0, bool transposeA = false, bool transposeB = false) const;
public:
    virtual ~CuArray();
    virtual size_t size() const = 0;
    virtual size_t bytes() const = 0;
    virtual void set(const T* hostData, cudaStream_t stream = 0) = 0;
    virtual void get(T* hostData, cudaStream_t stream = 0) const = 0;
    virtual void set(const CuArray<T>& src, cudaStream_t stream = 0) = 0;
    virtual void get(CuArray<T>& dst, cudaStream_t stream = 0) const = 0;
    virtual void set(std::istream& input_stream, cudaStream_t stream = 0) = 0;
    virtual void get(std::ostream& output_stream, cudaStream_t stream = 0) const = 0;
    T* data();
    const T* data() const;
    size_t getLD() const;
    std::shared_ptr<void> getPtr() const;

    CuArray<T>& operator=(const CuArray<T>& other) {
        if (this != &other) {
            if (_rows != other._rows || _cols != other._cols)
                throw std::runtime_error("CuArray assignment: row/col dimensions do not match");
            
            _ptr = other._ptr;
            _ld = other._ld;
        }
        return *this;
    }


    CuArray(const CuArray<T>& other) = default;
    
};

template <typename T>
class CuArray2D : public CuArray<T> {
    using CuArray<T>::mult;
private:
    void _scale_impl(T alpha, Handle* handle);
    
public:
    CuArray2D(size_t rows, size_t cols);
    CuArray2D(const CuArray2D<T>& superArray, size_t startRow, size_t startCol, size_t height, size_t width);
    size_t size() const override;
    size_t bytes() const override;
    void set(const T* src, cudaStream_t stream = 0) override;
    void get(T* dst, cudaStream_t stream = 0) const override;
    void set(const CuArray<T>& src, cudaStream_t stream = 0) override;
    void get(CuArray<T>& dst, cudaStream_t stream = 0) const override;
    void set(std::istream& input_stream, cudaStream_t stream = 0) override;
    void get(std::ostream& output_stream, cudaStream_t stream = 0) const override;
    
    CuArray2D<float> mult(const CuArray2D<float>& other, CuArray2D<float>* result = nullptr, Handle* handle = nullptr, float alpha = 1.0f, float beta = 0.0f, bool transposeA = false, bool transposeB = false) const;
    CuArray2D<double> mult(const CuArray2D<double>& other, CuArray2D<double>* result = nullptr, Handle* handle = nullptr, double alpha = 1.0, double beta = 0.0, bool transposeA = false, bool transposeB = false) const;

    CuArray1D<float> mult(const CuArray1D<float>& other, CuArray1D<float>* result = nullptr, Handle* handle = nullptr, float alpha = 1.0f, float beta = 0.0f, bool transpose = false) const;
    CuArray1D<double> mult(const CuArray1D<double>& other, CuArray1D<double>* result = nullptr, Handle* handle = nullptr, double alpha = 1.0, double beta = 0.0, bool transpose = false) const;
    
    CuArray1D<float> operator*(const CuArray1D<float>& other) const;
    CuArray1D<double> operator*(const CuArray1D<double>& other) const;

    CuArray2D<float> operator*(const CuArray2D<float>& other) const;
    CuArray2D<double> operator*(const CuArray2D<double>& other) const;
    
    CuArray2D<float> plus(const CuArray2D<float>& x, CuArray2D<float>* result = nullptr, float alpha = 1.0f, float beta = 1.0f, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr);
    CuArray2D<double> plus(const CuArray2D<double>& x, CuArray2D<double>* result = nullptr, double alpha = 1.0, double beta = 1.0, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr);
    CuArray2D<float> minus(const CuArray2D<float>& x, CuArray2D<float>* result = nullptr, float alpha = 1.0f, float beta = 1.0f, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr);
    CuArray2D<double> minus(const CuArray2D<double>& x, CuArray2D<double>* result = nullptr, double alpha = 1.0, double beta = 1.0, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr);

    void mult(float alpha, Handle* handle = nullptr);
    void mult(double alpha, Handle* handle = nullptr);

    CuArray1D<float> bandedMult(const CuArray1D<float>& other, CuArray1D<float>* result = nullptr, Handle* handle = nullptr, float alpha = 1.0f, float beta = 0.0f, int kl = 0, int ku = 0, bool transpose = false) const;
    CuArray1D<double> bandedMult(const CuArray1D<double>& other, CuArray1D<double>* result = nullptr, Handle* handle = nullptr, double alpha = 1.0, double beta = 0.0, int kl = 0, int ku = 0, bool transpose = false) const;

    /**
     * Multiply a sparse diagonal matrix (packed diagonals) with a 1D vector.
     *
     * This matrix must have at most 64 rows, representing a sparse matrix with up to 64 non zero diagonals.
     * 
     * @param diags Array of diagonal indices (negative=sub-diagonal, 0=main, positive=super-diagonal).  Each row of this matrix is treated as a diagonal with the coresponding index.
     * @param x Input vector.
     * @param result Optional pointer to store the result. If nullptr, a new CuArray1D is returned.
     * @param handle Optional Handle for managing CUDA streams/context.
     * @param stride Stride for the input vector x.
     * @return A new CuArray1D containing the result of the multiplication.
     */
    CuArray1D<double> diagMult(const int* diags, const CuArray1D<double>& x, CuArray1D<double>* result = nullptr, Handle* handle = nullptr, const double alpha = 1.0, const double beta = 0.0) const;
    CuArray1D<float> diagMult(const int* diags, const CuArray1D<float>& x, CuArray1D<float>* result = nullptr, Handle* handle = nullptr, const float alpha = 1.0f, const float beta = 0.0f) const;

};

template <typename T>
class CuArray1D : public CuArray<T> {
    using CuArray<T>::mult;
public:
    explicit CuArray1D(size_t length);
    CuArray1D(const CuArray1D<T>& superArray, size_t offset, size_t length, size_t stride = 1);
    CuArray1D(const CuArray2D<T>& extractFrom, int index, IndexType indexType);
    size_t size() const override;
    size_t bytes() const override;
    void set(const T* hostData, cudaStream_t stream = 0) override;
    void get(T* hostData, cudaStream_t stream = 0) const override;
    void set(const CuArray<T>& src, cudaStream_t stream = 0) override;
    void get(CuArray<T>& dst, cudaStream_t stream = 0) const override;
    void set(std::istream& input_stream, cudaStream_t stream = 0) override;
    void get(std::ostream& output_stream, cudaStream_t stream = 0) const override;

    CuArray1D<float> mult(const CuArray2D<float>& other, CuArray1D<float>* result = nullptr, Handle* handle = nullptr, float alpha = 1.0f, float beta = 0.0f, bool transpose = false) const;
    CuArray1D<double> mult(const CuArray2D<double>& other, CuArray1D<double>* result = nullptr, Handle* handle = nullptr, double alpha = 1.0, double beta = 0.0, bool transpose = false) const;

    float mult(const CuArray1D<float>& other, Handle* handle = nullptr) const;
    double mult(const CuArray1D<double>& other, Handle* handle = nullptr) const;

    CuArray1D<float> operator*(const CuArray2D<float>& other) const;
    CuArray1D<double> operator*(const CuArray2D<double>& other) const;
    float operator*(const CuArray1D<float>& other) const;
    double operator*(const CuArray1D<double>& other) const;

    void add(const CuArray1D<float>& x, float alpha = 1.0f, Handle* handle = nullptr);
    void sub(const CuArray1D<float>& x, float alpha = 1.0f,Handle* handle = nullptr);

    void add(const CuArray1D<double>& x, double alpha = 1.0, Handle* handle = nullptr);
    void sub(const CuArray1D<double>& x, double alpha = 1.0, Handle* handle = nullptr);

    void mult(float alpha, Handle* handle = nullptr);
    void mult(double alpha, Handle* handle = nullptr);

    void fillRandom(Handle* handle = nullptr);
};


/**
 * @brief Input formatted data (space-separated values) into CuArray1D<T> from a stream.
 * @tparam T Element type
 * @param is Input stream
 * @param arr Array to fill
 * @return Input stream
 */
template <typename T>
std::istream& operator>>(std::istream& is, CuArray1D<T>& arr) {
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
 * @brief Prints a CuArray1D<T> to a stream.
 * @tparam T Element type
 * @param os Output stream
 * @param arr The CuArray1D to print
 * @return Output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const CuArray1D<T>& arr) {
    std::vector<T> hostData(arr.size());
    arr.get(hostData.data());
    
    os << std::fixed << std::setprecision(4);
    
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
 * @brief Prints a CuArray2D<T> to a stream with improved formatting.
 *
 * This operator assumes a column-major memory layout and prints the matrix
 * row by row for a more readable output. It uses iomanip to format the
 * floating-point numbers to a fixed precision and set width.
 *
 * @tparam T Element type
 * @param os Output stream
 * @param arr The CuArray2D to print
 * @return Output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const CuArray2D<T>& arr) {
    std::vector<T> hostData(arr.size());
    arr.get(hostData.data());

    // Set formatting for prettier output
    os << std::fixed << std::setprecision(4);

    for (size_t r = 0; r < arr._rows; ++r) {
        for (size_t c = 0; c < arr._cols; ++c) {
            // Contiguous column-major access
            os << std::setw(12) << hostData[c * arr._rows + r] << " ";
        }
        os << "\n";
    }
    return os;
}


/**
 * @brief Reads formatted data (column by column) into CuArray2D<T> from a stream.
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
std::istream& operator>>(std::istream& is, CuArray2D<T>& arr) {
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


#endif // DEVICEARRAYS_H