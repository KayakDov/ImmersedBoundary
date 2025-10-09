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
#include "deviceArraySupport.h"
#include <cusolverDn.h>


template <typename T> class GpuArray;
template <typename T> class Vec;
template <typename T> class Mat;
template <typename T> class Singleton;
template <typename T> class Tensor;
template <typename T> class SquareMat;
template <typename T> class BandedMat;

void checkCudaErrors(cudaError_t err, const char* file, int line);
#define CHECK_CUDA_ERROR(err) checkCudaErrors(err, __FILE__, __LINE__)


inline void cudaFreeDeleter(void* ptr) {
    if (ptr) cudaFree(ptr);
}

template <typename T>
class GpuArray {
public:
    const size_t _rows;
    const size_t _cols;
    const std::shared_ptr<T> _ptr;
    const size_t _ld;
protected:
    GpuArray(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> _ptr);

    Mat<T>* _get_or_create_target(size_t rows, size_t cols, Mat<T>* result, std::unique_ptr<Mat<T>>& out_ptr_unique) const;
    Vec<T>* _get_or_create_target(size_t length, Vec<T> *result, std::unique_ptr<Vec<T>> &out_ptr_unique, cudaStream_t stream) const;
    Singleton<T>* _get_or_create_target(Singleton<T> *result, std::unique_ptr<Singleton<T>> &out_ptr_unique, cudaStream_t stream) const;
    const Singleton<T>* _get_or_create_target(T defaultVal, Handle& hand, const Singleton<T>* result, std::unique_ptr<Singleton<T>>& out_ptr_unique) const;

    virtual void mult(const GpuArray<T>& other, GpuArray<T>* result, Handle* handle, const Singleton<T> *alpha, const Singleton<T> *beta, bool transposeA, bool transposeB) const;
    
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

    virtual void fill(T val, cudaStream_t stream);
    T* data();
    const T* data() const;
    [[nodiscard]] size_t getLD() const;

    GpuArray(const GpuArray<T>& other) = default;
    
};

template <typename T>
class Mat : public GpuArray<T> {
    using GpuArray<T>::mult;
protected:
    Mat(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> _ptr);
    
public:

    [[nodiscard]] size_t size() const override;
    [[nodiscard]] size_t bytes() const override;
    void set(const T* src, cudaStream_t stream) override;
    void get(T* dst, cudaStream_t stream) const override;
    void set(const GpuArray<T>& src, cudaStream_t stream) override;
    void get(GpuArray<T>& dst, cudaStream_t stream) const override;
    void set(std::istream& input_stream, bool isText, bool readColMjr, cudaStream_t stream) override;
    void get(std::ostream& output_stream, bool isText, bool printColMjr, cudaStream_t stream) const override;
    Singleton<T> get(size_t row, size_t col);
    
    Mat<T> mult(const Mat<T>& other, Mat<T>* result = nullptr, Handle* handle = nullptr, const Singleton<T> *alpha = nullptr, const
                Singleton<T> *beta = nullptr, bool transposeA = false, bool transposeB = false) const;

    virtual Vec<T> mult(const Vec<T>& other, Vec<T>* result = nullptr, Handle* handle = nullptr, const Singleton<T> *alpha = nullptr, const
                        Singleton<T> *beta = nullptr, bool transpose = false) const;

    virtual Vec<T> operator*(const Vec<T>& other) const;

    virtual Mat<T> operator*(const Mat<T>& other) const;

    virtual Mat<T> plus(const Mat<T>& x, Mat<T>* result = nullptr, const Singleton<T>* alpha = nullptr, const Singleton<T>* beta = nullptr, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr);

    virtual Mat<T> minus(const Mat<T>& x, Mat<T>* result = nullptr, const Singleton<T>* alpha = nullptr, const Singleton<T>* beta = nullptr, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr);

    virtual void mult(const Singleton<T>& alpha, Handle* handle = nullptr);

    virtual void transpose(Mat<T>& result, Handle* handle = nullptr) const;

    virtual void transpose(Handle* handle = nullptr, Mat<T>* preAlocatedMem = nullptr);

    static Mat<T> create(size_t rows, size_t cols);

    [[nodiscard]] virtual Mat<T> subMat(size_t startRow, size_t startCol, size_t height, size_t width) const;

    virtual Vec<T> col(size_t index);

    virtual Vec<T> row(size_t index);

    virtual void normalizeCols(size_t setRowTo1, Handle* handle = nullptr);
};

template <typename T>
class SquareMat : public Mat<T> {
private:
    SquareMat(size_t rowsCols, size_t ld, std::shared_ptr<T> _ptr);

public:
    static SquareMat<T> create(size_t rowsCols);

    BandedMat<T> mapDenseToBanded(const Vec<int32_t> &indices, Mat<T> *result = nullptr, Handle *handle = nullptr) const;

    void eigen(Vec<T> &eVals, SquareMat<T> *eVecs, Mat<T> *temp = nullptr, Handle *handle = nullptr) const;
};

template <typename T>
class Tensor final : public Mat<T> {
private:
    Tensor(size_t rows, size_t cols, size_t layers, size_t ld, std::shared_ptr<T> _ptr);
public:
    static Tensor<T> create(size_t rows, size_t cols, size_t layers, cudaStream_t stream);
    Mat<T> layer(size_t index);
    Vec<T> depth(size_t row, size_t col);
    Singleton<T> get(size_t row, size_t col, size_t layer);
};

template <typename T>
class Vec : public GpuArray<T> {
    using GpuArray<T>::mult;
private:
    friend Vec<T> Mat<T>::row(size_t index);
    friend Vec<T> Mat<T>::col(size_t index);
    friend Vec<T> Tensor<T>::depth(size_t row, size_t col);
protected:
    Vec(size_t cols, std::shared_ptr<T> _ptr, size_t stride);
public:

    static Vec<T> create(size_t length, cudaStream_t stream = nullptr);

    Vec<T> subVec(size_t offset, size_t length, size_t stride = 1) const;

    [[nodiscard]] size_t size() const override;
    [[nodiscard]] size_t bytes() const override;
    void set(const T* hostData, cudaStream_t stream) override;
    void get(T* hostData, cudaStream_t stream) const override;
    void set(const GpuArray<T>& src, cudaStream_t stream) override;
    void get(GpuArray<T>& dst, cudaStream_t stream) const override;
    void set(std::istream& input_stream, bool isText, bool isColMjr, cudaStream_t stream) override;
    void get(std::ostream& output_stream, bool isText, bool isColMjr, cudaStream_t stream) const override;

    void fill(T val, cudaStream_t stream) override;

    Singleton<T> get(size_t i);

    Vec<T> mult(const Mat<T>& other, Vec<T>* result = nullptr, Handle* handle = nullptr, const Singleton<T>* alpha = nullptr, const Singleton<T>* beta = nullptr, bool transpose = false) const;

    T mult(const Vec<T>& other, Singleton<T>* result = nullptr, Handle* handle = nullptr) const;    

    Vec<T> operator*(const Mat<T>& other) const;    
    T operator*(const Vec<T>& other) const;
    

    void add(const Vec<T>& x, const Singleton<T> *alpha, Handle* handle);
    void sub(const Vec<T>& x, const Singleton<T>* alpha, Handle* handle);

    void mult(const Singleton<T>& alpha, Handle* handle = nullptr);

    void fillRandom(Handle* handle = nullptr);

    void EBEPow(const Singleton<T>& t, const Singleton<T>& n, cudaStream_t stream);

    void setSum(const Vec& a, const Vec& B, const Singleton<T>* alpha, const Singleton<T>* beta, Handle* handle);

};


template<typename T>
class BandedMat final : public Mat<T> {
private:
    const Vec<int32_t> _indices;
protected:
    BandedMat(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> _ptr, const Vec<int32_t> &indices);
public:
    BandedMat(const Mat<T>& copyFrom, const Vec<int32_t>& indices);

    static BandedMat create(size_t numDiagonals, size_t cols, const Vec<int32_t> &indices);
    static BandedMat createFromDense(const SquareMat<T> &src, const Vec<int32_t> &indices, Mat<T> *result = nullptr, Handle *handle = nullptr);

    Vec<T> mult(const Vec<T>& other, Vec<T>* result = nullptr, Handle* handle = nullptr, const Singleton<T> *alpha = nullptr, const Singleton<T> *beta = nullptr, bool transpose = false) const;

    Mat<T> mult(const Mat<T>& other, Mat<T>* result = nullptr, Handle* handle = nullptr, const Singleton<T> *alpha = nullptr, const
                Singleton<T> *beta = nullptr, bool transposeA = false, bool transposeB = false) const;

    void getDense(SquareMat<T> dense, Handle *handle = nullptr) const;


    Mat<T> operator*(const Mat<T>& other) const override;

    Mat<T> plus(const Mat<T>& x, Mat<T>* result = nullptr, const Singleton<T>* alpha = nullptr, const Singleton<T>* beta = nullptr, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr) override;

    Mat<T> minus(const Mat<T>& x, Mat<T>* result = nullptr, const Singleton<T>* alpha = nullptr, const Singleton<T>* beta = nullptr, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr) override;

    void mult(const Singleton<T>& alpha, Handle* handle = nullptr) override;

    void transpose(Mat<T>& result, Handle* handle = nullptr) const override;
    void transpose(Handle* handle = nullptr, Mat<T>* preAlocatedMem = nullptr) override;

    static Mat<T> create(size_t rows, size_t cols);

    [[nodiscard]] Mat<T> subMat(size_t startRow, size_t startCol, size_t height, size_t width) const override;

    Vec<T> col(size_t index) override;
    Vec<T> row(size_t index) override;

    void normalizeCols(size_t setRowTo1, Handle* handle = nullptr) override;
};

template <typename T>
class Singleton final : public Vec<T> {
private:
    friend Singleton<T> Vec<T>::get(size_t i);
    friend Singleton<T> Mat<T>::get(size_t row, size_t col);
    friend Singleton<T> Tensor<T>::get(size_t row, size_t col, size_t layer);

    explicit Singleton(std::shared_ptr<T> ptr);
public:
    static const Singleton<T> ONE, ZERO, MINUS_ONE;

    using Vec<T>::get;
    using Vec<T>::set;

    static Singleton<T> create(cudaStream_t stream = 0);
    static Singleton<T> create(T val,cudaStream_t = 0);

    T get(cudaStream_t stream = nullptr) const;
    void set(T val, cudaStream_t stream);
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
    Handle hand;
    arr.get(hostData.data(), hand.stream);
    hand.synch();

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

    Handle hand;
    arr.get(hostData.data(), hand.stream);
    hand.synch();

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