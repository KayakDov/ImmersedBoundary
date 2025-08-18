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

template<>
void CuArray<float>::mult(
    const CuArray<float>& other,
    CuArray<float>* result,
    CublasHandle* handle,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB
) const {
    
    CublasHandle* h = handle ? handle : new CublasHandle();

    cublasSgemm(h->handle,
        transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows, other._cols, this->_cols,
        &alpha,
        this->data(), this->getLD(),
        other.data(), other.getLD(),
        &beta,
        result->data(), result->getLD());

    if (!handle){
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
         delete h;
    }
}

template<>
void CuArray<double>::mult(
    const CuArray<double>& other,
    CuArray<double>* result,
    CublasHandle* handle,
    double alpha,
    double beta,
    bool transposeA,
    bool transposeB
) const {
    CublasHandle* h = handle ? handle : new CublasHandle();
    
    cublasDgemm(h->handle,
        transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows, other._cols, this->_cols,
        &alpha,
        this->data(), this->getLD(),
        other.data(), other.getLD(),
        &beta,
        result->data(), result->getLD());
    
    if (!handle){
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
         delete h;
    }
}

template<>
void CuArray<int>::mult(
    const CuArray<float>& other,
    CuArray<float>* result,
    CublasHandle* handle,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB
) const {
    throw std::logic_error("Multiplication not supported for int");
}

template<>
void CuArray<int>::mult(
    const CuArray<double>& other,
    CuArray<double>* result,
    CublasHandle* handle,
    double alpha,
    double beta,
    bool transposeA,
    bool transposeB
) const {
    throw std::logic_error("Multiplication not supported for int");
}


// --- Helper Functions and Macros Definitions ---
void checkCudaErrors(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}


// CublasHandle Definitions
CublasHandle::CublasHandle() {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle");
    }
}

CublasHandle::~CublasHandle() {
    cublasDestroy(handle);
}

cudaStream_t CublasHandle::getStream() const {
    cudaStream_t stream;
    if (cublasGetStream(handle, &stream) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to get cuBLAS stream");
    }
    return stream;
}

// Disallow mixed-type multiplication for float/double
template<>
void CuArray<float>::mult(
    const CuArray<double>& other,
    CuArray<double>* result,
    CublasHandle* handle,
    double alpha,
    double beta,
    bool transposeA,
    bool transposeB
) const {
    throw std::logic_error("Multiplication not supported between float and double");
}

template<>
void CuArray<double>::mult(
    const CuArray<float>& other,
    CuArray<float>* result,
    CublasHandle* handle,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB
) const {
    throw std::logic_error("Multiplication not supported between double and float");
}


// Explicit template instantiations to satisfy the linker

// CuFileHelper
template class CuFileHelper<int>;
template class CuFileHelper<float>;
template class CuFileHelper<double>;

// SetFromFile
template class SetFromFile<int>;
template class SetFromFile<float>;
template class SetFromFile<double>;

// GetToFile
template class GetToFile<int>;
template class GetToFile<float>;
template class GetToFile<double>;

// CuArray
template class CuArray<int>;
template class CuArray<float>;
template class CuArray<double>;

// If you also use mult() with mixed types (float vs double or int), explicitly instantiate those specializations:
// For example:
template void CuArray<float>::mult(
    const CuArray<float>&, CuArray<float>*, CublasHandle*, float, float, bool, bool
) const;

template void CuArray<double>::mult(
    const CuArray<double>&, CuArray<double>*, CublasHandle*, double, double, bool, bool
) const;