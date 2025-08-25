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
StreamHelper<T>::StreamHelper(size_t rows, size_t cols)
    : _totalCols(cols),
    _colsProcessed(0),
    _maxColsPerChunk(std::clamp(size_t((32ull * 1024ull * 1024ull) / (rows * sizeof(T))), size_t(1), size_t(cols))),
    _hostBuffer(_maxColsPerChunk * rows),    
    _rows(rows) {}

template <typename T>
StreamHelper<T>::~StreamHelper() = default;

template <typename T>
bool StreamHelper<T>::hasNext() const {
    return _colsProcessed < _totalCols;
}

template <typename T>
size_t StreamHelper<T>::getChunkWidth() const {
    return std::min(_maxColsPerChunk, _totalCols - _colsProcessed);
}

template <typename T>
std::vector<T>&  StreamHelper<T>::getBuffer() {
    return _hostBuffer;
}

template <typename T>
void StreamHelper<T>::updateProgress() {
    _colsProcessed += getChunkWidth();
}

template <typename T>
size_t StreamHelper<T>::getColsProcessed() const {
    return _colsProcessed;
}

// --- SetFromFile Definitions ---
template <typename T>
StreamSet<T>::StreamSet(size_t rows, size_t cols, std::istream& input_stream)
    : StreamHelper<T>(rows, cols), _input_stream(input_stream) {}

template <typename T>
void StreamSet<T>::readChunk(bool isText) {
    
    size_t num_elements = this->getChunkWidth() * this->_rows;
    size_t current_chunk_bytes = num_elements * sizeof(T);
    
    if(isText) {
        for (size_t i = 0; i < num_elements; ++i) 
            if (!(this->_input_stream >> this->_hostBuffer[i]))
                throw std::runtime_error("Failed to read enough elements. Failed at index " + i);
    } else this->_input_stream.read(reinterpret_cast<char*>(this->_hostBuffer.data()), current_chunk_bytes);

    if (!this->_input_stream) throw std::runtime_error("Stream read error or premature end of stream.");
    
}

// --- GetToFile Definitions ---
template <typename T>
StreamGet<T>::StreamGet(size_t rows, size_t cols, std::ostream& output_stream)
    : StreamHelper<T>(rows, cols), _output_stream(output_stream) {cudaDeviceSynchronize();}

template <typename T>
void StreamGet<T>::writeChunk(bool isText) {
    size_t num_elements = this->getChunkWidth() * this->_rows;
    size_t current_chunk_bytes = num_elements * sizeof(T);

    if (current_chunk_bytes > 0) {
        if(isText){
            for (size_t i = 0; i < this->getChunkWidth(); ++i) {
                for(size_t j = 0; j < this->_rows; ++j)
                    if (!(this->_output_stream << this->_hostBuffer[i * this->_rows + j] << '\t')) 
                        throw std::runtime_error("Failed to write enough elements");
                this->_output_stream << '\n';
            }                
                
        } 
        else this->_output_stream.write(reinterpret_cast<const char*>(this->_hostBuffer.data()), current_chunk_bytes);
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
    Handle* handle,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB
) const {
    
    Handle* h = handle ? handle : new Handle();

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
    Handle* handle,
    double alpha,
    double beta,
    bool transposeA,
    bool transposeB
) const {
    Handle* h = handle ? handle : new Handle();
    
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
    Handle* handle,
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
    Handle* handle,
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

Handle::Handle(cudaStream_t user_stream) {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Failed to create cuBLAS handle");    

    if (user_stream == nullptr) {
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            cublasDestroy(handle);
            throw std::runtime_error("Failed to create CUDA stream");
        }
        this->isOwner = true;
    } else {
        this->isOwner = false;
        this->stream = user_stream;
    }

    if (cublasSetStream(handle, this->stream) != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        if (this->isOwner) cudaStreamDestroy(this->stream);
        throw std::runtime_error("Failed to set CUBLAS stream");
    }
}

Handle::Handle() : Handle(nullptr) {}

Handle::~Handle() {
    cublasDestroy(handle);
    if (this->isOwner) {
        cudaStreamDestroy(stream);
    }
}

// Disallow mixed-type multiplication for float/double
template<>
void CuArray<float>::mult(
    const CuArray<double>& other,
    CuArray<double>* result,
    Handle* handle,
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
    Handle* handle,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB
) const {
    throw std::logic_error("Multiplication not supported between double and float");
}


// Explicit template instantiations to satisfy the linker

// CuFileHelper
template class StreamHelper<int>;
template class StreamHelper<float>;
template class StreamHelper<double>;

// SetFromFile
template class StreamSet<int>;
template class StreamSet<float>;
template class StreamSet<double>;

// GetToFile
template class StreamGet<int>;
template class StreamGet<float>;
template class StreamGet<double>;

// CuArray
template class CuArray<int>;
template class CuArray<float>;
template class CuArray<double>;

// If you also use mult() with mixed types (float vs double or int), explicitly instantiate those specializations:
// For example:
template void CuArray<float>::mult(
    const CuArray<float>&, CuArray<float>*, Handle*, float, float, bool, bool
) const;

template void CuArray<double>::mult(
    const CuArray<double>&, CuArray<double>*, Handle*, double, double, bool, bool
) const;