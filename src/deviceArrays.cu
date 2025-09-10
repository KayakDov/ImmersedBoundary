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
void StreamSet<T>::readChunk(bool isRowMajor) {
    
    size_t num_elements = this->getChunkWidth() * this->_rows;
    size_t current_chunk_bytes = num_elements * sizeof(T);
    
    if(isRowMajor) {
        for (size_t i = 0; i < num_elements; ++i) 
            if (!(this->_input_stream >> this->_hostBuffer[i]))
                throw std::runtime_error("Failed to read enough elements. Failed at index " + std::to_string(i));
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
            for (size_t col = 0; col < this->getChunkWidth(); ++col) {
                for(size_t row = 0; row < this->_rows; ++row)
                    if (!(this->_output_stream << this->_hostBuffer[col * this->_rows + row] << '\t')) 
                        throw std::runtime_error("Failed to write enough elements");
                this->_output_stream << '\n';
            }                
                
        } 
        else this->_output_stream.write(reinterpret_cast<const char*>(this->_hostBuffer.data()), current_chunk_bytes);
        if (!this->_output_stream) throw std::runtime_error("Stream write error.");
    }
}

template <typename T>
GpuArray<T>::GpuArray(size_t rows, size_t cols, size_t ld)
    : _rows(rows), _cols(cols), _ld(ld){}

template <typename T>
GpuArray<T>::~GpuArray() = default;

template <typename T>
T* GpuArray<T>::data() { return static_cast<T*>(_ptr.get()); }

template <typename T>
const T* GpuArray<T>::data() const { return static_cast<const T*>(_ptr.get()); }

template <typename T>
size_t GpuArray<T>::getLD() const { return _ld; }

template <typename T>
void GpuArray<T>::mult(
    const GpuArray<T>& other,
    GpuArray<T>* result,
    Handle* handle,
    const Singleton<T> *alpha,
    const Singleton<T> *beta,
    bool transposeA,
    bool transposeB
) const {
    
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Singleton<T>> temp_a_ptr;
    const Singleton<T>* a = _get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* b = _get_or_create_target(static_cast<T>(0), *h, beta, temp_b_ptr);

    if constexpr (std::is_same_v<T, float>)
        cublasSgemm(h->handle,
        transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows, other._cols, this->_cols,
        a->data(),
        this->data(), this->getLD(),
        other.data(), other.getLD(),
        b->data(),
        result->data(), result->getLD());
    else if constexpr (std::is_same_v<T, double>)
        cublasDgemm(h->handle,
        transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows, other._cols, this->_cols,
        a->data(),
        this->data(), this->getLD(),
        other.data(), other.getLD(),
        b->data(),
        result->data(), result->getLD());
    else throw std::invalid_argument("Unsupported type.");

    
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
    if (this->isOwner) cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void Handle::synch() const {
    CHECK_CUDA_ERROR(cudaStreamSynchronize(this->stream));
}

template <typename T>
Mat<T>* GpuArray<T>::_get_or_create_target(size_t rows, size_t cols, Mat<T>* result, std::unique_ptr<Mat<T>>& out_ptr_unique) const {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Mat<T>>(rows, cols);
        return out_ptr_unique.get();
    }
}

template <typename T>
Singleton<T>* GpuArray<T>::_get_or_create_target(Singleton<T>* result, std::unique_ptr<Singleton<T>>& out_ptr_unique) const {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Singleton<T>>();
        return out_ptr_unique.get();
    }
}

template <typename T>
const Singleton<T>* GpuArray<T>::_get_or_create_target(T defaultVal, Handle& hand, const Singleton<T>* result, std::unique_ptr<Singleton<T>>& out_ptr_unique) const {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Singleton<T>>();
        out_ptr_unique->set(defaultVal, hand.stream);
        return out_ptr_unique.get();
    }
}

template <typename T>
Vec<T>* GpuArray<T>::_get_or_create_target(size_t length, Vec<T>* result, std::unique_ptr<Vec<T>>& out_ptr_unique) const {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Vec<T>>(length);
        return out_ptr_unique.get();
    }
}

Handle* Handle::_get_or_create_handle(Handle* handle, std::unique_ptr<Handle>& out_ptr_unique) {
    if (handle) return handle;
    else {
        out_ptr_unique = std::make_unique<Handle>();
        return out_ptr_unique.get();
    }
}

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
template class GpuArray<int>;
template class GpuArray<float>;
template class GpuArray<double>;