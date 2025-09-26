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

template<typename T>
GpuArray<T>::GpuArray(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> _ptr):_rows(rows), _cols(cols), _ld(ld), _ptr(_ptr) {
}

template <typename T>
GpuArray<T>::~GpuArray() = default;



template <typename T>
__global__ void fill2dKernel(T* __restrict__ a, const size_t height, const size_t width, const size_t ld, const T val){

    if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < height * width)
        a[(idx / height) * ld + idx % height] = val;
}


template<typename T>
void GpuArray<T>::fill(T val, cudaStream_t stream) {

    constexpr size_t BLOCK_SIZE = 256;
    size_t num_blocks = (this->size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fill2dKernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(this->data(), this->_rows, this->_cols, this->_ld, val);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename T>
T* GpuArray<T>::data() { return _ptr.get(); }

template <typename T>
const T* GpuArray<T>::data() const {return _ptr.get(); }

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
        out_ptr_unique = std::make_unique<Mat<T>>(Mat<T>::create(rows, cols));
        return out_ptr_unique.get();
    }
}

template <typename T>
Singleton<T>* GpuArray<T>::_get_or_create_target(Singleton<T>* result, std::unique_ptr<Singleton<T>>& out_ptr_unique, cudaStream_t stream) const {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Singleton<T>>(Singleton<T>::create(stream));
        return out_ptr_unique.get();
    }
}

template <typename T>
const Singleton<T>* GpuArray<T>::_get_or_create_target(T defaultVal, Handle& hand, const Singleton<T>* result, std::unique_ptr<Singleton<T>>& out_ptr_unique) const {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Singleton<T>>(Singleton<T>::create(defaultVal, hand.stream));
        return out_ptr_unique.get();
    }
}

template <typename T>
Vec<T>* GpuArray<T>::_get_or_create_target(size_t length, Vec<T>* result, std::unique_ptr<Vec<T>>& out_ptr_unique, cudaStream_t stream) const {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Vec<T>>(Vec<T>::create(length, stream));
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
template class StreamHelper<float>;
template class StreamHelper<double>;
template class StreamHelper<int32_t>;
template class StreamHelper<size_t>;

// SetFromFile
template class StreamSet<float>;
template class StreamSet<double>;
template class StreamSet<int32_t>;
template class StreamSet<size_t>;

// GetToFile
template class StreamGet<float>;
template class StreamGet<double>;
template class StreamGet<int32_t>;
template class StreamGet<size_t>;

// CuArray
template class GpuArray<float>;
template class GpuArray<double>;
template class GpuArray<int32_t>;
template class GpuArray<size_t>;