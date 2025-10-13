/**
 * @file deviceArrays.cu
 * @brief Templated classes for managing 1D and 2D arrays on a CUDA device in column-major order.
 */
#include "../headers/singleton.h"
#include <vector>
#include <numeric>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fstream>


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
        this->data(), this->_ld,
        other.data(), other._ld,
        b->data(),
        result->data(), result->_ld);
    else if constexpr (std::is_same_v<T, double>)
        cublasDgemm(h->handle,
        transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows, other._cols, this->_cols,
        a->data(),
        this->data(), this->_ld,
        other.data(), other._ld,
        b->data(),
        result->data(), result->_ld);
    else throw std::invalid_argument("Unsupported type.");
}

// --- Helper Functions and Macros Definitions ---
void checkCudaErrors(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
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
// CuArray
template class GpuArray<float>;
template class GpuArray<double>;
template class GpuArray<int32_t>;
template class GpuArray<size_t>;
template class GpuArray<unsigned char>;