/**
 * @file deviceArrays.cu
 * @brief Templated classes for managing 1D and 2D arrays on a CUDA device in column-major order.
 */
#include "../headers/Singleton.h"
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

template<typename T>
size_t GpuArray<T>::bytes() const {
    return sizeof(T) * size();
}

template <typename T>
__global__ void fill2dKernel(DeviceData2d<T> a, const T val){
    if (const GridInd2dT ind; ind < a) a[ind] = val;
}

template <typename T>
__global__ void fill2dKernelT(DeviceData2d<T> a, const T val){
    if (const GridInd2dT ind; ind < a) a[ind] = val;
}

template<typename T>
void GpuArray<T>::fill(T val, cudaStream_t stream) {

    KernelPrep kp = kernelPrep(true);

    fill2dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(this->toKernel2d(), val);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template<typename T>
std::shared_ptr<T> GpuArray<T>::ptr() const{
    return this->_ptr;
}

template<typename T>
void GpuArray<T>::freeMem() {
    this->_ptr.reset();
}

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
    const Singleton<T>* a = Singleton<T>::_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* b = Singleton<T>::_get_or_create_target(static_cast<T>(0), *h, beta, temp_b_ptr);


    cublasOperation_t transA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
            transB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    if constexpr (std::is_same_v<T, float>)
        CHECK_CUBLAS_ERROR(cublasSgemm(*h,
        transA, transB,
        this->_rows, other._cols, this->_cols,
        a->toKernel1d(),
        this->data(), this->_ld,
        other.data(), other._ld,
        b->toKernel1d(),
        result->data(), result->_ld));
    else if constexpr (std::is_same_v<T, double>)
        CHECK_CUBLAS_ERROR(cublasDgemm(*h,
        transA, transB,
        this->_rows, other._cols, this->_cols,
        a->toKernel1d(),
        this->data(), this->_ld, other.data(), other._ld,
        b->data(), result->data(), result->_ld));
    else throw std::invalid_argument("Unsupported type.");
}

// --- kronecker_kernel.cu (or part of deviceArrays.cu) ---

#include <cuda_runtime.h>
// Include the deviceArrays.h header here to get GpuArray class definition

/**
 * @brief CUDA kernel for computing the Kronecker product C = A (x) B using a 2D block grid.
 *
 * This kernel assigns one thread to each element of the resulting matrix C by
 * mapping the thread's 2D global index (row, col) directly to the element's
 * position in C.
 *
 * The dimensions are:
 * - A: (heightA x widthA)
 * - B: (heightB x widthB)
 * - C: (heightA * heightB) x (widthA * widthB)
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @param A Pointer to matrix A data on the device.
 * @param B Pointer to matrix B data on the device.
 * @param result Pointer to result matrix C data on the device.
 */
template <typename T>
__global__ void kroneckerKernel(const DeviceData2d<T> a, const DeviceData2d<T> b, DeviceData2d<T> result) {

    if (const GridInd2d indR; indR < result)
        result[indR] += a(indR.row / b.rows, indR.col / b.cols) * b(indR.row % b.rows, indR.col % b.cols);

}

template<typename T>
void GpuArray<T>::multKronecker(const GpuArray<T>& other, GpuArray<T>& result, cudaStream_t stream) const {

    KernelPrep kp = kernelPrep();

    kroneckerKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        this->toKernel2d(),
        other.toKernel2d(),
        result.toKernel2d()
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
}

template<typename T>
T* GpuArray<T>::data() {
    return this->_ptr.get();
}

template<typename T>
const T* GpuArray<T>::data() const {
    return this->_ptr.get();
}

template<typename T>
KernelPrep GpuArray<T>::kernelPrep(bool transpose) const {
    return {this->_cols, this->_rows, transpose};
}

template <typename T>
GpuArray<T>::operator T*() {
    return data();
}

template <typename T>
GpuArray<T>::operator const T*() const {
    return data();
}

template<typename T>
GpuArray<T>::operator DeviceData2d<T>() {
    return toKernel2d();
}

template<typename T>
GpuArray<T>::operator DeviceData2d<T>() const {
    return toKernel2d();
}

template <typename T>
DeviceData2d<T> GpuArray<T>::toKernel2d() { return DeviceData2d<T>(this->_rows, this->_cols, this->_ld, this->_ptr.get()); }

template <typename T>
DeviceData2d<T> GpuArray<T>::toKernel2d() const { return DeviceData2d<T>(this->_rows, this->_cols, this->_ld, this->_ptr.get()); }

template class GpuArray<float>;
template class GpuArray<double>;
template class GpuArray<int32_t>;
template class GpuArray<int64_t>;
template class GpuArray<size_t>;
template class GpuArray<unsigned char>;
template class GpuArray<uint32_t>;
