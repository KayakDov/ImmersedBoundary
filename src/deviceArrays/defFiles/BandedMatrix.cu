#include <iostream>

#include "../headers/bandedMat.h"
#include "../headers/KernelSupport.cuh"
#include "../headers/singleton.h"
#include "../headers/squareMat.h"
#include "../headers/vec.h"

/**
 * Kernel for sparse diagonal matrix-vector multiplication.
 *
 * When calling this kernel, <<<numberOfBlocks, threadsPerBlock, sharedMemorySize, stream>>>,
 * Number of blocks should be the number of rows in the solution vector.
 * Threads per block should be 32.
 * Shared memory size should be sizeof(T) * 32.
 *
 * @param banded Packed diagonals of the matrix.  Trailing values are not read.  Each row is a diagonal, and the matrix is stored in column-major order.  There may be no more than 32 rows.
 * @param height Number of rows in the greater matrix, not the packed matrix.  This is also the number of columns in the greater matrix and the number of rows in x.
 * @param ld Leading dimension of the packed diagonals.
 * @param diags Indices of the diagonals.  Negative indices indicate sub-diagonals.
 * Positive indices indicate super-diagonals.
 * For example, diags = {-1, 0, 1} means the first diagonal is the sub-diagonal, the second is the main diagonal, and the third is the super-diagonal.
 * @param numDiags Number of nonzero diagonals.  This number must be less than or equal to 32.
 * @param x Input vector.
 * @param strideX Stride for the input vector x.
 * @param result Output vector.
 * @param strideR Stride for the output vector result.
 * @param alpha Scalar multiplier for the matrix-vector product.
 * @param beta Scalar multiplier for the existing values in the result vector.
 */
template <typename T>
__global__ void diagMatVecKernel(
    const T* __restrict__ banded, // packed diagonals
    const size_t height, const size_t ld,

    const int* __restrict__ diags, // which diagonals
    const size_t numDiags,

    const T* __restrict__ x,      // input vector
    const size_t strideX,

    T* __restrict__ result,       // output vecto
    const size_t strideR,

    const T* alpha,
    const T* beta
){
    const size_t rowX = blockIdx.x;
    const size_t bandedRow = threadIdx.x;
    const bool isValid = rowX < height && bandedRow < numDiags;
    T val;
    if (isValid){//TODO: this condition can be removed by requiring input matrices have exactly 32 rows, with extra rows having all 0's.  This may give a small speed boost.
        const int32_t d = diags[bandedRow];
        int32_t bandedCol = rowX;
        if(d < 0) bandedCol += d;
        val = 0 <= bandedCol && bandedCol < height - abs(d) ? banded[bandedCol*ld + bandedRow] * x[(rowX + d) * strideX] : 0;
    } else val = 0;

    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    size_t indR = rowX * strideR;

    if(isValid && bandedRow == 0) result[indR] = *alpha * val + *beta * result[indR];
}

/**
 * Multiplies a sparse diagonal matrix (stored in packed diagonal format) with a 1D vector.
 *
 * @param vec Input vector.
 * @param handle Optional Cuda handle for stream/context management.
 * @param diags Array of diagonal indices (negative=sub-diagonal, 0=main, positive=super-diagonal)
 * @return A new CuArray1D containing the result.
 *
 */
template <typename T>
Vec<T> BandedMat<T>::mult(
const Vec<T>& other,
Vec<T>* result,
Handle* handle,
const Singleton<T> *alpha,
const Singleton<T> *beta,
bool transpose
) const {
    if (this->_rows > 32)
        throw std::invalid_argument("height must be <= 32 for this kernel");

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Vec<T>> temp_res_ptr;
    Vec<T>* resPtr = this->_get_or_create_target(this->_rows, result, temp_res_ptr, h->stream);
    std::unique_ptr<Singleton<T>> temp_a_ptr;
    const Singleton<T>* a = this->_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* b = this->_get_or_create_target(static_cast<T>(0), *h, beta, temp_b_ptr);

    if (transpose) (const_cast<Vec<int32_t>&>(_indices)).mult(Singleton<int32_t>::MINUS_ONE, h);

    diagMatVecKernel<<<this->_cols, 32, 0, h->stream>>>(
        this->data(), this->_cols, this->_ld,
        _indices.data(), this->_rows,
        other.data(), other._ld,
        resPtr->data(),
        resPtr->_ld,
        a->data(), b->data()
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    if (transpose) (const_cast<Vec<int32_t>&>(_indices)).mult(Singleton<int32_t>::MINUS_ONE, h);

    return *resPtr;
}

template<typename T>
Mat<T> BandedMat<T>::mult(const Mat<T> &other, Mat<T> *result, Handle *handle, const Singleton<T> *alpha,
    const Singleton<T> *beta, bool transposeA, bool transposeB) const {
    throw std::runtime_error("Not implemented");

}


template <typename T>
__global__ void mapToDenseKernel(
    T* __restrict__ dense,
    const size_t heightWidth, const size_t denseLd,
    const T* __restrict__ banded,
    const size_t numDiags, const size_t bandedLd,
    const int32_t* __restrict__ indices
) {
    const size_t bandedRow = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t bandedCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (bandedRow < numDiags && bandedCol < heightWidth)
        if (const DenseInd denseInd(bandedRow, bandedCol, indices); !denseInd.outOfBounds(heightWidth))
            dense[denseInd.flat(denseLd)] = banded[bandedCol * bandedLd + bandedRow];
}

template<typename T>
void BandedMat<T>::getDense(SquareMat<T> dense, Handle *handle) const {//TODO: have a default nullptr dense value and return Square Mat
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    constexpr dim3 blockDim(16, 16);

    const dim3 gridDim(
        (this->_cols + blockDim.x - 1) / blockDim.x,
        (this->_indices.size() + blockDim.y - 1) / blockDim.y
    );

    mapToDenseKernel<T><<<gridDim, blockDim, 0, h->stream>>>(
        dense.data(),
        dense._cols,
        dense._ld,
        this->data(),
        this->_rows,
        this->_ld,
        this->_indices.data()
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template<typename T>
Mat<T> BandedMat<T>::operator*(const Mat<T> &other) const {
    throw std::runtime_error("Not implemented");
}

template<typename T>
Mat<T> BandedMat<T>::plus(const Mat<T> &x, Mat<T> *result, const Singleton<T> *alpha, const Singleton<T> *beta,
    bool transposeA, bool transposeB, Handle *handle) {
    throw std::runtime_error("Not implemented");
}

template<typename T>
Mat<T> BandedMat<T>::minus(const Mat<T> &x, Mat<T> *result, const Singleton<T> *alpha, const Singleton<T> *beta,
    bool transposeA, bool transposeB, Handle *handle) {
    throw std::runtime_error("Not implemented");
}

template<typename T>
void BandedMat<T>::mult(const Singleton<T> &alpha, Handle *handle) {
    throw std::runtime_error("Not implemented");
}

template<typename T>
void BandedMat<T>::transpose(Mat<T> &result, Handle *handle) const {
    throw std::runtime_error("Not implemented");
}

template<typename T>
void BandedMat<T>::transpose(Handle *handle, Mat<T> *preAlocatedMem) {
    throw std::runtime_error("Not implemented");
}

template<typename T>
Mat<T> BandedMat<T>::create(size_t rows, size_t cols) {
    throw std::runtime_error("Not implemented");
}

template<typename T>
Mat<T> BandedMat<T>::subMat(size_t startRow, size_t startCol, size_t height, size_t width) const {
    throw std::runtime_error("Not implemented");
}

template<typename T>
Vec<T> BandedMat<T>::col(size_t index) {
    throw std::runtime_error("Not implemented");
}

template<typename T>
Vec<T> BandedMat<T>::row(size_t index) {
    throw std::runtime_error("Not implemented");
}

template<typename T>
void BandedMat<T>::normalizeCols(size_t setRowTo1, Handle *handle) {
    throw std::runtime_error("Not implemented");
}


template<typename T>
BandedMat<T>::BandedMat(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> ptr, const Vec<int32_t> &indices):
    Mat<T>(rows, cols, ld, ptr), _indices(indices) {
}

template<typename T>
BandedMat<T>::BandedMat(const Mat<T>& copyFrom, const Vec<int32_t>& indices):
    BandedMat(copyFrom._rows, copyFrom._cols, copyFrom._ld, copyFrom._ptr, indices) {
    if (indices.size() != copyFrom._rows) throw std::invalid_argument("indices must be the same length as the number of rows in the matrix");
}

template<typename T>
BandedMat<T> BandedMat<T>::create(size_t numDiagonals, size_t cols, const Vec<int32_t> &indices) {
    return BandedMat<T>(Mat<T>::create(numDiagonals, cols), indices);
}

template <typename T>
__global__ void mapDenseToBandedKernel(
    const T* __restrict__ dense,
    const size_t heightWidth, const size_t denseLd,
    T* __restrict__ banded,
    const size_t numDiags, const size_t bandedLd,
    const int32_t* __restrict__ indices
) {
    const size_t bandedRow = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t bandedCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (bandedRow < numDiags && bandedCol < heightWidth){
        const size_t writeTo = bandedCol*bandedLd + bandedRow;
        if (const DenseInd denseInd(bandedRow, bandedCol, indices); denseInd.outOfBounds(heightWidth)) banded[writeTo] = NAN;
        else banded[writeTo] = dense[denseInd.flat(denseLd)];

    }

}

template<typename T>
void BandedMat<T>::setFromDense(const SquareMat<T> &denseMat, Handle *handle) {

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    constexpr dim3 blockDim(16, 16);

    const dim3 gridDim(
        (this->_cols + blockDim.x - 1) / blockDim.x,
        (this->_rows + blockDim.y - 1) / blockDim.y
    );

    mapDenseToBandedKernel<T><<<gridDim, blockDim, 0, h->stream>>>(
        denseMat.data(),
        denseMat._rows,
        denseMat._ld,
        this->data(),
        this->_rows,
        this->_ld,
        this->_indices.data()
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}


template class BandedMat<float>;
template class BandedMat<double>;
template class BandedMat<size_t>;
template class BandedMat<int32_t>;
template class BandedMat<unsigned char>;