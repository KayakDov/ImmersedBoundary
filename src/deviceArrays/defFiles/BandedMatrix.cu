#include <iostream>

#include "../headers/sparse/BandedMat.h"
#include "../headers/KernelSupport.cuh"
#include "../headers/Singleton.h"
#include "../headers/SquareMat.h"
#include "../headers/Vec.h"

/**
 * Checks if the index is in bounds.
 * @param minInclusive
 * @param maxExclusive
 * @param index
 * @return true if the index is in bounds, false otherwise.
 */
__device__ bool inBounds(int32_t minInclusive, int32_t maxExclusive, int32_t index) {
    return minInclusive <= index && index < maxExclusive;
}

/**
 * Sums all the elements in the block into this value.
 * @tparam T
 * @param val
 */
template<typename T>
__device__ void sumBlock(T& val) {
    for (int offset = 16; offset > 0; offset >>= 1) val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}

/**
 * Kernel for sparse diagonal matrix-vector multiplication.
 *
 * When calling this kernel, <<<numberOfBlocks, threadsPerBlock, sharedMemorySize, stream>>>,
 * Number of blocks should be the number of rows in the solution vector.
 * Threads per block should be 32.
 * Shared memory size should be sizeof(T) * 32.
 *
 * @param banded Packed diagonals of the matrix.  Trailing values are not read.  Each row is a diagonal, and the matrix is stored in column-major order.  There may be no more than 32 rows.
 * @param diags Indices of the diagonals.  Negative indices indicate sub-diagonals.
 * Positive indices indicate super-diagonals.
 * For example, diags = {-1, 0, 1} means the first diagonal is the sub-diagonal, the second is the main diagonal, and the third is the super-diagonal.
 * @param x Input vector.
 * @param result Output vector.
 * @param alpha Scalar multiplier for the matrix-vector product.
 * @param beta Scalar multiplier for the existing values in the result vector.
 */
template<typename T>
__global__ void multVecKernel(
    const DeviceData2d<T> banded, // packed diagonals
    const int32_t *__restrict__ diags, // the number of diagonals is banded.cols
    const DeviceData1d<T> x, // input vector

    DeviceData1d<T> result,

    const T *alpha,
    const T *beta
) {
    const size_t rowResult = blockIdx.x;
    const size_t bandedCol = threadIdx.x;

    const bool isValid = rowResult < x.cols && bandedCol < banded.cols;
    T val;
    if (isValid) {
        const int32_t d = diags[bandedCol];
        int32_t bandedRow = rowResult, xRow;
        bandedRow += (d < 0) * d;
        xRow = rowResult + d;
        val = inBounds(0, banded.rows - abs(d), bandedRow) && inBounds(0, x.cols, xRow) ? banded(bandedRow, bandedCol) * x[xRow]:0;
    } else val = 0;

    sumBlock(val);

    if (isValid && bandedCol == 0) result[rowResult] = *alpha * val + *beta * result[rowResult];
}

/**
 * Multiplies a sparse banded matrix (stored in packed diagonal format) with a 1D vector.
 * result <- alpha * other + beta * result
 * @param other The vector this matrix is multiplied by.
 * @param result The result of the multiplication will be put here.
 * @param handle Optional Cuda handle for stream/context management.
 * @param alpha multiplies the product of this and other.  By default, set to &Singleton<T>::ONE.
 * @param beta  Multiplies the result before the product is added.  If the result is meant to start with no values, set to &Singleton<T>::ZERO
 * @param transpose  Should this matrix be transposed.
 * @return A new CuArray1D containing the result.
 *
 */
template<typename T>
void BandedMat<T>::bandedMult(
    const Vec<T> &other,
    Vec<T> &result,
    Handle *handle,
    const Singleton<T> alpha,
    const Singleton<T> beta,
    bool transpose
) const {
    if (this->_cols > 32) throw std::invalid_argument("width must be <= 32 for this kernel");

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    if (transpose) (const_cast<Vec<int32_t> &>(_indices)).mult(Singleton<int32_t>::MINUS_ONE, h);

    multVecKernel<<<this->_rows, this->_cols, 0, *h>>>(
        this->toKernel2d(),
        _indices.toKernel1d().data,
        other.toKernel1d(),
        result.toKernel1d(),
        alpha.toKernel1d().data,
        beta.toKernel1d().data
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    if (transpose) (const_cast<Vec<int32_t> &>(_indices)).mult(Singleton<int32_t>::MINUS_ONE, h);
}

template<typename T>
__global__ void mapToDenseKernel(
    DeviceData2d<T> denseSquare,
    const DeviceData2d<T> banded, //num diagonals is width, length should be dense.width
    const int32_t *__restrict__ indices
) {
    GridInd2d sparseInd;
    if (sparseInd >= banded) return;
    int32_t diag = indices[sparseInd.col];
    GridInd2d denseInd(
    sparseInd.row - (diag < 0 ? diag : 0),
    sparseInd.row + (diag > 0 ? diag : 0)
    );
    denseSquare[denseInd] = banded[sparseInd];
}

template<typename T>
void BandedMat<T>::getDense(SquareMat<T> dense, Handle *handle) const {
    //TODO: have a default nullptr dense value and return Square Mat
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    const KernelPrep kp = this->kernelPrep();

    mapToDenseKernel<T><<<kp.numBlocks, kp.threadsPerBlock, 0, *h>>>(
        dense.toKernel2d(),
        this->toKernel2d(),
        this->_indices.toKernel1d()
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template<typename T>
BandedMat<T>::BandedMat(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> ptr,
                        const Vec<int32_t> &indices) : Mat<T>(rows, cols, ld, ptr), _indices(indices) {
}

template<typename T>
BandedMat<T>::BandedMat(const Mat<T> &copyFrom, const Vec<int32_t> &indices) : BandedMat(
    copyFrom._rows, copyFrom._cols, copyFrom._ld, copyFrom.ptr(), indices) {
    if (indices.size() != copyFrom._cols) throw std::invalid_argument(
        "indices must be the same length as the number of rows in the matrix");
}

template<typename T>
BandedMat<T> BandedMat<T>::create(size_t denseSqMatDim, size_t numDiagonals, const Vec<int32_t> &indices) {
    return BandedMat<T>(Mat<T>::create(denseSqMatDim, numDiagonals), indices);
}

template<typename T>
BandedMat<T> BandedMat<T>::create(size_t denseSqMatDim, size_t numDiagonals, const size_t ld, T *data, int32_t *indices, size_t indsStride) {
    return BandedMat<T>(
        Mat<T>::create(denseSqMatDim, numDiagonals, ld, data),
        Vec<int32_t>::create(numDiagonals, indsStride, indices)
    );
}

template<typename T>
__global__ void mapDenseToBandedKernel(
    const DeviceData2d<T> dense,
    DeviceData2d<T> banded,
    const int32_t *__restrict__ indices
) {
    if (const GridInd2d bandedInd; bandedInd < banded) {
        if (const DenseInd denseInd(bandedInd, indices); denseInd >= dense) banded[bandedInd] = NAN;
        else banded[bandedInd] = dense[denseInd];
    }
}

template<typename T>
void BandedMat<T>::setFromDense(const SquareMat<T> &denseMat, Handle *handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    const KernelPrep kp = this->kernelPrep();

    mapDenseToBandedKernel<T><<<kp.numBlocks, kp.threadsPerBlock, 0, *h>>>(
        denseMat.toKernel2d(),
        this->toKernel2d(),
        this->_indices.toKernel1d()
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}


template class BandedMat<float>;
template class BandedMat<double>;
template class BandedMat<size_t>;
template class BandedMat<int32_t>;
template class BandedMat<unsigned char>;
