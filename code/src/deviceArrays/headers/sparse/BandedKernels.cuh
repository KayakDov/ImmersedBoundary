/**
 * @file BandedKernels.cuh
 * @brief Declares CUDA kernels for banded-matrix operations.
 * @ingroup sparse_matrices
 *
 * @details
 * Sparse-matrix classes build on the device array layer and expose storage-format-specific operations while preserving explicit CUDA handle and stream control.
 */

#ifndef CUDABANDED_BANDEDKERNELS_CUH
#define CUDABANDED_BANDEDKERNELS_CUH
#include "deviceArrays/headers/DeviceData.cuh"

/**
 * Sums all the elements in the block into this value.
 * @tparam T
 * @param val
 */
template<typename T>
__device__ void sumBlock(T& val) {
    for (int offset = 16; offset > 0; offset >>= 1) val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}

template<typename T>
__device__ void multBandedVec(
    const DeviceData2d<T> banded, // packed diagonals
    const int32_t *__restrict__ diags, // the number of diagonals is banded.cols
    const DeviceData1d<T> x, // input vector
    DeviceData1d<T> result,
    const T *alpha, const T *beta
) {
    const size_t dstRow = idx();

    if (dstRow >= result.cols) return;

    T sum = 0;
    for (size_t bandedCol = 0; bandedCol < banded.cols; ++bandedCol) {
        AdjacencyInd adjInd(bandedCol, diags[bandedCol]);
        if (adjInd.inBoundsRow(dstRow, x.cols))
            sum += banded[adjInd.bandedIndRow(dstRow)] * x[adjInd.denseCol(dstRow)];
    }
    result[dstRow] = *alpha * sum + (*beta == 0 ? 0 : *beta * result[dstRow]);
}
/**
 * Kernel for sparse diagonal matrix-vector multiplication.
 *
 * When calling this kernel, <<<numberOfBlocks, threadsPerBlock, sharedMemorySize, stream>>>,
 * In the x dimension:
 *  Number of blocks should be the number of rows in the solution vector.
 *  Threads per block should be 32.
 * In the y dimension
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
__global__ void productBandedVec(
    const DeviceData2d<T> banded, // packed diagonals
    const int32_t *__restrict__ diags, // the number of diagonals is banded.cols
    const DeviceData1d<T> x, // input vector
    DeviceData1d<T> result,
    const T *alpha, const T *beta
) {
    multBandedVec(banded, diags, x, result, alpha, beta);
}

template<typename T>
__global__ void productBandedMat(
    const DeviceData2d<T> banded, // packed diagonals
    const int32_t *__restrict__ diags, // the number of diagonals is banded.cols
    const DeviceData2d<T> x, // input vector
    DeviceData2d<T> result,
    const T *alpha, const T *beta
) {
    const size_t dstCol = idy();
    if (dstCol >= x.cols) return;

    multBandedVec(banded, diags, x.col(dstCol), result.col(dstCol), alpha, beta);
}

template<typename T>
__device__ void multVecBanded(
    const DeviceData1d<T> x,  // input vector
    const DeviceData2d<T> banded, // packed diagonals
    const int32_t *__restrict__ diags, // the number of diagonals is banded.cols
    DeviceData1d<T> result,
    const T *alpha, const T *beta
) {
    const size_t dstCol = idx();

    if (dstCol >= result.cols) return;

    T sum = 0;
    for (size_t bandedCol = 0; bandedCol < banded.cols; ++bandedCol) {
        AdjacencyInd adjInd(bandedCol, diags[bandedCol]);
        if (adjInd.inBoundsCol(dstCol, x.cols))
            sum += banded[adjInd.bandedIndCol(dstCol)] * x[adjInd.denseRow(dstCol)];
    }
    result[dstCol] = *alpha * sum + (*beta == 0 ? 0 : *beta * result[dstCol]);
}


template<typename T>
__global__ void productVecBanded(
    const DeviceData1d<T> x, // input vector
    const DeviceData2d<T> banded, // packed diagonals
    const int32_t *__restrict__ diags, // the number of diagonals is banded.cols
    DeviceData1d<T> result,
    const T *alpha, const T *beta
) {
    multVecBanded(x, banded, diags, result, alpha, beta);
}

template<typename T>
__global__ void productMatBanded(
    const DeviceData2d<T> x, // input vector
    const DeviceData2d<T> banded, // packed diagonals
    const int32_t *__restrict__ diags, // the number of diagonals is banded.cols
    DeviceData2d<T> result,
    const T *alpha,
    const T *beta
) {
    const size_t rowDense = idy();
    if (rowDense >= result.rows) return;
    multVecBanded(x.row(rowDense), banded, diags, result.row(rowDense), alpha, beta);
}
#endif //CUDABANDED_BANDEDKERNELS_CUH
