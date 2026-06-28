#include <iostream>

#include "solvers/BiCGSTAB.cuh"
#include "../../headers/sparse/BandedMat.h"
#include "../../headers/KernelSupport.cuh"
#include "../../headers/Singleton.h"
#include "../../headers/SquareMat.h"
#include "../../headers/Vec.h"
#include "deviceArrays/headers/sparse/BandedKernels.cuh"


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
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    if (transpose) (const_cast<Vec<int32_t> &>(_indices)).mult(GPUScalar<int32_t>::get(-1), h);

    auto kp = result.kernelPrep();

    productBandedVec<<<kp.numBlocks, kp.threadsPerBlock, 0, *h>>>(
        this->toKernel2d(),
        _indices.toKernel1d().data,
        other.toKernel1d(),
        result.toKernel1d(),
        alpha.toKernel1d().data,
        beta.toKernel1d().data
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    if (transpose) (const_cast<Vec<int32_t> &>(_indices)).mult(GPUScalar<int32_t>::get(-1), h);
}

template<typename T>
void BandedMat<T>::bandedMult(
    const Mat<T> &other,
    Mat<T> &result,
    Handle *handle,
    const Singleton<T> alpha,
    const Singleton<T> beta,
    bool transpose
) const {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    if (transpose) (const_cast<Vec<int32_t> &>(_indices)).mult(GPUScalar<int32_t>::get(-1), h);

    auto kp = result.kernelPrep();

    productBandedMat<<<kp.numBlocks, kp.threadsPerBlock, 0, *h>>>(
        this->toKernel2d(),
        _indices.toKernel1d().data,
        other.toKernel2d(),
        result.toKernel2d(),
        alpha.toKernel1d().data,
        beta.toKernel1d().data
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    if (transpose) (const_cast<Vec<int32_t> &>(_indices)).mult(GPUScalar<int32_t>::get(-1), h);
}

template<typename T>
__global__ void mapToDenseKernel(
    DeviceData2d<T> denseSquareDst,
    const DeviceData2d<T> bandedSrc, //num diagonals is width, length should be dense.width
    const int32_t *__restrict__ indices
) {
    GridInd2d sparseInd;
    if (sparseInd >= bandedSrc) return;
    int32_t diag = indices[sparseInd.col];
    AdjacencyInd adj(sparseInd.col, indices[sparseInd.col]);
    auto denseInd = adj.denseInd(sparseInd.row);
    if (denseInd < denseSquareDst) denseSquareDst[denseInd] = bandedSrc[sparseInd];
}

template<typename T>
void BandedMat<T>::getDense(SquareMat<T> dense, Handle& hand) const {

    dense.fill(0, hand);
    const KernelPrep kp = this->kernelPrep();

    mapToDenseKernel<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        dense.toKernel2d(),
        this->toKernel2d(),
        this->_indices.toKernel1d()
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template<typename T>
SquareMat<T> BandedMat<T>::getDense(Handle& hand) const {
    auto result = SquareMat<T>::create(this->_rows);
    getDense(result, hand);
    return result;
}

template<typename T>
BandedMat<T>::BandedMat(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> ptr,
                        const Vec<int32_t> &indices) : Mat<T>(rows, cols, ld, ptr), _indices(indices) {
}

template<typename T>
BandedMat<T>::BandedMat(const Mat<T> &windowTo, const Vec<int32_t> &indices) : BandedMat(
    windowTo._rows, windowTo._cols, windowTo._ld, windowTo.ptr(), indices) {
    if (indices.size() != windowTo._cols && windowTo._rows != 0) throw std::invalid_argument(
        "indices must be the same length as the number of rows in the matrix");
}

template<typename T>
BandedMat<T> BandedMat<T>::create(size_t denseSqMatDim, const Vec<int32_t> &indices) {
    return BandedMat<T>(Mat<T>::create(denseSqMatDim, indices.size()), indices);
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
    const GridInd2d bandedInd;
    if (bandedInd >= banded) return;

    if (const DenseInd denseInd(bandedInd, indices); denseInd >= dense) {
        if constexpr (std::is_floating_point_v<T>) banded[bandedInd] = NAN;
        else banded[bandedInd] = 0;
    }
    else banded[bandedInd] = dense[denseInd];
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
