#include <iostream>

#include "../headers/bandedMat.h"
#include "../headers/KernelSupport.cuh"
#include "../headers/Singleton.h"
#include "../headers/SquareMat.h"
#include "../headers/Vec.h"

/**
 * Kernel for sparse diagonal matrix-vector multiplication.
 *
 * When calling this kernel, <<<numberOfBlocks, threadsPerBlock, sharedMemorySize, stream>>>,
 * Number of blocks should be the number of rows in the solution vector.
 * Threads per block should be 32.
 * Shared memory size should be sizeof(T) * 32.
 *
 * @param banded Packed diagonals of the matrix.  Trailing values are not read.  Each row is a diagonal, and the matrix is stored in column-major order.  There may be no more than 32 rows.
 * @param heightDense Number of rows in the greater matrix, not the packed matrix.  This is also the number of columns in the greater matrix and the number of rows in x.
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
__global__ void multVecKernel(
    const DeviceData2d<T> banded, // packed diagonals
    const int32_t* __restrict__ diags, // the number of diagonals is banded.cols
    const DeviceData1d<T> x,      // input vector

    DeviceData1d<T> result,

    const T* alpha,
    const T* beta
){
    const size_t rowX = blockIdx.x;
    const size_t bandedCol = threadIdx.x;

    const bool isValid = rowX < x.cols && bandedCol < banded.cols;
    T val;
    if (isValid){//TODO: this condition can be removed by requiring input matrices have exactly 32 rows, with extra rows having all 0's.  This may give a small speed boost.
        const int32_t d = diags[bandedCol];
        int32_t bandedRow = rowX;
        if(d < 0) bandedRow += d;
        val = 0 <= bandedRow && bandedRow < banded.rows - abs(d) ?
            banded(bandedRow, bandedCol) * x[rowX + d] :
            0;
    } else val = 0;

    for (int offset = 16; offset > 0; offset >>= 1) val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    if(isValid && bandedCol == 0) result[rowX] = *alpha * val + *beta * result[rowX];
}

/**
 * Multiplies a sparse banded matrix (stored in packed diagonal format) with a 1D vector.
 *
 * @param vec Input vector.
 * @param handle Optional Cuda handle for stream/context management.
 * @param diags Array of diagonal indices (negative=sub-diagonal, 0=main, positive=super-diagonal)
 * @return A new CuArray1D containing the result.
 *
 */
template <typename T>
void BandedMat<T>::mult(
    const Vec<T>& other,
    Vec<T>& result,
    Handle* handle,
    const Singleton<T> *alpha,
    const Singleton<T> *beta,
    bool transpose
) const {
    if (this->_cols > 32)
        throw std::invalid_argument("height must be <= 32 for this kernel");

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Singleton<T>> temp_a_ptr;
    const Singleton<T>* a = Singleton<T>::_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* b = Singleton<T>::_get_or_create_target(static_cast<T>(0), *h, beta, temp_b_ptr);

    if (transpose) (const_cast<Vec<int32_t>&>(_indices)).mult(Singleton<int32_t>::MINUS_ONE, h);

    multVecKernel<<<this->_rows, this->_cols, 0, h->stream>>>(
        this->toKernel2d(),
        _indices.toKernel1d().data,
        other.toKernel1d(),
        result.toKernel1d(),
        a->toKernel1d().data,
        b->toKernel1d().data
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    if (transpose) (const_cast<Vec<int32_t>&>(_indices)).mult(Singleton<int32_t>::MINUS_ONE, h);
}

template<typename T>
Mat<T> BandedMat<T>::mult(const Mat<T> &other, Mat<T> *result, Handle *handle, const Singleton<T> *alpha,
    const Singleton<T> *beta, bool transposeA, bool transposeB) const {
    throw std::runtime_error("Not implemented");

}


template <typename T>
__global__ void mapToDenseKernel(
    DeviceData2d<T> denseSquare,
    const DeviceData2d<T> banded,//num diagonals is width, length should be dense.width
    const int32_t* __restrict__ indices
) {
    if (const GridInd2d bandedInd; bandedInd < banded)
        if (const DenseInd denseInd(bandedInd, indices); !denseInd.outOfBounds(denseSquare.rows))
            denseSquare(denseInd) = banded[bandedInd];
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
        dense.toKernel(),
        this->toKernel(),
        this->_indices.toKernel()
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
    BandedMat(copyFrom._rows, copyFrom._cols, copyFrom._ld, copyFrom.ptr(), indices) {
    if (indices.size() != copyFrom._cols) throw std::invalid_argument("indices must be the same length as the number of rows in the matrix");
}

template<typename T>
BandedMat<T> BandedMat<T>::create(size_t rows, size_t numDiagonals, const Vec<int32_t> &indices) {
    return BandedMat<T>(Mat<T>::create(rows, numDiagonals), indices);
}

template <typename T>
__global__ void mapDenseToBandedKernel(
    const DeviceData2d<T> dense,
    DeviceData2d<T> banded,
    const int32_t* __restrict__ indices
) {
    if (const GridInd2d bandedInd; bandedInd < banded){

        if (const DenseInd denseInd(bandedInd, indices); denseInd >= dense) banded[bandedInd] = NAN;
        else banded[bandedInd] = dense[denseInd];
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
        denseMat.toKernel(),
        this->toKernel(),
        this->_indices.toKernel()
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}


template class BandedMat<float>;
template class BandedMat<double>;
template class BandedMat<size_t>;
template class BandedMat<int32_t>;
template class BandedMat<unsigned char>;