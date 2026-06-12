//
// Created by usr on 6/11/26.
//

#include "Eigen.cuh"
#include "BoundaryConfig.cuh"
#include "Laplacian1d.cuh"
#include "deviceArrays/headers/sparse/BandedMat.h"
#include "deviceArrays/headers/SquareMat.h"
#include "kronecker/KroneckerTriplet.h"
#include "math/XYZ.cuh"
#include "solvers/Event.h"


#ifndef PI_CONST
#define PI_CONST
/** * @brief High-precision constexpr PI for spectral calculations.
 */
template<typename T> __device__  T PI = T(3.14159265358979323846L);
#endif

template<typename T>
__device__ __forceinline__ T _sqrt(T x) {
    if constexpr (std::is_same_v<T, float>) return sqrtf(x);
    else return sqrt(x);
}
template<typename T>
__device__ __forceinline__ T _rsqrt(T x) {
    if constexpr (std::is_same_v<T, float>) return __frsqrt_rn(x);
    else return rsqrt(x);
}


// =============================================================================
// DIRICHLET - DIRICHLET (DD)
// =============================================================================

/**
 * @brief Computes eigenvectors for Dirichlet-Dirichlet boundary conditions.
 *
 * Mathematically maps to the Discrete Sine Transform (DST). Automatically shifts
 * the spatial evaluation and effective domain length based on the grid geometry.
 *
 * @tparam T Floating-point precision.
 * @param eVecs 2D device array to store the orthogonal basis. The number of columns defines N.
 * @param isNodeCentered True if the grid nodes are node-centered; False if cell centered or staggered.
 */
template<typename T>
__global__ void eigenMatLKernel_DD(DeviceData2d<T> eVecs, bool isNodeCentered) {
    if (const GridInd2d ind; ind < eVecs) {
        const T den = 1.0/(eVecs.rows + isNodeCentered);
        T normalize = (!isNodeCentered && ind.col == eVecs.cols - 1) ? _rsqrt<T>(eVecs.rows) : _sqrt<T>(den * 2);
        eVecs[ind] =  normalize *
            sin(PI<T> * (ind.row + (isNodeCentered ? 1 : 0.5)) * (ind.col + 1) * den);
    }
}

/**
 * @brief Computes eigenvalues for Dirichlet-Dirichlet boundary conditions.
 *
 * @tparam T Floating-point precision.
 * @param eVals 1D device array to store the Laplacian spectrum. The size defines N.
 * @param minFourOverDeltaSq The physical grid coefficient: $$-\frac{4}{\Delta^2}$$
 * @param isNodeCentered True if the grid nodes are node-centered; False if they are cell centered / staggered.
 */
template<typename T>
__global__ void eigenValLKernel_DD(DeviceData1d<T> eVals, const T minFourOverDeltaSq, bool isNodeCentered) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < eVals.cols) {
        const T den = 1.0/(eVals.cols + isNodeCentered * 1);
        const T sineComponent = sin(PI<T> * (idx + 1) * 0.5 * den);
        eVals[idx] = sineComponent * sineComponent * minFourOverDeltaSq;
    }
}

// =============================================================================
// NEUMANN - NEUMANN (NN)
// =============================================================================

/**
 * @brief Computes eigenvectors for Neumann-Neumann boundary conditions.
 *
 * Mathematically maps to the Discrete Cosine Transform (DCT). Adjusts between
 * DCT-I (node-centered) and DCT-II (staggered) automatically.
 *
 * @tparam T Floating-point precision.
 * @param eVecs 2D device array to store the orthogonal basis.
 * @param isNodeCentered True if the gird is node centered, false if it's staggered / cell centered.
 */
template<typename T>
__global__ void eigenMatLKernel_NN(DeviceData2d<T> eVecs, bool isNodeCentered) {
    const GridInd2d ind;
    if (ind.row >= eVecs.rows) return;

    T den = 1.0/eVecs.rows;
    if (ind.col == 0) eVecs[ind] = _rsqrt<T>(eVecs.rows);
    else if (ind.col < eVecs.cols)
        eVecs[ind] = _sqrt<T>(2.0 * den) * cos(PI<T> * ind.col * (ind.row  + 0.5) * den);
}


/**
 * @brief Computes eigenvalues for Neumann-Neumann boundary conditions.
 *
 * @tparam T Floating-point precision.
 * @param eVals 1D device array to store the Laplacian spectrum.
 * @param minFourOverDeltaSq The physical grid coefficient: $$-\frac{4}{\Delta^2}$$
 * @param isNodeCentered True if the grid nodes are node-centered; False if staggered / cell centered.
 */
template<typename T>
__global__ void eigenValLKernel_NN(DeviceData1d<T> eVals, const T minFourOverDeltaSq, bool isNodeCentered) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < eVals.cols) {
        const T den = 0.5/(eVals.cols);
        const T sineComponent = sin(PI<T> * idx * den);
        eVals[idx] = sineComponent * sineComponent * minFourOverDeltaSq;
    }
}

// =============================================================================
// DIRICHLET - NEUMANN (DN)
// =============================================================================


/**
 * @brief Computes eigenvectors for Dirichlet-Neumann boundary conditions.
 *
 * Implements a mixed spectral basis. The Dirichlet condition is enforced at
 * the lower spatial index, and Neumann at the higher index.
 *
 * @tparam T Floating-point precision.
 * @param eVecs 2D device array to store the orthogonal basis.
 * @param isNodeCentered True if the grid nodes are node-centered; False if staggered / cell centered.
 */
template<typename T>
__global__ void eigenMatLKernel_DN(DeviceData2d<T> eVecs, bool isNodeCentered) {
    if (const GridInd2d ind; ind < eVecs) {

        eVecs[ind] = _sqrt<T>(2 / (eVecs.rows + isNodeCentered * 0.5)) *
            sin(PI<T> * (ind.row + 0.5 + isNodeCentered * 0.5) * (2 * ind.col + 1) / (2 * eVecs.rows + isNodeCentered * 1));
    }
}

/**
 * @brief Computes eigenvalues for Dirichlet-Neumann boundary conditions.
 *
 * @tparam T Floating-point precision.
 * @param eVals 1D device array to store the Laplacian spectrum.
 * @param minFourOverDeltaSq The physical grid coefficient: $$-\frac{4}{\Delta^2}$$
 */
template<typename T>
__global__ void eigenValLKernel_DN(DeviceData1d<T> eVals, const T minFourOverDeltaSq, bool isNodeCentered) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < eVals.cols) {
        const T sineComponent = sin(PI<T> * (idx + 0.5)/(2 * eVals.cols + isNodeCentered));
        eVals[idx] = sineComponent * sineComponent * minFourOverDeltaSq;
    }
}

// =============================================================================
// NEUMANN - DIRICHLET (ND)
// =============================================================================

/**
 * @brief Computes eigenvectors for Neumann-Dirichlet boundary conditions.
 *
 * Implements a mixed spectral basis with the boundary conditions flipped
 * spatially relative to the DN setup.
 *
 * @tparam T Floating-point precision.
 * @param eVecs 2D device array to store the orthogonal basis.
 * @param isNodeCentered True if the grid nodes are node-centered; False if they are cell-centered / staggered.
 */
template<typename T>
__global__ void eigenMatLKernel_ND(DeviceData2d<T> eVecs, bool isNodeCentered) {
    if (const GridInd2d ind; ind < eVecs) {

        eVecs[ind] = _sqrt<T>(2 / (eVecs.rows + isNodeCentered * 0.5))
            * cos((PI<T> * (ind.row + 0.5) * (2 * ind.col + 1) / (2 * eVecs.rows + isNodeCentered * 1)));
    }
}

/** * @brief Computes eigenvalues for Neumann-Dirichlet boundary conditions.
 * * @note The energy spectrum for ND is mathematically identical to DN.
 * This kernel acts as a direct passthrough to ensure a consistent API.
 *
 * @tparam T Floating-point precision.
 * @param eVals 1D device array to store the Laplacian spectrum.
 * @param minFourOverDeltaSq The physical grid coefficient: $$-\frac{4}{\Delta^2}$$
 * @param isStaggered True if the grid nodes are cell-centered; False if node-centered.
 */
template<typename T>
__global__ void eigenValLKernel_ND(DeviceData1d<T> eVals, const T minFourOverDeltaSq, bool isNodeCentered) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < eVals.cols) {
        const T sineComponent = sin(PI<T> * (idx + 0.5)/(2 * eVals.cols + isNodeCentered));
        eVals[idx] = sineComponent * sineComponent * minFourOverDeltaSq;
    }
}

template<typename T>
void Eigen<T>::generateEigen(Handle& hand, SquareMat<T> eVecs, Vec<T> eVals, const UniformSegment<T>& seg) {

    KernelPrep vecKP = eVecs.kernelPrep();
    KernelPrep valKP = eVals.kernelPrep();

    T minFourOvDe = -4 * seg.start.inverseDeltaSquared;
    bool isNodeCent = seg.start.isNodeCentered();

    if (seg.start.isNeumann && seg.end.isNeumann) {
        eigenMatLKernel_NN<<<vecKP.numBlocks, vecKP.threadsPerBlock, 0, hand>>>(eVecs.toKernel2d(), isNodeCent);
        eigenValLKernel_NN<<<vecKP.numBlocks, valKP.threadsPerBlock, 0, hand>>>(eVals.toKernel1d(), minFourOvDe, isNodeCent);
    } else if (seg.start.isNeumann && seg.end.isDirichlet()) {
        eigenMatLKernel_ND<<<vecKP.numBlocks, vecKP.threadsPerBlock, 0, hand>>>(eVecs.toKernel2d(), isNodeCent);
        eigenValLKernel_ND<<<vecKP.numBlocks, valKP.threadsPerBlock, 0, hand>>>(eVals.toKernel1d(), minFourOvDe, isNodeCent);
    } else if (seg.start.isDirichlet() && seg.end.isNeumann) {
        eigenMatLKernel_DN<<<vecKP.numBlocks, vecKP.threadsPerBlock, 0, hand>>>(eVecs.toKernel2d(), isNodeCent);
        eigenValLKernel_DN<<<vecKP.numBlocks, valKP.threadsPerBlock, 0, hand>>>(eVals.toKernel1d(), minFourOvDe, isNodeCent);
    } else {
        eigenMatLKernel_DD<<<vecKP.numBlocks, vecKP.threadsPerBlock, 0, hand>>>( eVecs.toKernel2d(), isNodeCent);
        eigenValLKernel_DD<<<vecKP.numBlocks, valKP.threadsPerBlock, 0, hand>>>(eVals.toKernel1d(), minFourOvDe, isNodeCent);
    }

    CHECK_CUDA_ERROR (cudaGetLastError());
}


////////////////////////////////////////////////////Remainder of Eigen methods//////////////////////////////////////////
///
template<typename T>
void Eigen<T>::generateEigen(Handle& hand, SquareMat<T> eVecs, Vec<T> eVals, const VariableSegment<T> &axisSegment) {

    size_t n = eVecs._cols;
    auto buffer = Mat<T>::create(n, n + 3);
    auto cols3 = buffer.subMat(0,0,n, 3);
    TriDiagonal<T> triDiag(cols3, hand);

    Laplacian1d<T>::create(axisSegment, triDiag, hand);
    auto dense =buffer.sqSubMat(0, 3, n);

    triDiag.getDense(dense, hand);

    dense.eigen(eVals, &eVecs, hand);
}

template<typename T>
template<typename axisSegmentT>
void Eigen<T>::generateEigen(Handle &hand, Mat<T> eigins, const axisSegmentT &axisSegment) {
    auto vecs = eigins.sqSubMat(0, 0, eigins._rows);
    auto vals = eigins.col(eigins._cols - 1);
    generateEigen(hand, vecs, vals, axisSegment);
}

/**
* True if i is the first index that these conditions appear at.
* @param i The index to be checked.
* @return the value of the index repeated, or -1 if this is the first appearence.
*/
template <typename Real>
int repeat(const BoundaryConfig<Real>& boundary, int i) {
    for (size_t j = 0; j < i; ++j) if (boundary[j] == boundary[i]) return j;
    return -1;

}

/**
 * @brief Efficiently creates unique objects based on 1D Laplacian boundary conditions.
 *
 * This utility iterates through the X, Y, and Z dimensions of a Laplacian setup.
 * If the boundary conditions (dimensions, spacing, and types) for two dimensions
 * are identical, the function reuses the existing object by sharing ownership
 * via std::shared_ptr. This prevents redundant memory allocations and unnecessary
 * re-computation of Laplacian matrices or eigenvalue vectors.
 *
 * @tparam ResultType The type of object to be created (e.g., Mat<T> or Vec<T>).

 * @param[out] outputs  An array of 3 shared_ptrs to be populated with the results.
 * @param[in]  factory  A factory function: `ResultType factory(const LaplacianConditions<T>&)`.
 * * @note Because ResultType is wrapped in a shared_ptr, ResultType does not need
 * to be assignable, which is critical for classes with const data members.
 */
template <typename Real, typename ResultType, typename F>
void createUnique(const BoundaryConfig<Real>& boundaryConfig, std::shared_ptr<ResultType> (&outputs)[3], F factory) {
    size_t numDim = boundaryConfig.dim().numDims();
    for (size_t i = 0; i < numDim; ++i) {
        int repeatInd = repeat(boundaryConfig, i);
        if (repeatInd == -1) outputs[i] = std::make_shared<ResultType>(factory(boundaryConfig[i]));
        else outputs[i] = outputs[repeatInd];
    }
    if (numDim == 2) outputs[2] = nullptr;
}

/**
* Generates, including memory allocation, eigen values and vectors.  The matrices pointed to, that are retruned,
* hold the values in the last column, and the vectors in the first nxn cells.
* @param hands3 Used to create the different vectors in parrallel.  The number of handles should be equal to the number of dimentisons.
* @param events The number of events should be equal to the number of dimesnions.
* @param preAllocatedForL_iX3
* @return pointers to matrices containing the eigen values and vectors.
*/
template<typename Real>
void Eigen<Real>::generateEigen(const BoundaryConfig<Real>& boundary, Handle *hands3, Event *events, std::shared_ptr<Mat<Real>> (&preAllocatedForL_iX3)[3]) {

    createUnique(
        boundary,
        preAllocatedForL_iX3,
        [](const UniformSegment<Real>& c) {return Mat<Real>::create(c.numNodes, c.numNodes + 1);}
    );

    Eigen<Real>::generateEigen(hands3[0], *(preAllocatedForL_iX3[0]), boundary[0]);

    for (size_t i = 1; i < boundary.dim().numDims(); ++i)
    if (repeat(boundary, i) < 0){
        Eigen<Real>::generateEigen(hands3[i], *(preAllocatedForL_iX3[i]), boundary[i]);
        events[i - 1].record(hands3[i]);
        events[i - 1].hold(hands3[0]);
    }
}


template<typename T>
Eigen<T>::Eigen(const XYZ<Vec<T>> &vals, const XYZ<SquareMat<T>> &vecs) :
    vals(vals), vecs(vecs) {}


template<typename T>
Eigen<T> Eigen<T>::make(const BoundaryConfig<T>& boundary, Handle* hands3, Event* events) {

    bool is3d = boundary.dim().numDims() == 3;
    std::shared_ptr<Mat<T>> eigen[3];
    generateEigen(boundary, hands3, events, eigen);
    XYZ<Vec<T>> vals(eigen[0]->lastCol(), eigen[1]->lastCol(), is3d ? eigen[2]->lastCol() : SimpleArray<T>::empty());
    XYZ<SquareMat<T>> vecs(
        eigen[0]->sqSubMatFirstBiggest(),
        eigen[1]->sqSubMatFirstBiggest(),
        is3d ? eigen[2]->sqSubMatFirstBiggest() : SquareMat<T>::empty()//GPUConst<T>::get(0).matrix(1).sqSubMat(0,0,1)
    );
    return Eigen<T>(vals, vecs);
}

template<typename T>
GridDim Eigen<T>::dim() const {
    return GridDim(vals.y.size(), vals.x.size(), vals.z.size());
}

// Explicit instantiations for the member template (which are ignored by 'template class')
template void Eigen<float>::generateEigen<UniformSegment<float>>(Handle&, Mat<float>, const UniformSegment<float>&);
template void Eigen<float>::generateEigen<VariableSegment<float>>(Handle&, Mat<float>, const VariableSegment<float>&);

template void Eigen<double>::generateEigen<UniformSegment<double>>(Handle&, Mat<double>, const UniformSegment<double>&);
template void Eigen<double>::generateEigen<VariableSegment<double>>(Handle&, Mat<double>, const VariableSegment<double>&);

template class Eigen<float>;
template class Eigen<double>;
