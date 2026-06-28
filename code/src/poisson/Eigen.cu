//
// Created by usr on 6/11/26.
//

#include "Eigen.cuh"
#include "BoundaryConfig.cuh"
#include "Laplacian1d.cuh"
#include "deviceArrays/headers/sparse/BandedMat.h"
#include "deviceArrays/headers/SquareMat.h"
#include "deviceArrays/headers/sparse/Diagonal.h"
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
void Eigen<T>::generateEigen(Handle& hand, SquareMat<T>& eVecs, SquareMat<T>& eVecsInv, Vec<T>& eVals, const UniformSegment<T>& seg) {

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
///
///
template <typename T>
__global__ void setSymetrizationMatrix(DeviceData1d<T> symnetrizationBand, Delta1d<T> delta) {
    if (size_t i = idx(); i < symnetrizationBand._cols) {
        symnetrizationBand[i] = sqrt(delta[i] + delta[i + 1]);
    }
}

/**
 * Computes the matrix S = -D L D^{-1}, with D = diag(sqrt(axisSegment.delta[i] + axisSegment.delta[i + 1])).
 * @tparam T
 * @param mat The dense matrix S will be stored here.
 * @param primaryDiag  The primary diagonal of S.
 * @param subDiag The sub diagoanl of S.
 * @param superDiag The super diagonal of S.
 * @param axisSegment
 */
template <typename T>
__global__ void setSymetricMatrix(
    DeviceData1d<T> primaryDiag,
    DeviceData1d<T> subDiag,
    DeviceData1d<T> superDiag,
    const VariableSegment<T> axisSegment
) {
    size_t i = idx();
    if (i >= primaryDiag.cols) return;

    // d can be safely computed for all valid nodes
    auto d = sqrt(axisSegment.delta[i] + axisSegment.delta[i + 1]);

    auto& main = primaryDiag[i];

    if (i == 0) {
        axisSegment.start.setL(main, superDiag[i]);
        auto dp = sqrt(axisSegment.delta[i + 1] + axisSegment.delta[i + 2]);
        superDiag[i] *= -d / dp;
    }
    else if (i < axisSegment.numNodes - 1) {
        axisSegment.setInteriorL(main, subDiag[i - 1], superDiag[i], i);
        auto dp = sqrt(axisSegment.delta[i + 1] + axisSegment.delta[i + 2]);
        auto dm = sqrt(axisSegment.delta[i - 1] + axisSegment.delta[i]);

        subDiag[i - 1] *= -d / dm;
        superDiag[i] *= -d / dp;
    }
    else if (i == axisSegment.numNodes - 1) {
        axisSegment.end.setL(main, subDiag[i - 1]);
        auto dm = sqrt(axisSegment.delta[i - 1] + axisSegment.delta[i]);
        subDiag[i - 1] *= -d / dm;
    }
    main *= -1;
}

/**
 * Computes the eigen matrix and it's inverse.
 * @tparam T
 * @param eigen The eigen matrix for the symmetrical system, S, goes here.  S = -DLD^{-1}.
 * Where D = diag(sqrt(axisSegment.delta[i] + axisSegment.delta[i + 1])).  This will be replaced with the eigen matrix
 * for L with V_L = D^{-1}V_S.
 * @param eigenInv The inverse of the eigen matrix will be stored here, with V_L^{-1} = V_S^T D^{-1}
 * @param axisSegment
 */
template<typename T>
__global__ void mapEigenSymmToEigenLapInv(const DeviceData2d<T> eigen, DeviceData2d<T> eigenInv, const VariableSegment<T> axisSegment) {
    if (GridInd2d ind; ind < eigenInv) {
        // V_L^{-1}[i, j] = V_S[j, i] * D_j
        T d_col = sqrt(axisSegment.delta[ind.col] + axisSegment.delta[ind.col + 1]);
        eigenInv[ind] = eigen(ind.col, ind.row) * d_col;
    }
}

template<typename T>
__global__ void mapEigenSymmToEigenLap(DeviceData2d<T> eigen, const VariableSegment<T> axisSegment) {
    if (GridInd2d ind; ind < eigen) {
        // V_L[i, j] = V_S[i, j] / D_i
        T d_row = sqrt(axisSegment.delta[ind.row] + axisSegment.delta[ind.row + 1]);
        eigen[ind] /= d_row;
    }
}

template<typename T>
void Eigen<T>::generateEigen(Handle& hand, SquareMat<T>& eVecs, SquareMat<T>& eVecsInv, Vec<T>& eVals, const VariableSegment<T> &axisSegment) {

    eVecs.fill(0, hand);

    auto kp1d = eVals.kernelPrep();
    setSymetricMatrix<<<kp1d.numBlocks, kp1d.threadsPerBlock, 0, hand>>>(
        eVecs.diag(0).toKernel1d(),
        eVecs.diag(-1).toKernel1d(),
        eVecs.diag(1).toKernel1d(),
        axisSegment
    );

    eVecs.eigenSPD(eVals, hand);

    auto kp2d = eVecs.kernelPrep();
    mapEigenSymmToEigenLapInv<<<kp2d.numBlocks, kp2d.threadsPerBlock, 0, hand>>>(
        eVecs.toKernel2d(),
        eVecsInv.toKernel2d(),
        axisSegment
    );

    mapEigenSymmToEigenLap<<<kp2d.numBlocks, kp2d.threadsPerBlock, 0, hand>>>(
        eVecs.toKernel2d(),
        axisSegment
    );

     eVals.mult(GPUScalar<T>::get(-1), &hand);
}

template<typename T>
template<typename axisSegmentT>
void Eigen<T>::generateEigen(Handle &hand, Mat<T> eigins, const axisSegmentT &axisSegment) {
    auto vecs = eigins.sqSubMat(0, 0, eigins._rows);
    auto vecsInv = eigins.sqSubMat(0, eigins._rows, eigins._rows);
    auto vals = eigins.col(eigins._cols - 1);
    generateEigen(hand, vecs, vecsInv, vals, axisSegment);
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
template <typename ResultType, typename F, typename BoundaryConfigT>
void createUnique(const BoundaryConfigT& boundaryConfig, std::shared_ptr<ResultType> (&outputs)[3], Handle* hands3, Event* events2, F factory) {

    outputs[0] = std::make_shared<ResultType>(factory(boundaryConfig.x, hands3[0]));
    if (boundaryConfig.y == boundaryConfig.x)  outputs[1] = outputs[0];
    else {
        outputs[1] = std::make_shared<ResultType>(factory(boundaryConfig.y, hands3[1]));
        events2[0].record(hands3[1]);
        events2[0].hold(hands3[0]);

    }

    if (boundaryConfig.dim().numDims() == 3) {
        if (boundaryConfig.z == boundaryConfig.x)  outputs[2] = outputs[0];
        else if (boundaryConfig.z == boundaryConfig.y) outputs[2] = outputs[1];
        else {
            outputs[2] = std::make_shared<ResultType>(factory(boundaryConfig.z, hands3[2]));
            events2[1].record(hands3[2]);
            events2[1].hold(hands3[0]);
        }
    } else outputs[2] = nullptr;

}
/**
* Generates, including memory allocation, eigen values and vectors.  The matrices pointed to, that are retruned,
* hold the values in the last column, and the vectors in the first nxn cells.
* @param boundary The boundary configuration.
* @param hands3 Used to create the different vectors in parrallel.  The number of handles should be equal to the number of dimentisons.
* @param events The number of events should be equal to the number of dimesnions.
* @param preAllocatedForL_iX3
* @return pointers to matrices containing the eigen values and vectors.
 */
template<typename Real>
template<typename BoundaryConfigT>
void Eigen<Real>::generateEigen(const BoundaryConfigT& boundary, Handle *hands3, Event *events, std::shared_ptr<Mat<Real>> (&preAllocatedForL_iX3)[3]) {

    createUnique(
        boundary,
        preAllocatedForL_iX3,
        hands3,
        events,
        [](const auto& AxisSegmentT, Handle& hand) {

            size_t numCols = AxisSegmentT.numNodes + 1;
            if constexpr (std::is_same_v<std::decay_t<decltype(AxisSegmentT)>, VariableSegment<Real>>)
                numCols = AxisSegmentT.numNodes * 2 + 1;

            auto mat = Mat<Real>::create(AxisSegmentT.numNodes, numCols);
            Eigen<Real>::generateEigen(hand, mat, AxisSegmentT);
            return mat;
        }
    );

}

template<typename T>
Eigen<T>::Eigen(const XYZ<Vec<T>> &vals, const KroneckerTriplet<T>& vecs, const KroneckerTriplet<T>& vecsInv) :
    vals(vals), vecs(vecs), vecsInv(vecsInv) {}

template<typename T>
template<typename BoundaryConfigT>
Eigen<T> Eigen<T>::make(const BoundaryConfigT& boundary, Handle* hands3, Event* events2) {

    bool is3d = boundary.dim().numDims() == 3;
    std::shared_ptr<Mat<T>> eigen[3];
    generateEigen(boundary, hands3, events2, eigen);

    XYZ<Vec<T>> vals(eigen[0]->lastCol(), eigen[1]->lastCol(), is3d ? eigen[2]->lastCol() : SimpleArray<T>::empty());

    XYZ<SquareMat<T>> vecs(
        eigen[0]->sqSubMatFirstBiggest(),
        eigen[1]->sqSubMatFirstBiggest(),
        is3d ? eigen[2]->sqSubMatFirstBiggest() : SquareMat<T>::empty()
    );

    constexpr bool orthoX = !std::is_same_v<std::decay_t<decltype(boundary.x)>, VariableSegment<T>>;
    constexpr bool orthoY = !std::is_same_v<std::decay_t<decltype(boundary.y)>, VariableSegment<T>>;
    constexpr bool orthoZ = !std::is_same_v<std::decay_t<decltype(boundary.z)>, VariableSegment<T>>;

    XYZ<SquareMat<T>> vecsInv(
        orthoX ? vecs.x : eigen[0]->sqSubMat(0, vecs.x._cols, vecs.x._cols ),
        orthoY ? vecs.y : eigen[1]->sqSubMat(0, vecs.y._cols, vecs.y._cols ),
        !is3d || orthoZ ? vecs.z : eigen[2]->sqSubMat(0, vecs.z._cols, vecs.z._cols )
    );

    KroneckerTriplet<T> kt(vecs, {false, false, false});
    KroneckerTriplet<T> ktInv(vecsInv, {orthoX, orthoY, orthoZ});

    return Eigen<T>(vals, kt, ktInv);
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

#define INSTANTIATE_EIGEN_MAKE_BOUNDARY(Real, SegX, SegY, SegZ) \
template Eigen<Real> Eigen<Real>::make<BoundaryConfig<Real, SegX, SegY, SegZ>>( \
    const BoundaryConfig<Real, SegX, SegY, SegZ>&, Handle*, Event*);

#define INSTANTIATE_EIGEN_ALL(Real) \
INSTANTIATE_EIGEN_MAKE_BOUNDARY(Real, UniformSegment<Real>,  UniformSegment<Real>,  UniformSegment<Real>)  \
INSTANTIATE_EIGEN_MAKE_BOUNDARY(Real, UniformSegment<Real>,  UniformSegment<Real>,  VariableSegment<Real>) \
INSTANTIATE_EIGEN_MAKE_BOUNDARY(Real, UniformSegment<Real>,  VariableSegment<Real>, UniformSegment<Real>)  \
INSTANTIATE_EIGEN_MAKE_BOUNDARY(Real, UniformSegment<Real>,  VariableSegment<Real>, VariableSegment<Real>) \
INSTANTIATE_EIGEN_MAKE_BOUNDARY(Real, VariableSegment<Real>, UniformSegment<Real>,  UniformSegment<Real>)  \
INSTANTIATE_EIGEN_MAKE_BOUNDARY(Real, VariableSegment<Real>, UniformSegment<Real>,  VariableSegment<Real>) \
INSTANTIATE_EIGEN_MAKE_BOUNDARY(Real, VariableSegment<Real>, VariableSegment<Real>, UniformSegment<Real>)  \
INSTANTIATE_EIGEN_MAKE_BOUNDARY(Real, VariableSegment<Real>, VariableSegment<Real>, VariableSegment<Real>) \

INSTANTIATE_EIGEN_ALL(float)
INSTANTIATE_EIGEN_ALL(double)
