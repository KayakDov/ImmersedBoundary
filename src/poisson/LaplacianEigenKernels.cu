
/**
 * @file LaplacianEigenKernels.cuh
 * @brief Analytical Eigen-decomposition kernels for 1D Discrete Laplacians.
 *
 * Provides CUDA kernels for computing eigenvectors and eigenvalues for various
 * boundary condition combinations (Dirichlet and Neumann).
 * * To ensure ease of use, these kernels calculate their own spatial offsets and
 * frequency denominators based on the 'isStaggered' boolean. The caller only
 * needs to provide the target device arrays and the physical scaling constants.
 */
//TODO: declare kernels in .h file.
#ifndef LAPLACIAN_EIGEN_KERNELS_CUH
#define LAPLACIAN_EIGEN_KERNELS_CUH

#include <cmath>
#include <cuda_runtime.h>
#include "deviceArrays/headers/DeviceData.cuh"
#include "deviceArrays/headers/handle.h"
#include "deviceArrays/headers/SquareMat.h"
#include "poisson/BoundaryCondition.cuh"

#ifndef PI_CONST
#define PI_CONST
/** * @brief High-precision constexpr PI for spectral calculations.
 */
template<typename T> __device__  constexpr T PI = T(3.14159265358979323846L);
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
void BoundaryPair<T>::generateEigen(cudaStream_t stream, SquareMat<T> eVecs, Vec<T> eVals) const {

    KernelPrep vecKP = eVecs.kernelPrep();
    KernelPrep valKP = eVals.kernelPrep();

    T minFourOvDe = -4 * start.inverseDeltaSquared;
    bool isNodeCent = start.isNodeCentered();

    if (start.isNeumann && end.isNeumann) {
        eigenMatLKernel_NN<<<vecKP.numBlocks, vecKP.threadsPerBlock, 0, stream>>>(eVecs.toKernel2d(), isNodeCent);
        eigenValLKernel_NN<<<vecKP.numBlocks, valKP.threadsPerBlock, 0, stream>>>(eVals.toKernel1d(), minFourOvDe, isNodeCent);
    } else if (start.isNeumann && end.isDirichlet()) {
        eigenMatLKernel_ND<<<vecKP.numBlocks, vecKP.threadsPerBlock, 0, stream>>>(eVecs.toKernel2d(), isNodeCent);
        eigenValLKernel_ND<<<vecKP.numBlocks, valKP.threadsPerBlock, 0, stream>>>(eVals.toKernel1d(), minFourOvDe, isNodeCent);
    } else if (start.isDirichlet() && end.isNeumann) {
        eigenMatLKernel_DN<<<vecKP.numBlocks, vecKP.threadsPerBlock, 0, stream>>>(eVecs.toKernel2d(), isNodeCent);
        eigenValLKernel_DN<<<vecKP.numBlocks, valKP.threadsPerBlock, 0, stream>>>(eVals.toKernel1d(), minFourOvDe, isNodeCent);
    } else {
        eigenMatLKernel_DD<<<vecKP.numBlocks, vecKP.threadsPerBlock, 0, stream>>>( eVecs.toKernel2d(), isNodeCent);
        eigenValLKernel_DD<<<vecKP.numBlocks, valKP.threadsPerBlock, 0, stream>>>(eVals.toKernel1d(), minFourOvDe, isNodeCent);
    }

    CHECK_CUDA_ERROR (cudaGetLastError());
}



template class BoundaryPair<float>;
template class BoundaryPair<double>;

#endif // LAPLACIAN_EIGEN_KERNELS_CUH