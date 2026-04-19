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

#ifndef LAPLACIAN_EIGEN_KERNELS_CUH
#define LAPLACIAN_EIGEN_KERNELS_CUH

#include <cmath>
#include <cuda_runtime.h>
#include "deviceArrays/headers/DeviceData.cuh"

#ifndef PI_CONST
#define PI_CONST
/** * @brief High-precision constexpr PI for spectral calculations.
 */
template<typename T> __device__ __host__ constexpr T PI = T(3.14159265358979323846L);
#endif

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
        eVecs[ind] = std::sqrt(den * 2) *
            std::sin(PI<T> * (ind.row + (isNodeCentered ? 1 : 0.5)) * (ind.col + 1) * den);
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
        const T sineComponent = std::sin(PI<T> * (idx + 1) * 0.5 * den);
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
 * @param isNodeCentered True if the grid nodes are node-centered; False if staggered.
 */
template<typename T>
__global__ void eigenMatLKernel_NN(DeviceData2d<T> eVecs, bool isNodeCentered) {
    const GridInd2d ind;
    if (ind.row >= eVecs.rows) return;

    T den = 1.0/(eVecs.rows - isNodeCentered);
    if (ind.col == 0) eVecs[ind] = std::sqrt(den);
    else if (ind.col < eVecs.cols)
        eVecs[ind] = std::sqrt(2 * den) * std::cos(PI<T> * ind.col * (ind.row + (!isNodeCentered) * 0.5)* den);

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
        const T den = 1.0/(eVals.cols - isNodeCentered);
        const T sineComponent = std::sin(PI<T> * idx * 0.5 * den);
        eVals[idx] = sineComponent * sineComponent * minFourOverDeltaSq;
    }
}

// =============================================================================
// DIRICHLET - NEUMANN (DN)
// =============================================================================

/**
 * Computes the eigen values for both ND and DN, staggered and node centered, as these are all the same.
 * @param eVals
 */
template<typename T>
__device__ void eigenMixed(DeviceData1d<T> eVals, const T minFourOverDeltaSq) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < eVals.cols) {
        const T sineComponent = std::sin(PI<T> * (idx + 0.5)/(2 * eVals.cols));
        eVals[idx] = sineComponent * sineComponent * minFourOverDeltaSq;
    }
}

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
        const T den = 1.0/eVecs.rows;
        eVecs[ind] = std::sqrt(2 * den) * std::sin((PI<T> * (ind.row + 0.5 + isNodeCentered * 0.5) * (ind.col + 0.5) * 0.5 * den));
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
__global__ void eigenValLKernel_DN(DeviceData1d<T> eVals, const T minFourOverDeltaSq) {
    eigenMixed(eVals, minFourOverDeltaSq);
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
        const T den = 1.0/eVecs.rows;
        eVecs[ind] = std::sqrt(2 * den) * std::cos((PI<T> * (ind.row + (!isNodeCentered) * 0.5) * (ind.col + 0.5) * den));
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
__global__ void eigenValLKernel_ND(DeviceData1d<T> eVals, const T minFourOverDeltaSq) {
    eigenMixed(eVals, minFourOverDeltaSq);
}

#endif // LAPLACIAN_EIGEN_KERNELS_CUH