/**
 * @file LaplacianEigenKernels.cuh
 * @brief Analytical Eigen-decomposition kernels for 1D Discrete Laplacians.
 * * Provides CUDA kernels for computing eigenvectors and eigenvalues for various
 * boundary condition combinations. Supports both Node-Centered and Staggered
 * grid alignments via a boolean flag.
 */

#ifndef LAPLACIAN_EIGEN_KERNELS_CUH
#define LAPLACIAN_EIGEN_KERNELS_CUH

#include <cuda_runtime.h>
#include "deviceArrays/headers/DeviceData.cuh"

#ifndef PI_CONST
#define PI_CONST
/** @brief High-precision constexpr PI for spectral calculations. */
template<typename T> __device__ __host__ constexpr T PI = T(3.14159265358979323846L);
#endif

// =============================================================================
// DIRICHLET - DIRICHLET (DD)
// =============================================================================

/**
 * @brief Computes eigenvectors for DD boundary conditions.
 * * Formula: $$V_{jk} = \sqrt{\frac{2}{N+1}} \sin\left(\frac{\pi(j + 1 - \text{off})(k+1)}{N+1}\right)$$
 * Where $\text{off} = 0.5$ for staggered grids and $0.0$ for node-centered.
 * * @tparam T          Floating-point precision.
 * @param eVecs       Output 2D device array for eigenvectors.
 * @param den         The denominator term: $1 / (N + 1)$.
 * @param isStaggered True if using a staggered (cell-centered) grid.
 */
template<typename T>
__global__ void eigenMatLKernel_DD(DeviceData2d<T> eVecs, const T den, bool isStaggered) {
    if (const GridInd2d ind; ind < eVecs) {
        const T off = isStaggered ? T(0.5) : T(0.0);
        eVecs[ind] = std::sqrt(2 * den) * std::sin((ind.row + 1 - off) * (ind.col + 1) * PI<T> * den);
    }
}

/**
 * @brief Computes eigenvalues for DD boundary conditions.
 * * Formula: $$\lambda_k = - \frac{4}{\Delta^2} \sin^2\left(\frac{\pi(k+1)}{2(N+1)}\right)$$
 * * @tparam T                  Floating-point precision.
 * @param eVals               Output 1D device array for eigenvalues.
 * @param minFourOverDeltaSq  The coefficient $-4 / \Delta^2$.
 * @param piOverTwoNPlus1     The factor $\pi / (2(N + 1))$.
 */
template<typename T>
__global__ void eigenValLKernel_DD(DeviceData1d<T> eVals, const T minFourOverDeltaSq, const T piOverTwoNPlus1) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < eVals.cols) {
        T s = std::sin((idx + 1) * piOverTwoNPlus1);
        eVals[idx] = s * s * minFourOverDeltaSq;
    }
}

// =============================================================================
// NEUMANN - NEUMANN (NN)
// =============================================================================

/**
 * @brief Computes eigenvectors for NN boundary conditions.
 * * Formula: $$V_{jk} = \sqrt{\frac{\alpha_k}{N}} \cos\left(\frac{\pi k (j + 0.5 - \text{off})}{N}\right)$$
 * * @tparam T          Floating-point precision.
 * @param eVecs       Output 2D device array for eigenvectors.
 * @param invN        The inverse of the number of points $1/N$.
 * @param isStaggered True if using a staggered grid.
 */
template<typename T>
__global__ void eigenMatLKernel_NN(DeviceData2d<T> eVecs, const T invN, bool isStaggered) {
    if (const GridInd2d ind; ind < eVecs) {
        const T off = isStaggered ? T(0.0) : T(0.5);
        T norm = (ind.col == 0) ? std::sqrt(invN) : std::sqrt(2 * invN);
        eVecs[ind] = norm * std::cos(PI<T> * ind.col * (ind.row + 0.5f - off) * invN);
    }
}

/**
 * @brief Computes eigenvalues for NN boundary conditions.
 * * Formula: $$\lambda_k = - \frac{4}{\Delta^2} \sin^2\left(\frac{\pi k}{2N}\right)$$
 */
template<typename T>
__global__ void eigenValLKernel_NN(DeviceData1d<T> eVals, const T minFourOverDeltaSq, const T piOverTwoN) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < eVals.cols) {
        T s = std::sin(idx * piOverTwoN);
        eVals[idx] = s * s * minFourOverDeltaSq;
    }
}

// =============================================================================
// DIRICHLET - NEUMANN (DN)
// =============================================================================

/**
 * @brief Computes eigenvectors for DN boundary conditions.
 * * Formula: $$V_{jk} = \sqrt{\frac{2}{N}} \sin\left(\frac{\pi (j + 1 - \text{off}) (k + 0.5)}{N}\right)$$
 * * @tparam T          Floating-point precision.
 * @param eVecs       Output 2D device array for eigenvectors.
 * @param invN        The inverse of the number of points $1/N$.
 * @param isStaggered True if using a staggered grid.
 */
template<typename T>
__global__ void eigenMatLKernel_DN(DeviceData2d<T> eVecs, const T invN, bool isStaggered) {
    if (const GridInd2d ind; ind < eVecs) {
        const T off = isStaggered ? T(0.5) : T(0.0);
        eVecs[ind] = std::sqrt(2 * invN) * std::sin(PI<T> * (ind.row + 1 - off) * (ind.col + 0.5f) * invN);
    }
}

/**
 * @brief Computes eigenvalues for DN boundary conditions.
 * * Formula: $$\lambda_k = - \frac{4}{\Delta^2} \sin^2\left(\frac{\pi (k + 0.5)}{2N}\right)$$
 */
template<typename T>
__global__ void eigenValLKernel_DN(DeviceData1d<T> eVals, const T minFourOverDeltaSq, const T piOverTwoN) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < eVals.cols) {
        T s = std::sin((idx + 0.5f) * piOverTwoN);
        eVals[idx] = s * s * minFourOverDeltaSq;
    }
}

// =============================================================================
// NEUMANN - DIRICHLET (ND)
// =============================================================================

/**
 * @brief Computes eigenvectors for ND boundary conditions.
 * * Formula: $$V_{jk} = \sqrt{\frac{2}{N}} \cos\left(\frac{\pi (j + 0.5 - \text{off}) (k + 0.5)}{N}\right)$$
 */
template<typename T>
__global__ void eigenMatLKernel_ND(DeviceData2d<T> eVecs, const T invN, bool isStaggered) {
    if (const GridInd2d ind; ind < eVecs) {
        const T off = isStaggered ? T(0.5) : T(0.0);
        eVecs[ind] = std::sqrt(2 * invN) * std::cos(PI<T> * (ind.row + 0.5f - off) * (ind.col + 0.5f) * invN);
    }
}

/** * @brief Computes eigenvalues for ND boundary conditions.
 * @note Spectrum is identical to DN.
 */
template<typename T>
__global__ void eigenValLKernel_ND(DeviceData1d<T> eVals, const T minFourOverDeltaSq, const T piOverTwoN) {
    eigenValLKernel_DN<T>(eVals, minFourOverDeltaSq, piOverTwoN);
}

#endif // LAPLACIAN_EIGEN_KERNELS_CUH