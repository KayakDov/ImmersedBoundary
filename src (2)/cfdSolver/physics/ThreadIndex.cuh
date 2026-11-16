/**
 * @file ThreadIndex.cuh
 * @ingroup cfd_solver
 * @brief threadGridIndex(): this thread's 1-based (i,j,k) grid index.
 *
 * Every full-volume kernel in this project launches over a 3D grid and
 * needs its thread's (i,j,k) position from CUDA's own
 * blockIdx/blockDim/threadIdx built-ins. Before this file existed, that
 * computation --
 *   int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
 *   int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
 *   int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
 * -- was copy-pasted identically into every such kernel (21 times, across
 * Operators.cuh, Diagnostics.cuh, ElectromagneticForce.cuh, and
 * ElementwiseOps.cuh). It carries no per-axis physical meaning of its own
 * -- unlike Spacing's hx12/hy12/hz12, which this project deliberately
 * keeps as three separate, Fortran-matching pointers (see Operators.cuh's
 * header comment) -- so there is no cross-checking benefit to leaving it
 * unrolled, and every copy was a fresh chance for a pasted-but-not-updated
 * axis: the same shape of mistake as ConvMain_3D_Q2D.f90's
 * `VMz = VMx_Az` typo. Centralized here once instead.
 *
 * Deliberately NOT used by nusseltKernel/nusseltMiddleKernel in
 * Diagnostics.cuh: those two launch over a degenerate (1-wide) axis and
 * rely on reading threadIdx.x/.y/.z directly to guard against the
 * 8x/64x-inflation bug documented on those kernels -- see Diagnostics.cuh.
 */

#ifndef CFDSOLVER_THREADINDEX_CUH
#define CFDSOLVER_THREADINDEX_CUH

#ifdef __CUDACC__

#include "math/XYZ.cuh"

/**
 * @ingroup cfd_solver
 * @brief This thread's 1-based (i,j,k) grid index: .x/.y/.z hold the same
 * i/j/k every full-volume kernel in this project indexes its
 * FieldView/Spacing arguments with.
 * @return The thread's 1-based index, one component per axis.
 */
__device__ __forceinline__ XYZ<int> threadGridIndex() {
    return XYZ<int>(
        (int)(blockIdx.x * blockDim.x + threadIdx.x) + 1,
        (int)(blockIdx.y * blockDim.y + threadIdx.y) + 1,
        (int)(blockIdx.z * blockDim.z + threadIdx.z) + 1
    );
}

#endif //__CUDACC__

#endif //CFDSOLVER_THREADINDEX_CUH
