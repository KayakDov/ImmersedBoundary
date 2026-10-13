/**
 * @file Diagnostics.cuh
 * @ingroup cfd_solver
 * @brief Diagnostics: the kinetic energy, Nusselt number, and convergence
 * reductions, direct port of ekinem_3D.f90 and sublid_3D.f90.
 *
 * Every reduction follows the same two-stage pattern: a per-cell
 * contribution (host+device-callable, independently testable) accumulated
 * by a kernel into a single device scalar via block-level shared-memory
 * reduction plus one atomic operation per block. The accumulator must be
 * zeroed (sum reductions) or set to the field's identity value (max
 * reduction) before launch.
 *
 * Dist2D's convergence check (RNSx/RNSy/RNSz/RTmpr) is computed here over
 * each field's true interior range only (its own FieldView extent), not
 * the one-cell-wider range the original Fortran's literal loop bound used.
 * That wider range only ever compared boundary-derived values (ghost
 * cells, always a mirror or zero of an interior cell, never an independent
 * degree of freedom) against the equivalent in the other field -- a
 * difference of zero or a value entirely determined by the interior
 * comparison already being made. Restricting to the true interior changes
 * no convergence behavior and has no equivalent-shaped region to compare
 * in an unpadded design in the first place.
 */

#ifndef CFDSOLVER_DIAGNOSTICS_CUH
#define CFDSOLVER_DIAGNOSTICS_CUH

#include "Operators.cuh"

/**
 * @ingroup cfd_solver
 * @brief Per-cell diagnostic contributions and their final combination
 * into Nusselt/Nusselt_middle.
 */
class Diagnostics {
public:
    /**
     * @brief One cell's contribution to the volume-integrated kinetic
     * energy. Port of ekinem_3D.f90's Ekinem.
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx1].
     * @param j 1-based y index in [1, Ny1].
     * @param k 1-based z index in [1, Nz1].
     * @param velocityX Current VMx.
     * @param velocityY Current VMy.
     * @param velocityZ Current VMz.
     * @param spacing Mesh spacing.
     * @return This cell's contribution.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real kineticEnergyCell(int i, int j, int k, const FieldView<Real>& velocityX, const FieldView<Real>& velocityY,
                                   const FieldView<Real>& velocityZ, const Spacing<Real>& spacing) {
        Real sumX = BoundaryConditions::readVelocity(velocityX, i, j, k) + BoundaryConditions::readVelocity(velocityX, i - 1, j, k);
        Real sumY = BoundaryConditions::readVelocity(velocityY, i, j, k) + BoundaryConditions::readVelocity(velocityY, i, j - 1, k);
        Real sumZ = BoundaryConditions::readVelocity(velocityZ, i, j, k) + BoundaryConditions::readVelocity(velocityZ, i, j, k - 1);
        return Real(0.125) * (sumX * sumX + sumY * sumY + sumZ * sumZ) * spacing.hx12[i - 1] * spacing.hy12[j - 1] * spacing.hz12[k - 1];
    }

    /**
     * @brief One (j,k) cell's contribution to the wall-averaged Nusselt
     * sum. Port of ekinem_3D.f90's Nusselt, excluding the final
     * `1/AspRa + .../WidRa` combination (see nusseltFromSum()).
     * @tparam Real Floating-point type.
     * @param j 1-based y index in [1, Ny1].
     * @param k 1-based z index in [1, Nz1].
     * @param temperature Current Tmpr.
     * @param bc Temperature's boundary condition.
     * @param spacing Mesh spacing.
     * @return This cell's contribution.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real nusseltCell(int j, int k, const FieldView<Real>& temperature, AxisBoundaries bc, const Spacing<Real>& spacing) {
        Real hp0 = spacing.hpx[0], hp1 = spacing.hpx[1];
        Real divisor = hp0 * hp1 * (hp0 + hp1);
        Real f0 = hp1 * (hp1 + Real(2) * hp0);
        Real f1 = -(hp1 + hp0) * (hp1 + hp0);
        Real f2 = hp0 * hp0;
        Real derivative = (f0 * BoundaryConditions::readTemperature(temperature, 0, j, k, bc)
                          + f1 * BoundaryConditions::readTemperature(temperature, 1, j, k, bc)
                          + f2 * BoundaryConditions::readTemperature(temperature, 2, j, k, bc)) / divisor;
        return derivative * spacing.hy12[j - 1] * spacing.hz12[k - 1];
    }

    /**
     * @brief Combines the Nusselt sum into the final Nusselt number. Port
     * of ekinem_3D.f90's `Nusselt = 1.d0/AspRa + Snu / WidRa`.
     * @tparam Real Floating-point type.
     * @param sum The accumulated nusseltCell() sum.
     * @param aspectRatio The configuration's aspect ratio.
     * @param widthRatio The configuration's width ratio.
     * @return The Nusselt number.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real nusseltFromSum(Real sum, Real aspectRatio, Real widthRatio) {
        return Real(1) / aspectRatio + sum / widthRatio;
    }

    /**
     * @brief One k cell's contribution to the midplane Nusselt sum. Port
     * of ekinem_3D.f90's Nusselt_middle, excluding the final combination.
     * @tparam Real Floating-point type.
     * @param k 1-based z index in [1, Nz1].
     * @param midplaneY The y index Fortran wrote as Ny2/2 -- see
     * Diagnostics.cu's caller for how this is derived from Ny.
     * @param temperature Current Tmpr.
     * @param bc Temperature's boundary condition.
     * @param spacing Mesh spacing.
     * @return This cell's contribution.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real nusseltMiddleCell(int k, int midplaneY, const FieldView<Real>& temperature, AxisBoundaries bc, const Spacing<Real>& spacing) {
        Real derivative = (BoundaryConditions::readTemperature(temperature, 0, midplaneY, k, bc)
                          - BoundaryConditions::readTemperature(temperature, 1, midplaneY, k, bc)) / spacing.hpx[0];
        return derivative * spacing.hz12[k - 1];
    }

    /**
     * @brief Combines the midplane Nusselt sum into the final value. Port
     * of ekinem_3D.f90's `Nusselt_middle = 1.d0/AspRa + Snu`.
     * @tparam Real Floating-point type.
     * @param sum The accumulated nusseltMiddleCell() sum.
     * @param aspectRatio The configuration's aspect ratio.
     * @return The midplane Nusselt number.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real nusseltMiddleFromSum(Real sum, Real aspectRatio) {
        return Real(1) / aspectRatio + sum;
    }

    /**
     * @brief One cell's absolute value. Port of sublid_3D.f90-adjacent
     * usage: `RDP = MaxVal(Abs(Dprs))` in time_step_Q2D.f90.
     * @tparam Real Floating-point type.
     * @param i 1-based x index.
     * @param j 1-based y index.
     * @param k 1-based z index.
     * @param field The field to read.
     * @return That cell's absolute value.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real absoluteValueCell(int i, int j, int k, const FieldView<Real>& field) {
        Real value = field.at(i, j, k);
        return value < Real(0) ? -value : value;
    }

    /**
     * @brief One cell's relative (or absolute, near zero) difference
     * between two fields of identical shape. Port of sublid_3D.f90's
     * Dist2D per-cell formula.
     * @tparam Real Floating-point type.
     * @param i 1-based x index.
     * @param j 1-based y index.
     * @param k 1-based z index.
     * @param a The newer field.
     * @param b The older field, used as the relative-difference
     * denominator when its magnitude exceeds 0.1.
     * @return This cell's difference.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real relativeDifferenceCell(int i, int j, int k, const FieldView<Real>& a, const FieldView<Real>& b) {
        Real difference = a.at(i, j, k) - b.at(i, j, k);
        if (difference < Real(0)) difference = -difference;
        Real denominator = b.at(i, j, k);
        if (denominator < Real(0)) denominator = -denominator;
        if (denominator > Real(0.1)) difference /= denominator;
        return difference;
    }
};

#ifdef __CUDACC__

/**
 * @ingroup cfd_solver
 * @brief Adds one thread's value into *result via a block-level
 * shared-memory reduction followed by one atomicAdd per block.
 * @tparam Real Floating-point type.
 * @param value This thread's contribution (0 for out-of-range threads).
 * @param result Device accumulator, must be zeroed before the kernel
 * launches.
 */
template<typename Real>
__device__ __forceinline__ void blockSumInto(Real value, Real* result) {
    extern __shared__ unsigned char sharedBytes[];
    Real* shared = reinterpret_cast<Real*>(sharedBytes);
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, shared[0]);
}

/// @return The bits of a non-negative double, reinterpreted as an
/// unsigned 64-bit integer whose ordering matches the double's.
__device__ __forceinline__ unsigned long long nonNegativeDoubleBits(double value) {
    return __double_as_longlong(value);
}

/**
 * @brief Raises *result to value if value is larger, via the standard
 * atomicMax-on-reinterpreted-bits pattern (CUDA has no native atomicMax
 * for double). Valid only for non-negative values -- see this file's
 * verify_bitcast_ordering.cpp companion for the proof that non-negative
 * IEEE-754 double ordering matches unsigned 64-bit integer ordering of
 * the same bits.
 * @param value This thread's non-negative candidate maximum.
 * @param result Device accumulator (as its bit pattern), must be zeroed
 * before the kernel launches.
 */
__device__ __forceinline__ void atomicMaxNonNegativeDouble(double value, unsigned long long* result) {
    atomicMax(result, nonNegativeDoubleBits(value));
}

/// @copydoc Diagnostics::kineticEnergyCell
template<typename Real>
__global__ void kineticEnergyKernel(FieldView<Real> velocityX, FieldView<Real> velocityY, FieldView<Real> velocityZ,
                                     Spacing<Real> spacing, int nx1, int ny1, int nz1, Real* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    Real value = (i <= nx1 && j <= ny1 && k <= nz1)
        ? Diagnostics::kineticEnergyCell(i, j, k, velocityX, velocityY, velocityZ, spacing) : Real(0);
    blockSumInto(value, result);
}

/// @copydoc Diagnostics::nusseltCell
template<typename Real>
__global__ void nusseltKernel(FieldView<Real> temperature, AxisBoundaries bc, Spacing<Real> spacing, int ny1, int nz1, Real* result) {
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    Real value = (j <= ny1 && k <= nz1) ? Diagnostics::nusseltCell(j, k, temperature, bc, spacing) : Real(0);
    blockSumInto(value, result);
}

/// @copydoc Diagnostics::nusseltMiddleCell
template<typename Real>
__global__ void nusseltMiddleKernel(FieldView<Real> temperature, AxisBoundaries bc, Spacing<Real> spacing, int midplaneY, int nz1, Real* result) {
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    Real value = (k <= nz1) ? Diagnostics::nusseltMiddleCell(k, midplaneY, temperature, bc, spacing) : Real(0);
    blockSumInto(value, result);
}

/// @copydoc Diagnostics::absoluteValueCell
template<typename Real>
__global__ void maxAbsoluteValueKernel(FieldView<Real> field, unsigned long long* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > field.nx() || j > field.ny() || k > field.nz()) return;
    atomicMaxNonNegativeDouble((double)Diagnostics::absoluteValueCell(i, j, k, field), result);
}

/// @copydoc Diagnostics::relativeDifferenceCell
template<typename Real>
__global__ void maxRelativeDifferenceKernel(FieldView<Real> a, FieldView<Real> b, unsigned long long* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > a.nx() || j > a.ny() || k > a.nz()) return;
    atomicMaxNonNegativeDouble((double)Diagnostics::relativeDifferenceCell(i, j, k, a, b), result);
}

#endif //__CUDACC__

#endif //CFDSOLVER_DIAGNOSTICS_CUH
