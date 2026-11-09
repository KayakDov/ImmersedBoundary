/**
 * @file ElementwiseOps.cuh
 * @ingroup cfd_solver
 * @brief ElementwiseOps: small, generic per-cell operations shared across
 * every field's right-hand-side build.
 */

#ifndef CFDSOLVER_ELEMENTWISEOPS_CUH
#define CFDSOLVER_ELEMENTWISEOPS_CUH

#include "../fields/BoundaryConditions.cuh"
#include "ThreadIndex.cuh"

/**
 * @ingroup cfd_solver
 * @brief Generic per-cell operations used by several fields' right-hand-
 * side builds, factored out once rather than duplicated per field.
 */
class ElementwiseOps {
public:
    /**
     * @brief Subtracts a three-level time derivative from one cell of a
     * right-hand side: `rhs -= coefficient * (4*current - old)`. Port of
     * the `RHSx = RHSx - (4*VMx - VMxOld)/Ht` pattern shared by
     * Temperature and all three velocity components in time_step_Q2D.f90
     * (coefficient absorbs any extra scalar, such as Temperature's
     * dt_temp/Ht vs. velocity's plain 1/Ht).
     * @tparam Real Floating-point type.
     * @param i 1-based x index.
     * @param j 1-based y index.
     * @param k 1-based z index.
     * @param rhs The right-hand side, mutated in place.
     * @param current This field's value at the start of the current step.
     * @param old This field's value one step back.
     * @param coefficient The scalar multiplying the derivative term.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static void subtractTimeDerivative(int i, int j, int k, FieldView<Real>& rhs,
                                        const FieldView<Real>& current, const FieldView<Real>& old, Real coefficient) {
        rhs.at(i, j, k) -= coefficient * (Real(4) * current.at(i, j, k) - old.at(i, j, k));
    }

    /**
     * @brief Multiplies one cell of a field by a scalar. Port of the
     * `GPU_RHS_T = FDRHP * GrPr` / `GPU_RHS_Vx = RHSx / DGr` scaling
     * pattern shared by every field's right-hand side before its solve.
     * @tparam Real Floating-point type.
     * @param i 1-based x index.
     * @param j 1-based y index.
     * @param k 1-based z index.
     * @param field The field, mutated in place.
     * @param factor The scalar to multiply by.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static void scale(int i, int j, int k, FieldView<Real>& field, Real factor) {
        field.at(i, j, k) *= factor;
    }

    /**
     * @brief Adds one field's value into another, in place:
     * `target += source`. Port of the `Prs = Prs + Dprs` pattern.
     * @tparam Real Floating-point type.
     * @param i 1-based x index.
     * @param j 1-based y index.
     * @param k 1-based z index.
     * @param target The field being added into, mutated in place.
     * @param source The field being added.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static void addInto(int i, int j, int k, FieldView<Real>& target, const FieldView<Real>& source) {
        target.at(i, j, k) += source.at(i, j, k);
    }

    /**
     * @brief Corrects one cell of a velocity component by its pressure
     * gradient: `velocity -= gradient * coefficient`. Port of the
     * `VMxNew = VMxNew - RHSx * Htime / Ckor` pattern.
     * @tparam Real Floating-point type.
     * @param i 1-based x index.
     * @param j 1-based y index.
     * @param k 1-based z index.
     * @param velocity The velocity component, mutated in place.
     * @param gradient The pressure-correction gradient at this cell.
     * @param coefficient Htime / Ckor.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static void correctVelocity(int i, int j, int k, FieldView<Real>& velocity, const FieldView<Real>& gradient, Real coefficient) {
        velocity.at(i, j, k) -= gradient.at(i, j, k) * coefficient;
    }

    /**
     * @brief Accumulates a scaled copy of one field into another, in place:
     * `target += source * factor`. Port of the running time-integral
     * pattern shared by every term of Solution_time_Q2D.f90's Fourier_sum
     * (`VMx_Av = VMx_Av + VMxNew * Htime`, and, for i=1..N_Fourier,
     * `Tmp_Amplitude(i,:,:,:) = Tmp_Amplitude(i,:,:,:) + TmpNew(:,:,:) *
     * Htime * cos(Omega(i)*TimCur)`) -- the per-cell operation is identical
     * either way, only `factor` differs (plain `Htime` for the running
     * mean, `Htime*cos(...)`/`Htime*sin(...)`, computed once on the host
     * per call, for a Fourier term). See
     * CfdSimulation::accumulateTimeAverage().
     * @tparam Real Floating-point type.
     * @param i 1-based x index.
     * @param j 1-based y index.
     * @param k 1-based z index.
     * @param target The running accumulator, mutated in place.
     * @param source The field being accumulated (this step's value).
     * @param factor The scalar multiplying source before it's added.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static void accumulateScaled(int i, int j, int k, FieldView<Real>& target, const FieldView<Real>& source, Real factor) {
        target.at(i, j, k) += source.at(i, j, k) * factor;
    }
};

#ifdef __CUDACC__

/// @copydoc ElementwiseOps::subtractTimeDerivative
template<typename Real>
__global__ void subtractTimeDerivativeKernel(FieldView<Real> rhs, FieldView<Real> current, FieldView<Real> old, Real coefficient) {
    XYZ<int> idx = threadGridIndex();
    int i = idx.x, j = idx.y, k = idx.z;
    if (i > rhs.nx() || j > rhs.ny() || k > rhs.nz()) return;
    ElementwiseOps::subtractTimeDerivative(i, j, k, rhs, current, old, coefficient);
}

/// @copydoc ElementwiseOps::scale
template<typename Real>
__global__ void scaleKernel(FieldView<Real> field, Real factor) {
    XYZ<int> idx = threadGridIndex();
    int i = idx.x, j = idx.y, k = idx.z;
    if (i > field.nx() || j > field.ny() || k > field.nz()) return;
    ElementwiseOps::scale(i, j, k, field, factor);
}

/// @copydoc ElementwiseOps::addInto
template<typename Real>
__global__ void addIntoKernel(FieldView<Real> target, FieldView<Real> source) {
    XYZ<int> idx = threadGridIndex();
    int i = idx.x, j = idx.y, k = idx.z;
    if (i > target.nx() || j > target.ny() || k > target.nz()) return;
    ElementwiseOps::addInto(i, j, k, target, source);
}

/// @copydoc ElementwiseOps::correctVelocity
template<typename Real>
__global__ void correctVelocityKernel(FieldView<Real> velocity, FieldView<Real> gradient, Real coefficient) {
    XYZ<int> idx = threadGridIndex();
    int i = idx.x, j = idx.y, k = idx.z;
    if (i > velocity.nx() || j > velocity.ny() || k > velocity.nz()) return;
    ElementwiseOps::correctVelocity(i, j, k, velocity, gradient, coefficient);
}

/// @copydoc ElementwiseOps::accumulateScaled
template<typename Real>
__global__ void accumulateScaledKernel(FieldView<Real> target, FieldView<Real> source, Real factor) {
    XYZ<int> idx = threadGridIndex();
    int i = idx.x, j = idx.y, k = idx.z;
    if (i > target.nx() || j > target.ny() || k > target.nz()) return;
    ElementwiseOps::accumulateScaled(i, j, k, target, source, factor);
}

#endif //__CUDACC__

#endif //CFDSOLVER_ELEMENTWISEOPS_CUH
