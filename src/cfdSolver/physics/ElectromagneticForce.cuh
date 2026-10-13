/**
 * @file ElectromagneticForce.cuh
 * @ingroup cfd_solver
 * @brief ElectromagneticForce: the electric-potential solve and the
 * resulting Lorentz-force contribution to the velocity right-hand sides.
 *
 * Direct port of EM_forcing.f90 (Get_Potential_Launch, Get_Potential_
 * Finish, EM_force).
 */

#ifndef CFDSOLVER_ELECTROMAGNETICFORCE_CUH
#define CFDSOLVER_ELECTROMAGNETICFORCE_CUH

#include "Operators.cuh"

/**
 * @ingroup cfd_solver
 * @brief The electric-potential curl right-hand side, Lorentz force, and
 * the mean-subtraction needed by the pure-Poisson potential solve.
 */
class ElectromagneticForce {
public:
    /**
     * @brief The potential equation's right-hand side at one cell: the curl
     * of the previous-timestep velocity field. Port of Get_Potential_
     * Launch's inner loop.
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx].
     * @param j 1-based y index in [1, Ny1].
     * @param k 1-based z index in [1, Nz].
     * @param velocityX Current VMx.
     * @param velocityZ Current VMz.
     * @param spacing Mesh spacing.
     * @return This cell's right-hand-side value.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real potentialRightHandSide(int i, int j, int k, const FieldView<Real>& velocityX, const FieldView<Real>& velocityZ, const Spacing<Real>& spacing) {
        Real dVxDz = (BoundaryConditions::readVelocity(velocityX, i, j, k + 1) - BoundaryConditions::readVelocity(velocityX, i, j, k)) / spacing.hpz[k];
        Real dVzDx = (BoundaryConditions::readVelocity(velocityZ, i + 1, j, k) - BoundaryConditions::readVelocity(velocityZ, i, j, k)) / spacing.hpx[i];
        return dVzDx - dVxDz;
    }

    /**
     * @brief Subtracts a fixed reference value from every cell, fixing
     * the pure-Poisson solve's gauge freedom. Port of Get_Potential_
     * Finish's `Potential = Potential - Potential(1,1,1)` step.
     *
     * The reference value is supplied as a separate pointer rather than
     * read from `potential.at(1,1,1)` directly: every thread in the
     * launched kernel runs this concurrently, and potential(1,1,1) is
     * itself one of the cells being mutated, so reading it from the same
     * buffer every thread is writing to is a race -- some threads could
     * see it before it is zeroed, others after. The caller (CfdSimulation)
     * copies potential(1,1,1) into its own single-element scratch buffer,
     * device-to-device, before launching the kernel that calls this --
     * see CfdSimulation.cu's finishPotentialSolve().
     *
     * @tparam Real Floating-point type.
     * @param i 1-based x index.
     * @param j 1-based y index.
     * @param k 1-based z index.
     * @param potential The solved potential field, mutated in place.
     * @param referenceValue Pointer to potential's pre-copied (1,1,1)
     * value, in a buffer distinct from `potential` itself.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static void subtractReferenceValue(int i, int j, int k, FieldView<Real>& potential, const Real* referenceValue) {
        potential.at(i, j, k) -= *referenceValue;
    }

    /**
     * @brief The Lorentz force's x contribution at one cell, added to
     * RHSx. Port of EM_force's first loop.
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx].
     * @param j 1-based y index in [1, Ny1].
     * @param k 1-based z index in [1, Nz1].
     * @param rhsX RHSx, mutated in place.
     * @param velocityX Current VMx.
     * @param potential The solved, mean-subtracted potential field.
     * @param potentialBc Potential's per-axis boundary condition.
     * @param spacing Mesh spacing.
     * @param coefficient DGr * (Hartmann * widthRatio)^2.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static void applyForceX(int i, int j, int k, FieldView<Real>& rhsX, const FieldView<Real>& velocityX,
                             const FieldView<Real>& potential, AxisBoundaries potentialBc,
                             const Spacing<Real>& spacing, Real coefficient) {
        Real dPotentialDz = (BoundaryConditions::readTemperature(potential, i, j, k, potentialBc)
                            - BoundaryConditions::readTemperature(potential, i, j, k - 1, potentialBc)) / spacing.hz12[k - 1];
        rhsX.at(i, j, k) += coefficient * (dPotentialDz + BoundaryConditions::readVelocity(velocityX, i, j, k));
    }

    /**
     * @brief The Lorentz force's z contribution at one cell, subtracted
     * from RHSz. Port of EM_force's second loop.
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx1].
     * @param j 1-based y index in [1, Ny1].
     * @param k 1-based z index in [1, Nz].
     * @param rhsZ RHSz, mutated in place.
     * @param velocityZ Current VMz.
     * @param potential The solved, mean-subtracted potential field.
     * @param potentialBc Potential's per-axis boundary condition.
     * @param spacing Mesh spacing.
     * @param coefficient DGr * (Hartmann * widthRatio)^2.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static void applyForceZ(int i, int j, int k, FieldView<Real>& rhsZ, const FieldView<Real>& velocityZ,
                             const FieldView<Real>& potential, AxisBoundaries potentialBc,
                             const Spacing<Real>& spacing, Real coefficient) {
        Real dPotentialDx = (BoundaryConditions::readTemperature(potential, i, j, k, potentialBc)
                            - BoundaryConditions::readTemperature(potential, i - 1, j, k, potentialBc)) / spacing.hx12[i - 1];
        rhsZ.at(i, j, k) -= coefficient * (dPotentialDx - BoundaryConditions::readVelocity(velocityZ, i, j, k));
    }
};

#ifdef __CUDACC__

/// @copydoc ElectromagneticForce::potentialRightHandSide
template<typename Real>
__global__ void potentialRightHandSideKernel(FieldView<Real> rhs, FieldView<Real> velocityX, FieldView<Real> velocityZ, Spacing<Real> spacing) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > rhs.nx() || j > rhs.ny() || k > rhs.nz()) return;
    rhs.at(i, j, k) = ElectromagneticForce::potentialRightHandSide(i, j, k, velocityX, velocityZ, spacing);
}

/// @copydoc ElectromagneticForce::subtractReferenceValue
template<typename Real>
__global__ void subtractReferenceValueKernel(FieldView<Real> potential, const Real* referenceValue) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > potential.nx() || j > potential.ny() || k > potential.nz()) return;
    ElectromagneticForce::subtractReferenceValue(i, j, k, potential, referenceValue);
}

/// @copydoc ElectromagneticForce::applyForceX
template<typename Real>
__global__ void applyForceXKernel(FieldView<Real> rhsX, FieldView<Real> velocityX, FieldView<Real> potential,
                                   AxisBoundaries potentialBc, Spacing<Real> spacing, Real coefficient) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > rhsX.nx() || j > rhsX.ny() || k > rhsX.nz()) return;
    ElectromagneticForce::applyForceX(i, j, k, rhsX, velocityX, potential, potentialBc, spacing, coefficient);
}

/// @copydoc ElectromagneticForce::applyForceZ
template<typename Real>
__global__ void applyForceZKernel(FieldView<Real> rhsZ, FieldView<Real> velocityZ, FieldView<Real> potential,
                                   AxisBoundaries potentialBc, Spacing<Real> spacing, Real coefficient) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > rhsZ.nx() || j > rhsZ.ny() || k > rhsZ.nz()) return;
    ElectromagneticForce::applyForceZ(i, j, k, rhsZ, velocityZ, potential, potentialBc, spacing, coefficient);
}

#endif //__CUDACC__

#endif //CFDSOLVER_ELECTROMAGNETICFORCE_CUH
