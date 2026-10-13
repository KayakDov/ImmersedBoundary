/**
 * @file Operators.cuh
 * @ingroup cfd_solver
 * @brief Finite-difference operator kernels: FiniteDifferenceOperators.
 *
 * Direct port of vgradf_3D.f90, gradp_3D.f90, and divvel_3D_alloc.f90. Each
 * operator's per-cell math is a static, host+device-callable method,
 * callable identically from a host test harness (g++) or from its
 * accompanying `__global__` kernel (nvcc) -- the code tested on the host is
 * byte-for-byte the code that runs on the GPU, not a separate
 * reimplementation that could silently drift. See test_operators_host.cpp
 * for the harness that exercises these methods directly, and
 * verify_padding_equivalence.cpp for the proof that the boundary
 * substitution they use is exact.
 *
 * Every formula is a line-by-line transcription of its Fortran source;
 * variable names (A1..A7, px/py/pz) are kept identical to its origin to
 * ease cross-checking.
 */

#ifndef CFDSOLVER_OPERATORS_CUH
#define CFDSOLVER_OPERATORS_CUH

#include "../fields/BoundaryConditions.cuh"

/**
 * @ingroup cfd_solver
 * @brief Mesh spacing arrays, device-resident, built once at startup by
 * Mesh and passed by value into every kernel that needs grid spacing.
 * @tparam Real Floating-point type.
 */
template<typename Real>
struct Spacing {
    const Real* hx12; ///< Cell-to-cell spacing on x, index 0..Nx.
    const Real* hy12; ///< Cell-to-cell spacing on y, index 0..Ny.
    const Real* hz12; ///< Cell-to-cell spacing on z, index 0..Nz.
    const Real* hpx;  ///< Node-to-node spacing on x, index 0..Nx+1.
    const Real* hpy;  ///< Node-to-node spacing on y, index 0..Ny+1.
    const Real* hpz;  ///< Node-to-node spacing on z, index 0..Nz+1.
};

/**
 * @ingroup cfd_solver
 * @brief Finite-difference operators on the staggered MAC grid.
 */
class FiniteDifferenceOperators {
public:
    /**
     * @brief Temperature advection term, one cell. Port of vgradf_3D.f90's
     * VgrTmp.
     *
     * The buoyancy reference field (Teta) is passed as a direct,
     * 0-indexed lookup table over x alone, not as a FieldView read through
     * readTemperature(): Teta is set once, directly, as `1 -
     * X12(i)/aspectRatio` across its ENTIRE x extent including both
     * boundary positions (ConvMain_3D_Q2D.f90's init loop runs i=0..Nx2,
     * and EVDbounds never touches Teta at all) -- it does not share
     * Temperature's Dirichlet/Neumann boundary treatment, and resolving
     * it that way would give the wrong value at the x boundaries whenever
     * that does not coincidentally match the linear profile's true
     * extension. Teta does not vary with y or z, so no such table is
     * needed on those axes.
     *
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx1].
     * @param j 1-based y index in [1, Ny1].
     * @param k 1-based z index in [1, Nz1].
     * @param temperature Current temperature field.
     * @param buoyancyByX Teta's value at every x position, 0-indexed,
     * covering index 0 through nx1+1 inclusive (size nx1+2).
     * @param velocityX Current VMx.
     * @param velocityY Current VMy.
     * @param velocityZ Current VMz.
     * @param spacing Mesh spacing.
     * @param bc Temperature's boundary condition.
     * @return The advection term's contribution to this cell's RHS.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real temperatureAdvection(int i, int j, int k,
                                      const FieldView<Real>& temperature, const Real* buoyancyByX,
                                      const FieldView<Real>& velocityX, const FieldView<Real>& velocityY, const FieldView<Real>& velocityZ,
                                      const Spacing<Real>& spacing, AxisBoundaries bc) {
        Real px = Real(0.5) / spacing.hx12[i-1];
        Real py = Real(0.5) / spacing.hy12[j-1];
        Real pz = Real(0.5) / spacing.hz12[k-1];

        Real a1 = -BoundaryConditions::readVelocity(velocityY, i, j-1, k) * py;
        Real a4 =  BoundaryConditions::readVelocity(velocityY, i, j,   k) * py;
        Real a2 = -BoundaryConditions::readVelocity(velocityX, i-1, j, k) * px;
        Real a5 =  BoundaryConditions::readVelocity(velocityX, i,   j, k) * px;
        Real a6 = -BoundaryConditions::readVelocity(velocityZ, i, j, k-1) * pz;
        Real a7 =  BoundaryConditions::readVelocity(velocityZ, i, j, k)   * pz;
        Real a3 = -(a1 + a2 + a4 + a5 + a6 + a7);

        return a1 * total(temperature, buoyancyByX, i, j-1, k, bc)
             + a2 * total(temperature, buoyancyByX, i-1, j, k, bc)
             + a3 * total(temperature, buoyancyByX, i, j, k, bc)
             + a4 * total(temperature, buoyancyByX, i, j+1, k, bc)
             + a5 * total(temperature, buoyancyByX, i+1, j, k, bc)
             + a6 * total(temperature, buoyancyByX, i, j, k-1, bc)
             + a7 * total(temperature, buoyancyByX, i, j, k+1, bc);
    }

    /**
     * @brief x-velocity self-advection, one cell. Port of vgradf_3D.f90's
     * VgrdVx.
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx].
     * @param j 1-based y index in [1, Ny1].
     * @param k 1-based z index in [1, Nz1].
     * @param velocityX Current VMx.
     * @param velocityY Current VMy.
     * @param velocityZ Current VMz.
     * @param spacing Mesh spacing.
     * @return The self-advection term's contribution to this cell's RHS.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real velocityXAdvection(int i, int j, int k,
                                    const FieldView<Real>& velocityX, const FieldView<Real>& velocityY, const FieldView<Real>& velocityZ,
                                    const Spacing<Real>& spacing) {
        Real px = Real(0.25) / spacing.hpx[i];
        Real py = px / spacing.hy12[j-1], py1 = spacing.hx12[i-1]*py, py2 = spacing.hx12[i]*py;
        Real pz = px / spacing.hz12[k-1], pz1 = spacing.hx12[i-1]*pz, pz2 = spacing.hx12[i]*pz;

        Real a1 = -py2*BoundaryConditions::readVelocity(velocityY,i+1,j-1,k) - py1*BoundaryConditions::readVelocity(velocityY,i,j-1,k);
        Real a4 =  py2*BoundaryConditions::readVelocity(velocityY,i+1,j,k)   + py1*BoundaryConditions::readVelocity(velocityY,i,j,k);
        Real a6 = -pz2*BoundaryConditions::readVelocity(velocityZ,i+1,j,k-1) - pz1*BoundaryConditions::readVelocity(velocityZ,i,j,k-1);
        Real a7 =  pz2*BoundaryConditions::readVelocity(velocityZ,i+1,j,k)   + pz1*BoundaryConditions::readVelocity(velocityZ,i,j,k);
        Real a2 = -BoundaryConditions::readVelocity(velocityX,i-1,j,k)*px;
        Real a5 =  BoundaryConditions::readVelocity(velocityX,i+1,j,k)*px;
        Real a3 = -(a1+a4+a6+a7);

        return a1*BoundaryConditions::readVelocity(velocityX,i,j-1,k) + a2*BoundaryConditions::readVelocity(velocityX,i-1,j,k)
             + a3*BoundaryConditions::readVelocity(velocityX,i,j,k)
             + a4*BoundaryConditions::readVelocity(velocityX,i,j+1,k) + a5*BoundaryConditions::readVelocity(velocityX,i+1,j,k)
             + a6*BoundaryConditions::readVelocity(velocityX,i,j,k-1) + a7*BoundaryConditions::readVelocity(velocityX,i,j,k+1);
    }

    /**
     * @brief y-velocity self-advection, one cell. Port of vgradf_3D.f90's
     * VgrdVy.
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx1].
     * @param j 1-based y index in [1, Ny].
     * @param k 1-based z index in [1, Nz1].
     * @param velocityX Current VMx.
     * @param velocityY Current VMy.
     * @param velocityZ Current VMz.
     * @param spacing Mesh spacing.
     * @return The self-advection term's contribution to this cell's RHS.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real velocityYAdvection(int i, int j, int k,
                                    const FieldView<Real>& velocityX, const FieldView<Real>& velocityY, const FieldView<Real>& velocityZ,
                                    const Spacing<Real>& spacing) {
        Real py = Real(0.25) / spacing.hpy[j];
        Real px = py / spacing.hx12[i-1], px1 = spacing.hy12[j-1]*px, px2 = spacing.hy12[j]*px;
        Real pz = py / spacing.hz12[k-1], pz1 = spacing.hy12[j-1]*pz, pz2 = spacing.hy12[j]*pz;

        Real a1 = -BoundaryConditions::readVelocity(velocityY,i,j-1,k)*py;
        Real a4 =  BoundaryConditions::readVelocity(velocityY,i,j+1,k)*py;
        Real a2 = -px2*BoundaryConditions::readVelocity(velocityX,i-1,j+1,k) - px1*BoundaryConditions::readVelocity(velocityX,i-1,j,k);
        Real a5 =  px2*BoundaryConditions::readVelocity(velocityX,i,j+1,k)   + px1*BoundaryConditions::readVelocity(velocityX,i,j,k);
        Real a6 = -pz2*BoundaryConditions::readVelocity(velocityZ,i,j+1,k-1) - pz1*BoundaryConditions::readVelocity(velocityZ,i,j,k-1);
        Real a7 =  pz2*BoundaryConditions::readVelocity(velocityZ,i,j+1,k)   + pz1*BoundaryConditions::readVelocity(velocityZ,i,j,k);
        Real a3 = -(a2+a5+a6+a7);

        return a1*BoundaryConditions::readVelocity(velocityY,i,j-1,k) + a2*BoundaryConditions::readVelocity(velocityY,i-1,j,k)
             + a3*BoundaryConditions::readVelocity(velocityY,i,j,k)
             + a4*BoundaryConditions::readVelocity(velocityY,i,j+1,k) + a5*BoundaryConditions::readVelocity(velocityY,i+1,j,k)
             + a6*BoundaryConditions::readVelocity(velocityY,i,j,k-1) + a7*BoundaryConditions::readVelocity(velocityY,i,j,k+1);
    }

    /**
     * @brief z-velocity self-advection, one cell. Port of vgradf_3D.f90's
     * VgrdVz.
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx1].
     * @param j 1-based y index in [1, Ny1].
     * @param k 1-based z index in [1, Nz].
     * @param velocityX Current VMx.
     * @param velocityY Current VMy.
     * @param velocityZ Current VMz.
     * @param spacing Mesh spacing.
     * @return The self-advection term's contribution to this cell's RHS.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real velocityZAdvection(int i, int j, int k,
                                    const FieldView<Real>& velocityX, const FieldView<Real>& velocityY, const FieldView<Real>& velocityZ,
                                    const Spacing<Real>& spacing) {
        Real pz = Real(0.25) / spacing.hpz[k];
        Real px = pz / spacing.hx12[i-1], px1 = spacing.hz12[k-1]*px, px2 = spacing.hz12[k]*px;
        Real py = pz / spacing.hy12[j-1], py1 = spacing.hz12[k-1]*py, py2 = spacing.hz12[k]*py;

        Real a6 = -BoundaryConditions::readVelocity(velocityZ,i,j,k-1)*pz;
        Real a7 =  BoundaryConditions::readVelocity(velocityZ,i,j,k+1)*pz;
        Real a2 = -px2*BoundaryConditions::readVelocity(velocityX,i-1,j,k+1) - px1*BoundaryConditions::readVelocity(velocityX,i-1,j,k);
        Real a5 =  px2*BoundaryConditions::readVelocity(velocityX,i,j,k+1)   + px1*BoundaryConditions::readVelocity(velocityX,i,j,k);
        Real a1 = -py2*BoundaryConditions::readVelocity(velocityY,i,j-1,k+1) - py1*BoundaryConditions::readVelocity(velocityY,i,j-1,k);
        Real a4 =  py2*BoundaryConditions::readVelocity(velocityY,i,j,k+1)   + py1*BoundaryConditions::readVelocity(velocityY,i,j,k);
        Real a3 = -(a2+a5+a1+a4);

        return a1*BoundaryConditions::readVelocity(velocityZ,i,j-1,k) + a2*BoundaryConditions::readVelocity(velocityZ,i-1,j,k)
             + a3*BoundaryConditions::readVelocity(velocityZ,i,j,k)
             + a4*BoundaryConditions::readVelocity(velocityZ,i,j+1,k) + a5*BoundaryConditions::readVelocity(velocityZ,i+1,j,k)
             + a6*BoundaryConditions::readVelocity(velocityZ,i,j,k-1) + a7*BoundaryConditions::readVelocity(velocityZ,i,j,k+1);
    }

    /**
     * @brief The buoyancy force's contribution to one cell of RHSz. Port
     * of time_step_Q2D.f90's `RHSz = RHSz - 0.5*Bu_Gr*(TmpNew(i,j,k)+
     * TmpNew(i,j,k+1)) - 0.5*Bu_Gr*(Teta(i,j,k)+Teta(i,j,k+1))`.
     *
     * Teta's two terms simplify to `2*buoyancyByX[i]` (Teta does not vary
     * with j or k -- see temperatureAdvection's comment), folded into the
     * single buGr*buoyancyByX[i] term below. k+1 is always within
     * newTemperature's real z range for every (i,j,k) this is called at
     * (Vz's own k loop tops out at Nz, one less than Temperature's own
     * Nz1), so this needs no boundary-conditional read.
     *
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx1].
     * @param j 1-based y index in [1, Ny1].
     * @param k 1-based z index in [1, Nz].
     * @param rhsZ RHSz, mutated in place.
     * @param newTemperature The just-solved Temperature field (TmpNew).
     * @param buoyancyByX Teta's x-indexed lookup table.
     * @param buGr Bu_Gr = Gr * DGr^2.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static void buoyancyForceZ(int i, int j, int k, FieldView<Real>& rhsZ, const FieldView<Real>& newTemperature,
                                const Real* buoyancyByX, Real buGr) {
        Real temperatureSum = newTemperature.at(i, j, k) + newTemperature.at(i, j, k + 1);
        rhsZ.at(i, j, k) -= Real(0.5) * buGr * temperatureSum + buGr * buoyancyByX[i];
    }

    /**
     * @brief Pressure gradient, x component, one cell. Port of
     * gradp_3D.f90's GradPx.
     *
     * Prs/Dprs were never ghost-padded in the original Fortran (allocated
     * exactly to their real extent), so this needs no boundary
     * substitution: every read is already within range by construction.
     *
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx].
     * @param j 1-based y index in [1, Ny1].
     * @param k 1-based z index in [1, Nz1].
     * @param pressure The pressure (or pressure-correction) field.
     * @param spacing Mesh spacing.
     * @return The x component of the pressure gradient at this cell.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real pressureGradientX(int i, int j, int k, const FieldView<Real>& pressure, const Spacing<Real>& spacing) {
        return (pressure.at(i+1,j,k) - pressure.at(i,j,k)) / spacing.hpx[i];
    }

    /**
     * @brief Pressure gradient, y component, one cell. Port of
     * gradp_3D.f90's GradPy.
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx1].
     * @param j 1-based y index in [1, Ny].
     * @param k 1-based z index in [1, Nz1].
     * @param pressure The pressure (or pressure-correction) field.
     * @param spacing Mesh spacing.
     * @return The y component of the pressure gradient at this cell.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real pressureGradientY(int i, int j, int k, const FieldView<Real>& pressure, const Spacing<Real>& spacing) {
        return (pressure.at(i,j+1,k) - pressure.at(i,j,k)) / spacing.hpy[j];
    }

    /**
     * @brief Pressure gradient, z component, one cell. Port of
     * gradp_3D.f90's GradPz.
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx1].
     * @param j 1-based y index in [1, Ny1].
     * @param k 1-based z index in [1, Nz].
     * @param pressure The pressure (or pressure-correction) field.
     * @param spacing Mesh spacing.
     * @return The z component of the pressure gradient at this cell.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real pressureGradientZ(int i, int j, int k, const FieldView<Real>& pressure, const Spacing<Real>& spacing) {
        return (pressure.at(i,j,k+1) - pressure.at(i,j,k)) / spacing.hpz[k];
    }

    /**
     * @brief Velocity divergence, one cell. Port of
     * divvel_3D_alloc.f90's FDdiv.
     * @tparam Real Floating-point type.
     * @param i 1-based x index in [1, Nx1].
     * @param j 1-based y index in [1, Ny1].
     * @param k 1-based z index in [1, Nz1].
     * @param velocityXNew Updated VMxNew.
     * @param velocityYNew Updated VMyNew.
     * @param velocityZNew Updated VMzNew.
     * @param spacing Mesh spacing.
     * @return The divergence at this cell.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real divergence(int i, int j, int k,
                            const FieldView<Real>& velocityXNew, const FieldView<Real>& velocityYNew, const FieldView<Real>& velocityZNew,
                            const Spacing<Real>& spacing) {
        return (BoundaryConditions::readVelocity(velocityXNew,i,j,k) - BoundaryConditions::readVelocity(velocityXNew,i-1,j,k)) / spacing.hx12[i-1]
             + (BoundaryConditions::readVelocity(velocityYNew,i,j,k) - BoundaryConditions::readVelocity(velocityYNew,i,j-1,k)) / spacing.hy12[j-1]
             + (BoundaryConditions::readVelocity(velocityZNew,i,j,k) - BoundaryConditions::readVelocity(velocityZNew,i,j,k-1)) / spacing.hz12[k-1];
    }

private:
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real total(const FieldView<Real>& temperature, const Real* buoyancyByX, int i, int j, int k, AxisBoundaries bc) {
        return BoundaryConditions::readTemperature(temperature, i, j, k, bc) + buoyancyByX[i];
    }
};

#ifdef __CUDACC__

/**
 * @ingroup cfd_solver
 * @brief Launches FiniteDifferenceOperators::temperatureAdvection over the
 * whole grid, accumulating into rhs.
 * @tparam Real Floating-point type.
 */
template<typename Real>
__global__ void temperatureAdvectionKernel(FieldView<Real> rhs, FieldView<Real> temperature, const Real* buoyancyByX,
                                            FieldView<Real> velocityX, FieldView<Real> velocityY, FieldView<Real> velocityZ,
                                            Spacing<Real> spacing, AxisBoundaries bc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > rhs.nx() || j > rhs.ny() || k > rhs.nz()) return;
    rhs.at(i,j,k) += FiniteDifferenceOperators::temperatureAdvection(i, j, k, temperature, buoyancyByX, velocityX, velocityY, velocityZ, spacing, bc);
}

/// @copydoc FiniteDifferenceOperators::buoyancyForceZ
template<typename Real>
__global__ void buoyancyForceZKernel(FieldView<Real> rhsZ, FieldView<Real> newTemperature, const Real* buoyancyByX, Real buGr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > rhsZ.nx() || j > rhsZ.ny() || k > rhsZ.nz()) return;
    FiniteDifferenceOperators::buoyancyForceZ(i, j, k, rhsZ, newTemperature, buoyancyByX, buGr);
}

/// @copydoc temperatureAdvectionKernel
template<typename Real>
__global__ void velocityXAdvectionKernel(FieldView<Real> rhs, FieldView<Real> velocityX, FieldView<Real> velocityY, FieldView<Real> velocityZ, Spacing<Real> spacing) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > velocityX.nx() || j > velocityX.ny() || k > velocityX.nz()) return;
    rhs.at(i,j,k) += FiniteDifferenceOperators::velocityXAdvection(i, j, k, velocityX, velocityY, velocityZ, spacing);
}

/// @copydoc temperatureAdvectionKernel
template<typename Real>
__global__ void velocityYAdvectionKernel(FieldView<Real> rhs, FieldView<Real> velocityX, FieldView<Real> velocityY, FieldView<Real> velocityZ, Spacing<Real> spacing) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > velocityY.nx() || j > velocityY.ny() || k > velocityY.nz()) return;
    rhs.at(i,j,k) += FiniteDifferenceOperators::velocityYAdvection(i, j, k, velocityX, velocityY, velocityZ, spacing);
}

/// @copydoc temperatureAdvectionKernel
template<typename Real>
__global__ void velocityZAdvectionKernel(FieldView<Real> rhs, FieldView<Real> velocityX, FieldView<Real> velocityY, FieldView<Real> velocityZ, Spacing<Real> spacing) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > velocityZ.nx() || j > velocityZ.ny() || k > velocityZ.nz()) return;
    rhs.at(i,j,k) += FiniteDifferenceOperators::velocityZAdvection(i, j, k, velocityX, velocityY, velocityZ, spacing);
}

/// @copydoc temperatureAdvectionKernel
template<typename Real>
__global__ void pressureGradientXKernel(FieldView<Real> result, FieldView<Real> pressure, Spacing<Real> spacing) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > result.nx() || j > result.ny() || k > result.nz()) return;
    result.at(i,j,k) = FiniteDifferenceOperators::pressureGradientX(i, j, k, pressure, spacing);
}

/// @copydoc temperatureAdvectionKernel
template<typename Real>
__global__ void pressureGradientYKernel(FieldView<Real> result, FieldView<Real> pressure, Spacing<Real> spacing) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > result.nx() || j > result.ny() || k > result.nz()) return;
    result.at(i,j,k) = FiniteDifferenceOperators::pressureGradientY(i, j, k, pressure, spacing);
}

/// @copydoc temperatureAdvectionKernel
template<typename Real>
__global__ void pressureGradientZKernel(FieldView<Real> result, FieldView<Real> pressure, Spacing<Real> spacing) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > result.nx() || j > result.ny() || k > result.nz()) return;
    result.at(i,j,k) = FiniteDifferenceOperators::pressureGradientZ(i, j, k, pressure, spacing);
}

/// @copydoc temperatureAdvectionKernel
template<typename Real>
__global__ void divergenceKernel(FieldView<Real> rhs, FieldView<Real> velocityXNew, FieldView<Real> velocityYNew, FieldView<Real> velocityZNew, Spacing<Real> spacing) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > rhs.nx() || j > rhs.ny() || k > rhs.nz()) return;
    rhs.at(i,j,k) = FiniteDifferenceOperators::divergence(i, j, k, velocityXNew, velocityYNew, velocityZNew, spacing);
}

#endif //__CUDACC__

#endif //CFDSOLVER_OPERATORS_CUH
