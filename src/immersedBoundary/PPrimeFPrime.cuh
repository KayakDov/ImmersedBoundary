//
// Created by usr on 1/30/26.
//

#ifndef CUDABANDED_PPRIMEFPRIME_CUH
#define CUDABANDED_PPRIMEFPRIME_CUH

#include "ImerssedEquation.h"
#include "SparseCSC.cuh"
#include "Tensor.h"

template <typename Real, typename Int>
class PPrimeFprime {

    const GridDim& dim;
    const SimpleArray<Real> uStar, u, v, w, UGamma;
    const SparseCSC<Real, Int> R;//maps from Lagrangian Space where F lives to Eularian space where u^* lives.
    const Real3d& delta;
    const Singleton<Real> dT;

    public:

    /**
     * @brief Constructs the Pressure-Force (PPrimeFprime) system container by slicing the
     * concatenated intermediate velocity vector.
     *
     * This constructor maps the flat, monolithic intermediate velocity field @p uStar into
     * separate staggered grid views (u, v, w) and initializes the physical parameters
     * required for the Schur complement system.
     *
     * @tparam Real The floating-point precision (float or double).
     * @tparam Int  The integer type for indexing.
     *
     * @param dim     The dimensions of the scalar (pressure) grid (Nx, Ny, Nz).
     * @param uStar   The concatenated vector containing all velocity components.
     * @param R       The regularization/spreading operator data.  maps from Lagrangian Space where F lives to
     * Eularian space where u^* lives.
     * @param UGamma  The target boundary velocities (no-slip condition).
     * @param delta   The grid spacing (dx, dy, dz).
     * @param deltaT  The simulation time step size (dt).
     *
     * @note **Vector Dimension Requirements for uStar:**
     * The input @p uStar must be a contiguous array of size:
     * \f[ \text{size}(u^*) = (N_x+1)N_y N_z + N_x(N_y+1)N_z + N_x N_y(N_z+1) \f]
     * where:
     * - **u (x-component):** Uses the first \f$ (N_x+1) \times N_y \times N_z \f$ elements.
     * - **v (y-component):** Uses the next \f$ N_x \times (N_y+1) \times N_z \f$ elements.
     * - **w (z-component):** Uses the remaining \f$ N_x \times N_y \times (N_z+1) \f$ elements.
     *
     * @details The constant @p dT is computed as \f$ \frac{3}{2\Delta t} \f$, which corresponds
     * to the second-order backward differentiation (BDF2) coefficient used in the
     * SIMPLE-IB pressure-correction step.
     */
    PPrimeFprime(const GridDim &dim, const SimpleArray<Real> &uStar, SparseCSC<Real, Int> R, SimpleArray<Real> UGamma,
                 const Real3d &delta, double deltaT, ImmersedEq<Real> imEq, Real *pPrimeHost, Real *fPrimeHost,
                 bool multiStream);

    void divergence(SimpleArray<Real> result, Singleton<Real> scalar, Handle &hand);
    void setRHSFPrime(SimpleArray<Real> result, SimpleArray<Real> &sparseMultBuffer, Handle &hand);
    void setRHSPPrime(SimpleArray<Real> result, Handle &hand);
};


#endif //CUDABANDED_PPRIMEFPRIME_CUH
