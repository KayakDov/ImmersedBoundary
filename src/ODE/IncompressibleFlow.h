/**
 * @file IncompressibleFlow.h
 * @brief ODE child class that keeps its state on the divergence-free manifold using a Fast Poisson Solver.
 */

#ifndef INCOMPRESSIBLE_FLOW_H
#define INCOMPRESSIBLE_FLOW_H

#include "ODE.h"
#include "poisson/EigenDecompSolver.h"

/**
 * @brief An ODE whose state is a velocity field constrained to be
 *        discretely divergence-free (incompressible flow), enforced with
 *        a Fast Poisson Solver (EigenDecompSolver).
 *
 * A derived class still supplies dxdt() -- the momentum-equation
 * right-hand side (advection, diffusion, forcing), exactly as for any
 * other ODE subclass, without needing to account for pressure: this
 * class adds the incompressibility correction on top, automatically, in
 * two places:
 *
 *   1. nextPoint() -- after ODE's ordinary RK4 step advances the state
 *      using only the momentum terms from dxdt(), the result is passed
 *      through the same projection used below to remove whatever
 *      component would make it non-divergence-free. This is the
 *      classical fractional-step / projection method for incompressible
 *      Navier-Stokes: an explicit substep followed by a single
 *      pressure-projection correction per full timestep, rather than
 *      per RK stage.
 *
 *   2. TimeSequence::projection() -- the identical projection, reused
 *      unmodified, applied to a Lyapunov-exponent deviation vector
 *      instead of to the flow state. Because div = 0 is a LINEAR
 *      constraint, and nextPoint() already guarantees its output is
 *      exactly divergence-free for any input, the difference of two
 *      divergence-free states is automatically divergence-free too --
 *      so this override is drift insurance more than a strict per-step
 *      necessity, but it is nearly free to keep.
 *
 * TimeSequence::norm() is intentionally left at its default (the plain
 * Euclidean norm): once a vector has been projected, there is no
 * remaining spurious direction to exclude, so the default norm is
 * already the correct measure of physical size here.
 *
 * dxdt() remains pure virtual, inherited unimplemented from ODE<Real>: a
 * further-derived class supplies the actual momentum equation. This
 * class only ever needs to be instantiated through that further
 * derivation, the same way Lorenz<Real> derives from ODE<Real> to supply
 * a concrete dxdt() for a simpler system.
 *
 * @tparam Real Floating-point type.
 */
template<typename Real>
class IncompressibleFlow : public ODE<Real> {
public:
    /**
     * @param numDims        Total length of the flattened velocity-field
     *                        state vector (as passed to ODE's constructor).
     * @param stepSizeScalar RK4 timestep.
     * @param hand           GPU execution handle used for setup and
     *                        stored by ODE -- see TimeSequence::handle().
     * @param poissonSolver  Solver enforcing div = 0. Must be built on
     *                        the same grid as the flow, with
     *                        allNeumann = true (a singular Neumann
     *                        pressure-Poisson solve).
     * @param rhsScratch     Scratch scalar field, sized to the
     *                        pressure/phi grid, holding div(field)
     *                        during projection.
     * @param phiScratch     Scratch scalar field, same size, holding the
     *                        solved potential phi during projection.
     */
    IncompressibleFlow(
        size_t numDims,
        Real stepSizeScalar,
        Handle& hand,
        EigenDecompSolver<Real>& poissonSolver,
        SimpleArray<Real> rhsScratch,
        SimpleArray<Real> phiScratch
    ) : ODE<Real>(numDims, stepSizeScalar, hand),
        poissonSolver_(poissonSolver),
        rhs_(rhsScratch),
        phi_(phiScratch)
    {}

    /**
     * @brief Advances one RK4 step using dxdt(), then re-projects the
     *        result onto the divergence-free manifold.
     *
     * @param currentPoint Input state; must not be modified.
     * @param nextPoint    Destination, completely overwritten with a
     *                     divergence-free result.
     */
    void nextPoint(
        const Vec<Real>& currentPoint,
        Vec<Real> nextPoint
    ) const override {
        ODE<Real>::nextPoint(currentPoint, nextPoint);
        project(nextPoint);
    }

protected:
    /**
     * @brief Lyapunov deviation-vector hook (see TimeSequence::projection).
     *
     * Reuses the identical divergence-free projection used by
     * nextPoint(), applied here to a deviation vector instead of the
     * flow state itself.
     *
     * @param v Deviation vector to project, modified in place.
     */
    void projection(Vec<Real>& v) const override {
        project(v);
    }

private:
    EigenDecompSolver<Real>& poissonSolver_;

    // Written to inside const member functions (project() is called from
    // both the const nextPoint() and the const projection() overrides),
    // so these follow the same mutable-scratch convention already used
    // for EigenDecompSolver::sizeOfB.
    mutable SimpleArray<Real> rhs_;
    mutable SimpleArray<Real> phi_;

    /**
     * @brief Helmholtz-Hodge projection: removes the gradient-of-a-scalar
     *        component of @p v, leaving it divergence-free.
     *
     * @param v Field to project, modified in place.
     */
    void project(Vec<Real>& v) const {
        Handle& hand = this->handle();

        divergence(v, rhs_, hand);              // rhs_ <- div(v)
        poissonSolver_.solve(phi_, rhs_, hand);  // grad^2 phi = div(v)
        subtractGradient(v, phi_, hand);         // v <- v - grad(phi)
    }

    // --- Not implemented here: wire these to your actual grid operators. ---
    //
    // The velocity-field layout (staggered vs. collocated, how a
    // multi-component field is packed across Vec/Mat, boundary stencil
    // conventions) isn't something I have visibility into. Whatever
    // divergence/gradient kernels your existing pressure-projection code
    // already uses are almost certainly exactly what belongs here --
    // reuse them rather than writing new ones. Point me at that code and
    // I'll fill these in for real instead of leaving them as declarations.
    void divergence(const Vec<Real>& v, SimpleArray<Real>& divOut, Handle& hand) const;
    void subtractGradient(Vec<Real>& v, const SimpleArray<Real>& phi, Handle& hand) const;
};

#endif // INCOMPRESSIBLE_FLOW_H
