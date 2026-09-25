/**
 * @file ODE.h
 * @brief Abstract GPU ODE and an allocation-free classical (4-stage) RK4 integrator.
 */

#ifndef ODE_H
#define ODE_H

#include <cstddef>

#include "deviceArrays/headers/SimpleArray.h"
#include "deviceArrays/headers/Singleton.h"
#include "TimeSequence.h"

template <typename T> class Vec;
template <typename T> class Singleton;
class Handle;


/**
 * @class ODE
 * @brief Base class for systems of first-order ordinary differential equations.
 *
 * A derived class defines
 *
 * @f[
 *     \frac{d\mathbf{x}}{dt} = \mathbf{f}(t,\mathbf{x})
 * @f]
 *
 * by overriding dxdt().  ODE supplies an explicit classical 4-stage
 * Runge-Kutta (RK4) time step and satisfies TimeSequence::handle() using
 * the single Handle supplied at construction.
 *
 * All state and workspace vectors reside on the GPU.  This class never
 * allocates device or host workspace.  The caller owns every Vec supplied to
 * nextStep() and stepsOut().
 *
 * @tparam Real Floating-point type.  The implementation is explicitly
 *              instantiated for float and double.
 */
template<typename Real>
class ODE: public  TimeSequence<Real> {

    mutable Mat<Real> buffers;
    SimpleArray<Real> scalars;
    Singleton<Real> stepSizeOver6, stepSizeOver3, stepSizeOver2, stepSize;
    Real stepSizeScalar;
    Handle& hand;

public:

    /**
     * @param numDims        Logical length of the state vector.
     * @param stepSizeScalar Fixed RK4 timestep.
     * @param hand           GPU execution handle. Stored by reference for
     *                        the lifetime of this object and returned by
     *                        handle() -- see TimeSequence::handle(). The
     *                        caller must keep it alive at least as long
     *                        as this ODE.
     */
    ODE(size_t numDims, Real stepSizeScalar, Handle& hand);

    virtual ~ODE() = default;

    /**
     * @brief Evaluates the ODE right-hand side at a state and time.
     *
     * A derived implementation must compute
     *
     * @f[
     *     \mathtt{derivative} = \mathbf{f}(\mathtt{location}).
     * @f]
     *
     * with dst = f(x + scalar * addToX)
     * It may launch CUDA kernels on handle()'s stream (see
     * TimeSequence::handle()).  The input and output vectors must have the
     * same logical length and must not overlap.
     *
     * @param x   State vector at that time; not modified.
     */
    virtual void dxdt(Real t, const Vec<Real>& x, Vec<Real> dst, Vec<Real> addToX, const Singleton<Real> scalarForAddToX) const = 0;

    /**
     * @brief Advances one timestep using the classical 4-stage Runge-Kutta method (RK4).
     *
     * Computes the state at @p t + stepSizeScalar from @p x, where
     * stepSizeScalar is the fixed step size supplied to the constructor.
     * The algorithm is fourth-order accurate for sufficiently smooth
     * problems and requires four evaluations of dxdt().
     *
     * No synchronization is performed. All work is enqueued on handle()'s
     * stream.
     *
     * @param t   Time associated with @p x.
     * @param x   Input state at @p t; read throughout the step and not
     *            modified until the final write to @p dst.
     * @param dst Output state at @p t + stepSizeScalar.
     *
     * @pre @p x and @p dst have the same logical length.
     * @note @p dst may safely alias @p x (in-place stepping is supported):
     *       every dxdt() evaluation and the final combination read @p x by
     *       index before writing the corresponding index of @p dst, so
     *       there is no read-after-write hazard even when they share
     *       storage. TimeSequence::pointAt() and
     *       TimeSequence::lyapunovInterval() rely on this.
     */
    void rungeKutta4(Real t, const Vec<Real> &x, Vec<Real> dst) const;

    void nextPoint(
        const Vec<Real>& currentPoint,
        Vec<Real> nextPoint
    ) const override {
        rungeKutta4(Real{0}, currentPoint, nextPoint);
    }

protected:
    /**
     * @brief Returns the Handle supplied at construction.
     *
     * Satisfies TimeSequence::handle(). Shared by every ODE subclass --
     * e.g. Lorenz does not need to provide its own.
     */
    Handle& handle() const override { return hand; }

    ODE(const ODE&) = default;
    ODE(ODE&&) = default;
    // Default construction and copy/move assignment are intentionally
    // unavailable: hand_ is a reference, so it must be bound to a real
    // Handle at construction and can never be rebound afterward. (This
    // is a change from before hand_ existed, when ODE() and both
    // assignment operators were defaulted; check whether anything relied
    // on default-constructing or assigning an ODE.)
};


#endif // ODE_H
