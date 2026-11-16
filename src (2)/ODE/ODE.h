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
 * Runge-Kutta (RK4) time step as the TimeSequence stepping rule.
 *
 * An ODE holds nothing that a step writes to: the only device memory it
 * owns is the four read-only step-size constants created at
 * construction. The RK4 stage vectors are caller-supplied scratch
 * (scratchColumns() is 3), so any number of paths can step the same ODE
 * at once, each on its own Handle, through TimeSequence::iterator().
 *
 * @tparam Real Floating-point type.  The implementation is explicitly
 *              instantiated for float and double.
 */
template<typename Real>
class ODE: public  TimeSequence<Real> {

    SimpleArray<Real> scalars;
    Singleton<Real> stepSizeOver6, stepSizeOver3, stepSizeOver2, stepSize;
    Real stepSizeScalar;

public:

    /**
     * @param stepSizeScalar Fixed RK4 timestep.
     * @param hand           Handle used only to create and upload the
     *                       step-size constants; not stored. It is
     *                       synchronized before the constructor returns,
     *                       so the constants are ready for every stream.
     */
    ODE(Real stepSizeScalar, Handle& hand);

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
     * It may launch CUDA kernels on @p hand's stream.  The input and
     * output vectors must have the same logical length and must not
     * overlap.
     *
     * @param x   State vector at that time; not modified.
     * @param hand Handle whose stream all work is enqueued on.
     */
    virtual void dxdt(
        Real t,
        const Vec<Real>& x,
        Vec<Real> dst,
        Vec<Real> addToX,
        const Singleton<Real> scalarForAddToX,
        Handle& hand
    ) const = 0;

    /**
     * @brief Advances one timestep using the classical 4-stage Runge-Kutta method (RK4).
     *
     * Computes the state at @p t + stepSizeScalar from @p x, where
     * stepSizeScalar is the fixed step size supplied to the constructor.
     * The algorithm is fourth-order accurate for sufficiently smooth
     * problems and requires four evaluations of dxdt().
     *
     * No synchronization is performed. All work is enqueued on @p hand's
     * stream.
     *
     * @param t       Time associated with @p x.
     * @param x       Input state at @p t; read throughout the step and not
     *                modified until the final write to @p dst.
     * @param dst     Output state at @p t + stepSizeScalar.
     * @param scratch Exactly three columns, each as long as the state;
     *                overwritten, and not to be used by any other path
     *                while this call is in flight.
     * @param hand    Handle whose stream all work is enqueued on.
     *
     * @pre @p x and @p dst have the same logical length.
     * @note @p dst may safely alias @p x (in-place stepping is supported):
     *       every dxdt() evaluation and the final combination read @p x by
     *       index before writing the corresponding index of @p dst, so
     *       there is no read-after-write hazard even when they share
     *       storage. TimeSequence::Iterator relies on this.
     */
    void rungeKutta4(
        Real t,
        const Vec<Real>& x,
        Vec<Real> dst,
        Mat<Real> scratch,
        Handle& hand
    ) const;

    /** RK4 needs three stage vectors: the running sum and two slopes. */
    size_t scratchColumns() const override { return 3; }

    void nextPoint(
        const Vec<Real>& currentPoint,
        Vec<Real> nextPoint,
        Handle& hand,
        Mat<Real> scratch
    ) const override {
        rungeKutta4(Real{0}, currentPoint, nextPoint, scratch, hand);
    }
};


#endif // ODE_H
