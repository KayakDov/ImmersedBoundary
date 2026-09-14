/**
 * @file ODE.h
 * @brief Abstract GPU ODE and an allocation-free third-order integrator.
 */

#ifndef ODE_H
#define ODE_H

#include <cstddef>

#include "deviceArrays/headers/SimpleArray.h"
#include "deviceArrays/headers/Singleton.h"

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
 * by overriding equation().  ODE supplies an explicit third-order
 * strong-stability-preserving Runge--Kutta (SSPRK(3,3)) time step and a method
 * for taking several such steps.
 *
 * All state and workspace vectors reside on the GPU.  This class never
 * allocates device or host workspace.  The caller owns every Vec supplied to
 * nextStep() and stepsOut().
 *
 * @tparam Real Floating-point type.  The implementation is explicitly
 *              instantiated for float and double.
 */
template<typename Real>
class ODE {

    mutable Mat<Real> buffers;
    SimpleArray<Real> scalars;
    Singleton<Real> stepSizeOver6, stepSizeOver3, stepSizeOver2, stepSize;
    Real stepSizeScalar;

public:

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
     * with dst += scalar * dxdt
     * It may launch CUDA kernels on @p handle.  The input and output vectors must have the same
     * logical length and must not overlap.
     *
     * @param x   State vector at that time; not modified.
     * @param handle     Handle whose CUDA stream is used for all work.
     */
    virtual void dxdt(Real t, const Vec<Real>& x, Vec<Real> dst, Vec<Real> addToX, const Singleton<Real> scalarForAddToX, Handle& handle) const = 0;

    /**
     * @brief Advances one timestep with SSPRK(3,3).
     *
     * Computes the state at @p time + @p timeStep from @p location.  The
     * algorithm is third-order accurate for sufficiently smooth problems and
     * requires three evaluations of equation().
     *
     * No synchronization is performed.  All work is enqueued on the stream
     * represented by @p handle.
     *
     * @param t             Time associated with @p location.
     * @param timeStep         Timestep; may be negative for backward stepping.
     * @param x         Input state at @p time.
     * @param nextLocation     Output state at @p time + @p timeStep.
     * @param stageBuffer      Caller-owned state-sized RK stage buffer.
     * @param derivativeBuffer Caller-owned state-sized derivative buffer.Runge
     * @param handle           Handle whose CUDA stream is used.
     *
     * @pre All four vectors have the same logical length.
     * @pre @p location and @p nextLocation do not overlap.
     * @pre Neither workspace vector overlaps another argument.
     */
    void rungeKutta4(Real t, Vec<Real>& x, Handle& handle) const;

    /**
     * @brief Advances a starting state by a specified number of timesteps.
     *
     * On return, @p result contains the state at
     *
     * @f[
     *     \mathtt{startTime} + \mathtt{numberOfSteps}\,\mathtt{timeStep}.
     * @f]
     *
     * State buffers are ping-ponged internally by reference; no storage is
     * allocated.  Depending on the parity of @p numberOfSteps, one final
     * device-to-device copy may be enqueued to ensure that the answer is in
     * @p result.
     *
     * @param startTime            Time associated with @p startingLocation.
     * @param timeStep             Size of each timestep.
     * @param numberOfSteps        Number of timesteps to take.
     * @param x0     Initial state; not modified.
     * @param result               Caller-owned vector receiving the final state.
     * @param alternateStateBuffer Caller-owned state-sized ping-pong buffer.
     * @param stageBuffer          Caller-owned state-sized RK stage buffer.
     * @param derivativeBuffer     Caller-owned state-sized derivative buffer.
     * @param handle               Handle whose CUDA stream is used.
     *
     * @pre Every vector has the same logical length.
     * @pre All state and workspace vectors are mutually non-overlapping.
     */
    void stepsOut(size_t numberOfSteps, const Vec<Real>& x0, Handle& handle) const;

protected:
    ODE() = default;
    ODE(const ODE&) = default;
    ODE& operator=(const ODE&) = default;
    ODE(ODE&&) = default;
    ODE& operator=(ODE&&) = default;
};


#endif // ODE_H
