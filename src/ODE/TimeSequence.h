/**
 * @file timeSequence.h
 * @brief Abstract GPU state sequence with trajectory and repeated-step methods.
 */

#ifndef CUDABANDED_TIME_SEQUENCE_H
#define CUDABANDED_TIME_SEQUENCE_H

#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/Vec.h"

#include <cstddef>
#include <stdexcept>

#include "deviceArrays/headers/Singleton.h"

/**
 * @brief Generates successive states using a derived stepping method.
 *
 * A derived class implements nextPoint(), for example using RK4, and
 * supplies the single GPU execution handle used for every operation via
 * handle() (see below). All state storage is supplied externally.
 *
 * Operations use handle()'s stream without synchronizing. Returned
 * Vec/Mat objects are shallow wrappers sharing the supplied GPU storage.
 *
 * @tparam Real State element type.
 */
template<typename Real>
class TimeSequence {
public:

    virtual ~TimeSequence() = default;

    /**
     * @brief Computes one successor state.
     * @param currentPoint Input state; must not be modified.
     * @param nextPoint Destination, completely overwritten.
     *
     * @pre Input and destination have equal lengths and disjoint storage.
     * @note The implementation must enqueue its work on handle()'s stream.
     */
    virtual void nextPoint(
        const Vec<Real>& currentPoint,
        Vec<Real> nextPoint
    ) const = 0;

    /**
     * @brief Fills a trajectory beginning with the state in column zero.
     * @param points Matrix containing one state per column.
     *
     * Column zero is preserved. Each subsequent column contains the successor
     * of the previous column. Zero or one column requires no stepping.
     *
     * Each successive column depends on the one before it, so every step
     * is issued on handle()'s single stream in sequence regardless -- there
     * is never a reason for a different handle per call.
     */
    void trajectory(Mat<Real>& points) const {
        for (size_t col = 1; col < points._cols; ++col) {
            auto current = points.col(col - 1);
            auto destination = points.col(col);
            nextPoint(current, destination);
        }
    }

    /**
     * @brief Computes the state after t steps without retaining a trajectory.
     * @param t Integer number of steps; zero copies the initial state.
     * @param initialPoint Starting state, preserved.
     * @param result Caller-owned destination for the final state.
     * @return Shallow copy of result.
     *
     * @pre initialPoint, result, and workspace have equal lengths.
     * @pre Their storage is mutually non-overlapping.
     * @pre Neither result nor workspace overlaps the derived stepper's scratch.
     */
    void pointAt(
        size_t t,
        const Vec<Real>& initialPoint,
        Vec<Real> result
    ) const {
        Handle& hand = handle();

        result.set(initialPoint, hand);

        for (size_t step = 0; step < t; ++step)
            nextPoint(result, result);
    }

protected:

    /**
     * @brief The GPU execution handle used for every operation this
     *        sequence performs.
     *
     * Pure virtual: TimeSequence itself owns no Handle. Every point in a
     * sequence depends on the one before it, so there is never a reason
     * to use a different handle/stream across calls on the same object --
     * a derived class is expected to store exactly one Handle (by
     * reference) and return it here, rather than every method threading
     * its own handle argument through the whole class.
     *
     * @return Reference to the handle this sequence's derived class owns.
     */
    virtual Handle& handle() const = 0;

    /**
     * @brief Measures the size of a deviation vector.
     *
     * Default: the underlying Vec's own Euclidean norm over all n
     * components -- correct for an unconstrained system, where every
     * coordinate is a true degree of freedom. A derived class representing
     * a constrained system may override this with a restricted norm that
     * only measures the d true degrees of freedom, so that a redundant
     * (constraint-dependent) coordinate carried in the state never
     * inflates the measured divergence.
     *
     * @param v          Deviation vector to measure.
     * @param normResult Destination for the scalar result.
     */
    virtual void norm(Vec<Real>& v, Singleton<Real>& normResult) const {
        v.norm(normResult, handle());
    }

    /**
     * @brief Projects a deviation vector onto the constraint-satisfying
     *        tangent space, in place.
     *
     * Default is a no-op: every coordinate is a true degree of freedom,
     * correct for an unconstrained system. A derived class representing a
     * constrained system overrides this to remove whatever component of
     * @p v would violate the constraint (e.g. a divergence-free
     * projection for an incompressible velocity field).
     *
     * @param v Deviation vector to project, modified in place.
     */
    virtual void projection(Vec<Real>& v) const {}

public:

    /**
     * @brief Evolves states x and y over one interval tau, measures divergence,
     *        and pulls y back to distance epsilon from x along their vector difference.
     *
     * The raw difference y - x is passed through projection() (a no-op
     * unless overridden) before its size is measured with norm() (the
     * plain Euclidean norm unless overridden), so a constrained system's
     * spurious directions never contribute to the measured divergence.
     *
     * @param bufferNX3 Pre-allocated workspace matrix (col 0 = x, col 1 = y, col 2 = v).
     * @param singletons2 Pre-allocated vector for scalar workspace (indices 0 and 1).
     * @param stepsPerInterval Number of integration steps per interval tau.
     * @param epsilon Target perturbation distance.
     * @return Logarithmic expansion/divergence ln(d / epsilon) for this interval.
     */
    Real lyapunovInterval(
        Vec<Real>& x,
        Vec<Real>& y,
        Vec<Real>& v,
        Vec<Real>& singletons2,
        size_t stepsPerInterval,
        Real epsilon
    ) const {
        Handle& hand = handle();

        Singleton<Real> normVal = singletons2.get(0);
        Singleton<Real> scaleScalar = singletons2.get(1);
        const Singleton<Real>& one = GPUScalar<Real>::get(1, hand);

        for (size_t step = 0; step < stepsPerInterval; ++step) {
            nextPoint(x, x);
            nextPoint(y, y);
        }

        v.setDifference(y, x, one, one, &hand);
        projection(v);

        norm(v, normVal);
        Real d = normVal.get(hand);

        scaleScalar.set(epsilon / d, hand);
        v.mult(scaleScalar, &hand);

        y.set(x, hand);
        y.add(v, &one, &hand);

        return std::log(d / epsilon);
    }

    /**
     * @brief Estimates the maximal Lyapunov exponent across multiple tau intervals.
     *
     * The initial random deviation is passed through projection() before
     * being measured with norm() and normalized to size epsilon.
     *
     * @param initialPoint Initial baseline state vector.
     * @param buffer Pre-allocated workspace matrix (needs at least 3 columns).
     * @param singletons Pre-allocated vector workspace (needs at least 2 singletons).
     * @param numIntervals Number of tau intervals to run.
     * @param stepsPerInterval Number of stepping steps per interval tau.
     * @param epsilon Initial perturbation magnitude.
     * @param dt Integration step size (defaults to 1 for discrete maps).
     * @return Calculated largest Lyapunov exponent.
     */
    Real largestLyapunovExponent(
        const Vec<Real>& initialPoint,
        Mat<Real>& buffer,
        Vec<Real>& singletons,
        size_t numIntervals,
        size_t stepsPerInterval,
        Real epsilon,
        Real dt = static_cast<Real>(1)
    ) const {
        Handle& hand = handle();

        auto x = buffer.col(0);
        auto y = buffer.col(1);
        auto v = buffer.col(2);

        Singleton<Real> normVal = singletons.get(0);
        Singleton<Real> scaleScalar = singletons.get(1);
        const Singleton<Real>& one = GPUScalar<Real>::get(1, hand);

        x.set(initialPoint, hand);

        v.fillRandom(&hand);
        projection(v);

        norm(v, normVal);
        Real vNorm = normVal.get(hand);

        scaleScalar.set(epsilon / vNorm, hand);
        v.mult(scaleScalar, &hand);

        y.set(x, hand);
        y.add(v, &one, &hand);

        Real totalLogDivergence = static_cast<Real>(0);

        for (size_t interval = 0; interval < numIntervals; ++interval) {
            totalLogDivergence += lyapunovInterval(
                x, y, v,
                singletons,
                stepsPerInterval,
                epsilon
            );
        }

        Real totalTime = static_cast<Real>(numIntervals * stepsPerInterval) * dt;
        return totalLogDivergence / totalTime;
    }

};

#endif // CUDABANDED_TIME_SEQUENCE_H
