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

template<typename Real> class LargestLyapunovExponent;

/**
 * @brief Generates successive states using a derived stepping method.
 *
 * A derived class implements nextPoint(), for example using RK4, and
 * supplies the single GPU execution handle used for every operation via
 * handle() (see below). All state storage is supplied externally.
 *
 * This class is deliberately just the stepping interface: a single state,
 * one step at a time, nothing more. It carries no notion of Lyapunov
 * exponents or any workspace for them -- that's one specific thing you
 * might want to do with a TimeSequence, not a property every TimeSequence
 * needs to carry around. See LargestLyapunovExponent (a separate class
 * that wraps a TimeSequence rather than extending it) for that.
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

    // The one external collaborator allowed to reach handle()/norm()/
    // projection() below: it needs the exact same handle/stream this
    // sequence steps on (never a separate one), and a constrained
    // subclass's norm()/projection() overrides, to do its job correctly.
    // Nothing else gets this access -- these stay non-public for everyone
    // else, same as before this class existed.
    friend class LargestLyapunovExponent<Real>;

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

};

#endif // CUDABANDED_TIME_SEQUENCE_H
