/**
 * @file LargestLyapunovExponent.h
 * @brief Estimates the largest Lyapunov exponent of a TimeSequence.
 */

#ifndef LARGEST_LYAPUNOV_EXPONENT_H
#define LARGEST_LYAPUNOV_EXPONENT_H

#include "TimeSequence.h"
#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/Vec.h"
#include "deviceArrays/headers/Singleton.h"

#include <cstddef>
#include <cmath>
#include <stdexcept>

/**
 * @brief Estimates the largest Lyapunov exponent of a TimeSequence, using
 *        the rescaled deviation-vector method (Hartl, "Lyapunov exponents
 *        in constrained and unconstrained ordinary differential
 *        equations," 2003, Sec. II A).
 *
 * Wraps a TimeSequence rather than extending it: computing a Lyapunov
 * exponent is one thing you might want to do with a stepper, not a
 * property every stepper needs to carry. Its workspace (two state-length
 * columns plus a small scalar buffer) is supplied by the caller at
 * construction, not allocated here -- see the constructor.
 *
 * Every GPU operation performed here goes through sequence.handle() --
 * the same stream the wrapped sequence steps on, never a separate or
 * newly created handle -- so stepping and Lyapunov bookkeeping stay
 * correctly ordered on one stream. A constrained TimeSequence's
 * norm()/projection() overrides (see TimeSequence.h) are used
 * automatically: the raw deviation is projected onto the
 * constraint-satisfying tangent space before its size is measured.
 *
 * @tparam Real Floating-point type.
 */
template<typename Real>
class LargestLyapunovExponent {
public:
    /**
     * @param sequence The system to measure. Stored by reference, never
     *        copied; must outlive this object.
     * @param buffer Workspace: 3 columns (x, y, deviation-vector scratch),
     *        each as long as one state vector for @p sequence. Not
     *        allocated here -- supplied by the caller, who keeps
     *        ownership and can reuse it for anything else once this
     *        LargestLyapunovExponent no longer needs it, or share one
     *        allocation across several LargestLyapunovExponent instances
     *        whose state dimension matches. Requiring it up front rather
     *        than silently allocating it is deliberate: it forces whoever
     *        constructs this to decide where the memory comes from and
     *        whether it's being used efficiently, rather than leaving
     *        every instance to quietly grab its own.
     * @param singletons Scalar workspace: at least 2 slots (norm result,
     *        rescale factor), same ownership model as @p buffer.
     *
     * @throws std::invalid_argument if buffer doesn't have exactly 3
     *         columns, or singletons has fewer than 2 elements. There's
     *         no way to check buffer's row count against sequence's
     *         actual state length from in here -- TimeSequence is
     *         deliberately dimension-agnostic -- so getting that right is
     *         still on the caller.
     */
    LargestLyapunovExponent(TimeSequence<Real>& sequence, Mat<Real> buffer, Vec<Real> singletons)
        : sequence_(sequence), buffer_(buffer), singletons_(singletons)
    {
        if (buffer._cols != 3 || singletons.size() < 2) {
            throw std::invalid_argument(
                "LargestLyapunovExponent requires a 3-column buffer and at least 2 singletons."
            );
        }
    }

    /**
     * @brief Evolves the stored x/y state over one interval tau, measures
     *        divergence, and pulls y back to distance epsilon from x
     *        along their vector difference.
     *
     * x, y, and the deviation-vector scratch are columns 0, 1, and 2 of
     * the workspace supplied at construction, and carry over from the
     * previous call, so repeated calls continue the same run rather than
     * starting fresh each time.
     *
     * The raw difference y - x is passed through sequence's projection()
     * (a no-op unless overridden) before its size is measured with
     * sequence's norm() (the plain Euclidean norm unless overridden), so
     * a constrained system's spurious directions never contribute to the
     * measured divergence.
     *
     * @param stepsPerInterval Number of integration steps per interval tau.
     * @param epsilon Target perturbation distance.
     * @return Logarithmic expansion/divergence ln(d / epsilon) for this interval.
     *
     * @pre x and y hold a valid state/deviation pair -- normally
     *      established by compute()'s own setup before its first call
     *      into this method.
     */
    Real interval(size_t stepsPerInterval, Real epsilon) const {
        Handle& hand = sequence_.handle();

        auto x = buffer_.col(0);
        auto y = buffer_.col(1);
        auto v = buffer_.col(2);

        Singleton<Real> normVal = singletons_.get(0);
        Singleton<Real> scaleScalar = singletons_.get(1);
        const Singleton<Real>& one = GPUScalar<Real>::get(1, hand);

        for (size_t step = 0; step < stepsPerInterval; ++step) {
            sequence_.nextPoint(x, x);
            sequence_.nextPoint(y, y);
        }

        v.setDifference(y, x, one, one, &hand);
        sequence_.projection(v);

        sequence_.norm(v, normVal);
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
     * Builds an initial baseline/deviation pair from initialPoint -- a
     * random deviation of magnitude epsilon away from it -- then projects
     * the resulting deviated point y through sequence's projection() so
     * it satisfies the constraint exactly, before handing off to
     * interval() for each successive interval.
     *
     * @param initialPoint Initial baseline state vector.
     * @param numIntervals Number of tau intervals to run.
     * @param stepsPerInterval Number of stepping steps per interval tau.
     * @param epsilon Initial perturbation magnitude.
     * @param dt Integration step size (defaults to 1 for discrete maps).
     *        Not stored as a member: there's no reason to assume this
     *        stays fixed from one call to the next, and this class has no
     *        notion of a step size on its own -- it is purely the unit
     *        conversion used in the final division below.
     * @return Calculated largest Lyapunov exponent.
     */
    Real compute(
        const Vec<Real>& initialPoint,
        size_t numIntervals,
        size_t stepsPerInterval,
        Real epsilon,
        Real dt = static_cast<Real>(1)
    ) const {
        Handle& hand = sequence_.handle();

        auto x = buffer_.col(0);
        auto y = buffer_.col(1);
        auto v = buffer_.col(2);

        Singleton<Real> normVal = singletons_.get(0);
        Singleton<Real> scaleScalar = singletons_.get(1);
        const Singleton<Real>& one = GPUScalar<Real>::get(1, hand);

        x.set(initialPoint, hand);

        v.fillRandom(&hand);

        sequence_.norm(v, normVal);
        Real vNorm = normVal.get(hand);

        scaleScalar.set(epsilon / vNorm, hand);
        v.mult(scaleScalar, &hand);

        y.set(x, hand);
        y.add(v, &one, &hand);
        sequence_.projection(y);

        Real totalLogDivergence = static_cast<Real>(0);

        for (size_t i = 0; i < numIntervals; ++i)
            totalLogDivergence += interval(stepsPerInterval, epsilon);

        return totalLogDivergence / (numIntervals * stepsPerInterval * dt);
    }

private:
    TimeSequence<Real>& sequence_;
    mutable Mat<Real> buffer_;
    mutable Vec<Real> singletons_;
};

#endif // LARGEST_LYAPUNOV_EXPONENT_H
