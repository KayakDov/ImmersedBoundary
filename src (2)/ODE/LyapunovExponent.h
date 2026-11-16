/**
 * @file LyapunovExponent.h
 * @brief Estimates the top n Lyapunov exponents of a TimeSequence, one handle per exponent.
 */

#ifndef LYAPUNOV_EXPONENT_H
#define LYAPUNOV_EXPONENT_H

#include "TimeSequence.h"
#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/Vec.h"
#include "deviceArrays/headers/Singleton.h"
#include "Norm.h"
#include "deviceArrays/headers/Handles.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "deviceArrays/headers/Support/Streamable.h"

/**
 * @brief Estimates the n largest Lyapunov exponents of a TimeSequence
 *        with the deviation-vector method, orthonormalizing the
 *        deviations by Gram-Schmidt after every interval (Hartl,
 *        "Lyapunov exponents in constrained and unconstrained ordinary
 *        differential equations," 2003, Sec. II B 4).
 *
 * Each interval starts n + 1 paths through the sequence: a reference
 * path from x, and one path per exponent from its Gram-Schmidt point
 * y_i = x + epsilon q_i. They advance stepsPerInterval steps, the
 * deviations v_i = y_i - x are orthonormalized left to right, and the
 * length v_i has just before it is normalized is how far the i-th axis
 * stretched over the interval: ln(length / epsilon) summed over
 * intervals and divided by the elapsed time estimates the i-th
 * exponent. Each y_i is then reset to x + epsilon q_i for the next
 * interval. Column 0 is only rescaled, so it follows the fastest-growing
 * direction and its exponent is the largest.
 *
 * The paths are iterators created at the start of each interval and
 * discarded at its end; the class holds no path between intervals,
 * only the points they start from.
 *
 * This takes a Handles group of exactly n handles, one per exponent.
 * handle i advances the path for y_i; handle 0 also advances the reference
 * path and forms the differences, which leaves that stream with extra
 * work. An extra handle for the reference alone would balance it.
 *
 * Each interval has four joins, all through Handles and never through the
 * host: handle 0 waits for the others when stepping is done, the others
 * wait for handle 0 when the differences are in, handle 0 waits for the
 * others when their resets are done, and the others wait for handle 0
 * again before anyone steps. The second of these is the entry to Gram-Schmidt,
 * which spreads its columns over the handles. Stepping, Gram-Schmidt and
 * the resets are the parallel phases; only the differences and the
 * accumulation run on hand[0] alone.
 *
 * The deviations v_i share their columns with the stepping scratch. They
 * are alive only between the first join and the last, a window in which
 * nothing steps, and the scratch is alive only while stepping. That is
 * also why the last join signals every handle and not just hand[0]: a
 * handle's scratch can sit on columns another handle is still reading
 * as a v_i.
 *
 * Nothing is allocated here: the workspace, handles and scratch are
 * supplied by the caller. See bufferColumns() and numSingletons().
 *
 * @par Choice of norm
 * Gram-Schmidt and the reset are always Euclidean, so every perturbation
 * is exactly epsilon long whatever the norm. compute() and interval()
 * take a Norm, Euclidean by default, and it changes only how growth is
 * measured. A pseudo-norm N gives the reset perturbation a length of its
 * own: an interval starts with the deviation epsilon q_i, whose length
 * under N is epsilon N(q_i) and not epsilon, and ends with r_i q_i',
 * where r_i is the Euclidean length Gram-Schmidt reports and q_i' the
 * new direction. The growth is the ratio, ln(r_i N(q_i') / (epsilon
 * N(q_i))), so each axis's N(q_i) is remembered from one interval to the
 * next, measured once at the start of a run and after that taken from
 * the previous interval's end. Dividing by epsilon instead is wrong, not
 * merely inexact. Over a run the N(q) terms cancel in pairs and leave
 * only the first and the last, so the exponents agree with the Euclidean
 * ones up to those end terms.
 *
 * Every norm takes this same path, the Euclidean one included. Under it
 * N(q) is 1 for every unit direction and the ratio reduces to
 * ln(r_i / epsilon) with no special case, at the price of n measurements
 * per interval that the Euclidean case could have skipped. They run on
 * the handles that carry the paths, in parallel, and next to the stepping
 * they are small.
 *
 * @par Growth stays on the device
 * Nothing in an interval touches the host, so it can enqueue intervals
 * ahead of the GPU, up to the depth of the launch queue. At the end of
 * each interval, on hand[0], each axis's growth ratio
 * (r_i / epsilon)(N(q_i') / N(q_i)) is formed by
 * Singleton::setProductOfQuotients, raised to the power 1/T by
 * Vec::EBEPow, where T is the elapsed time of the whole run, and
 * multiplied into a running product. The product over the run is
 * exp(lambda_i), which the host reads once, at the end of compute(), and
 * takes the logarithm of. No kernel of its own is needed: there is no
 * logarithm on the device, so the logarithm is taken once on the host
 * instead of once per interval.
 *
 * The 1/T is folded in at each interval, not applied at the end, because
 * the raw ratios do not fit in a double over a long run: Lorenz's third
 * axis shrinks by about e^-1.46 every interval of 0.1, and its product
 * underflows to zero within a few hundred intervals. Raised to 1/T each
 * ratio stays near 1, and the product ends at exp(lambda_i). A norm that
 * measures a deviation as zero length turns a product into 0 or NaN, and
 * is noticed at the end of the run, as an exponent that is not finite,
 * and not in the interval where it happened.
 *
 * @par Required Mat method
 * `void Mat<Real>::gramSchmidt(Mat<Real>& dst, Vec<Real>& lengths, Vec<Real>& inverseLengths, Handles& hands) const`
 * orthonormalizes the columns of this matrix into dst, left to right
 * under the Euclidean inner product: column 0 is only normalized, column
 * 1 first has its component along column 0 removed, and so on.
 * lengths[i] receives the length column i had immediately before it was
 * normalized (the diagonal of R in A = QR), and inverseLengths[i] its
 * reciprocal. dst may be the same matrix as this one. It takes one handle
 * per column and spreads the columns over them: the source
 * must be ready on hands[0]'s stream, and on return column i, lengths[i]
 * and inverseLengths[i] are ready on hands[i]'s stream, not on hands[0]'s,
 * so the caller joins before reading across handles. Here lengths is
 * norms_, and inverseLengths borrows ratios_, which is not used again
 * until the end of the interval.
 *
 * @tparam Real Floating-point type.
 */
template<typename Real>
class LyapunovExponent {
public:
    /**
     * @brief Columns the workspace buffer must have.
     *
     * One column for x, n for the points y_i, and then one shared
     * region, as wide as the larger of what it holds: the n deviations
     * v_i, or the stepping scratch, scratchColumns() for each of the n
     * handles.
     */
    static size_t bufferColumns(const TimeSequence<Real>& sequence, size_t numExponents) {
        return 1 + numExponents + std::max(numExponents, numExponents * sequence.scratchColumns());
    }

    /**
     * @brief Scalars the singletons vector must have: n Gram-Schmidt
     *        lengths, n measurements under the norm, n start lengths,
     *        n growth ratios, n running products, epsilon, and the
     *        exponent 1/T.
     */
    static size_t numSingletons(size_t numExponents) {
        return 5 * numExponents + 2;
    }

    /**
     * @param sequence   The system to measure. Stored by reference,
     *                   never copied; must outlive this object.
     * @param buffer     Workspace of bufferColumns() columns, each as
     *                   long as one state vector. Laid out as x, then
     *                   y_0..y_{n-1}, then the shared region: v_0..v_{n-1}
     *                   from its first column, and one scratch block per
     *                   handle from its first column, overlapping. Not
     *                   allocated here: the caller keeps ownership and
     *                   can reuse it for anything else once this object
     *                   is done with it.
     * @param singletons numSingletons() contiguous scalars, in this
     *                   order: the n lengths Gram-Schmidt reports, the n
     *                   measurements under the norm, the n start
     *                   lengths, the n growth ratios, the n running
     *                   products, epsilon, and the exponent 1/T. This is
     *                   also how the number of exponents n is known.
     * @param hands      Exactly n handles, one per exponent, as described
     *                   above. Must outlive this object.
     *
     * @throws std::invalid_argument if singletons doesn't have a count of
     *         the form 5n + 2 with n at least one, hands doesn't have
     *         exactly n handles, or buffer doesn't have bufferColumns()
     *         columns. The row count of buffer can't be checked against
     *         the state length here -- TimeSequence is deliberately
     *         dimension-agnostic -- so that stays on the caller.
     */
    LyapunovExponent(
        TimeSequence<Real>& sequence,
        Mat<Real> buffer,
        Vec<Real> singletons,
        Handles& hands
    ) : numExponents(validated(sequence, buffer, singletons, hands)),
        sequence_(sequence),
        hands_(hands),
        x_(buffer.col(0)),
        y_(buffer.subMat(0, 1, buffer._rows, numExponents)),
        v_(buffer.subMat(0, 1 + numExponents, buffer._rows, numExponents)),
        scratch_(buffer.subMat(0, 1 + numExponents, buffer._rows, numExponents * sequence.scratchColumns())),
        norms_(singletons.vec(0, 1, numExponents)),
        measured_(singletons.vec(numExponents, 1, numExponents)),
        startLengths_(singletons.vec(2 * numExponents, 1, numExponents)),
        ratios_(singletons.vec(3 * numExponents, 1, numExponents)),
        products_(singletons.vec(4 * numExponents, 1, numExponents)),
        epsilonScalar_(singletons.get(5 * numExponents)),
        exponentScalar_(singletons.get(5 * numExponents + 1))
    {
    }

    /**
     * @brief Runs one interval: advances the reference and every
     *        deviated path, orthonormalizes the deviations, multiplies
     *        each axis's growth into its running product on the device,
     *        and resets the deviated points.
     *
     * x, y_i, epsilon, the exponent 1/T, each axis's start length and
     * its running product are held in the workspace supplied at
     * construction and carry over between calls, so repeated calls
     * continue the same run. Nothing is read back: the host only
     * enqueues work.
     *
     * @param stepsPerInterval Number of integration steps per interval.
     * @param norm             How growth is measured. Must be the norm
     *                         the run was started with.
     *
     * @pre x holds the reference state, each y_i holds its Gram-Schmidt
     *      point, and epsilon, 1/T, the start lengths and the products
     *      are set -- normally established by compute() before its first
     *      interval, and kept so by this method at the end of every one.
     */
    void interval(
        size_t stepsPerInterval,
        const Norm<Real>& norm = euclideanNorm<Real>()
    ) const {
        Handle& lead = hands_[0];
        const Singleton<Real>& one = GPUScalar<Real>::get(1, lead);

        referencePath().advance(stepsPerInterval);
        for (size_t exponentIndex = 0; exponentIndex < numExponents; ++exponentIndex)
            deviatedPath(exponentIndex).advance(stepsPerInterval);

        hands_.waitForAll(0);

        for (size_t exponentIndex = 0; exponentIndex < numExponents; ++exponentIndex)
            v_.col(exponentIndex).setDifference(y_.col(exponentIndex), x_, one, one, &lead);

        orthonormalize();

        for (size_t exponentIndex = 0; exponentIndex < numExponents; ++exponentIndex) {
            Handle& hand = hands_[exponentIndex];
            auto deviated = y_.col(exponentIndex);
            deviated.set(x_, hand);
            deviated.add(v_.col(exponentIndex), &epsilonScalar_, &hand);
            norm.measure(v_.col(exponentIndex), measured_.get(exponentIndex), hand);
        }

        hands_.waitForAll(0);

        accumulateGrowth();

        hands_.allWaitFor(0);
    }

    /**
     * @brief Estimates the top n Lyapunov exponents over many intervals.
     *
     * Starts the reference path at initialPoint, advances it warmupSteps
     * steps, draws random deviations, orthonormalizes them, places each
     * deviated point at x + epsilon q_i (projected through the
     * sequence's projection()), and runs numIntervals intervals. Blocks
     * until every handle is idle.
     *
     * epsilon and 1/T are uploaded once here and reused by every interval,
     * and the products are read back once, after the last.
     *
     * @param initialPoint     Baseline state; must already be ready on
     *                         hand[0]'s stream.
     * @param numIntervals     Number of intervals to run.
     * @param stepsPerInterval Integration steps per interval.
     * @param epsilon          Perturbation size each interval starts from.
     * @param dt               Integration step size (defaults to 1 for
     *                         discrete maps). Not stored: this class has
     *                         no notion of a step size on its own, and it
     *                         is purely the unit conversion in the final
     *                         division.
     * @param warmupSteps      Steps the reference path takes before the
     *                         deviations are drawn, so measurement starts
     *                         on the attractor rather than in the approach
     *                         to it. Not counted toward the elapsed time.
     * @param norm             How growth is measured; see the class
     *                         comment.
     * @return The n exponents, largest first.
     *
     * @throws std::invalid_argument if numIntervals or stepsPerInterval
     *         is zero.
     * @throws std::runtime_error if an exponent is not finite, which is
     *         what a norm that measures a deviation as zero length
     *         produces.
     */
    std::vector<Real> compute(
        const Vec<Real>& initialPoint,
        size_t numIntervals,
        size_t stepsPerInterval,
        Real epsilon,
        Real dt = static_cast<Real>(1),
        size_t warmupSteps = 0,
        const Norm<Real>& norm = euclideanNorm<Real>()
    ) const {
        if (numIntervals == 0 || stepsPerInterval == 0) throw std::invalid_argument("There must be at least one interval of at least one step.");

        Handle& lead = hands_[0];

        x_.set(initialPoint, lead);
        referencePath().advance(warmupSteps);

        for (size_t exponentIndex = 0; exponentIndex < numExponents; ++exponentIndex)
            v_.col(exponentIndex).fillRandom(&hands_[exponentIndex], exponentIndex);

        orthonormalize();
        hands_.waitForAll(0);

        const Real totalTime = numIntervals * stepsPerInterval * dt;

        epsilonScalar_.set(epsilon, lead);
        exponentScalar_.set(static_cast<Real>(1) / totalTime, lead);
        for (size_t exponentIndex = 0; exponentIndex < numExponents; ++exponentIndex) {
            auto deviated = y_.col(exponentIndex);
            deviated.set(x_, lead);
            deviated.add(v_.col(exponentIndex), &epsilonScalar_, &lead);
            sequence_.projection(deviated, lead);
        }

        measureStartLengths(norm);

        hands_.allWaitFor(0);

        for (size_t intervalIndex = 0; intervalIndex < numIntervals; ++intervalIndex)
            interval(stepsPerInterval, norm);

        std::vector<Real> exponents(numExponents);
        products_.get(exponents.data(), lead);

        hands_.synch();

        for (size_t exponentIndex = 0; exponentIndex < numExponents; ++exponentIndex) {
            const Real product = exponents[exponentIndex];
            exponents[exponentIndex] = std::log(product);

            if (!std::isfinite(exponents[exponentIndex]))
                throw std::runtime_error(
                    "Exponent " + std::to_string(exponentIndex) + " of " + std::to_string(numExponents)
                    + " is not finite: its running product is " + std::to_string(product)
                    + ". The norm may have measured a deviation as zero length."
                );
        }

        return exponents;
    }

private:
    const size_t numExponents;
    TimeSequence<Real>& sequence_;
    Handles& hands_;
    mutable Vec<Real> x_;
    mutable Mat<Real> y_, v_, scratch_;
    mutable Vec<Real> norms_, measured_, startLengths_, ratios_, products_;
    mutable Singleton<Real> epsilonScalar_, exponentScalar_;

    /**
     * @brief Orthonormalizes the deviations in place, spread over every
     *        handle, leaving their Euclidean lengths in norms_.
     *
     * The deviations must be ready on hand[0]'s stream. Afterwards
     * deviation i is ready on hand[i]'s stream, not on hand[0]'s.
     */
    void orthonormalize() const {
        v_.gramSchmidt(v_, norms_, ratios_, hands_);
    }

    /**
     * @brief Measures every freshly orthonormalized deviation under
     *        @p norm, keeps the results as the start lengths of the
     *        first interval, and sets the running products to 1, all on
     *        hand[0].
     */
    void measureStartLengths(const Norm<Real>& norm) const {
        Handle& lead = hands_[0];

        for (size_t exponentIndex = 0; exponentIndex < numExponents; ++exponentIndex)
            norm.measure(v_.col(exponentIndex), measured_.get(exponentIndex), lead);

        startLengths_.set(measured_, lead);
        products_.fill(static_cast<Real>(1), lead);
    }

    /**
     * @brief Multiplies the interval just run into every running
     *        product, on hand[0], and keeps this interval's measurement
     *        as the next one's start length.
     *
     * Each axis's growth ratio is (length / epsilon) * (measured / start),
     * and enters the product raised to the power 1/T. Under the Euclidean
     * norm the measurement is 1 and the ratio is length / epsilon.
     */
    void accumulateGrowth() const {
        Handle& lead = hands_[0];
        const Singleton<Real>& one = GPUScalar<Real>::get(1, lead);

        for (size_t exponentIndex = 0; exponentIndex < numExponents; ++exponentIndex)
            ratios_.get(exponentIndex).setProductOfQuotients(
                norms_.get(exponentIndex),
                epsilonScalar_,
                measured_.get(exponentIndex),
                startLengths_.get(exponentIndex),
                lead
            );

        ratios_.EBEPow(one, exponentScalar_, lead);

        for (size_t exponentIndex = 0; exponentIndex < numExponents; ++exponentIndex) {
            Singleton<Real> product = products_.get(exponentIndex);
            product.setProductOfQuotients(product, one, ratios_.get(exponentIndex), one, lead);
        }

        startLengths_.set(measured_, lead);
    }


    /**
     * @brief A path through the sequence from the reference state x,
     *        advanced in place on hand[0].
     */
    typename TimeSequence<Real>::Iterator referencePath() const {
        return sequence_.iterator(x_, hands_[0], scratchBlock(0));
    }

    /**
     * @brief A path through the sequence from the i-th Gram-Schmidt
     *        point y_i, advanced in place on hand[i].
     */
    typename TimeSequence<Real>::Iterator deviatedPath(size_t exponentIndex) const {
        return sequence_.iterator(y_.col(exponentIndex), hands_[exponentIndex], scratchBlock(exponentIndex));
    }

    /**
     * @brief The scratch block belonging to one handle.
     */
    Mat<Real> scratchBlock(size_t handleIndex) const {
        const size_t columns = sequence_.scratchColumns();
        if (columns == 0)
            return scratch_;

        return scratch_.subMat(0, handleIndex * columns, scratch_._rows, columns);
    }

    /**
     * @brief Checks the constructor's arguments and returns the number
     *        of exponents, n, which is a fifth of two less than the
     *        number of singletons.
     *
     * Runs first in the member initializer list, so a bad argument throws
     * before any of the views built from it exist.
     */
    static size_t validated(
        const TimeSequence<Real>& sequence,
        const Mat<Real>& buffer,
        const Vec<Real>& singletons,
        const Handles& hands
    ) {
        if (singletons.size() < 7 || (singletons.size() - 2) % 5 != 0)
            throw std::invalid_argument("LyapunovExponent needs 5n + 2 singletons, for n exponents.");

        const size_t numExponents = (singletons.size() - 2) / 5;

        if (hands.size() != numExponents)
            throw std::invalid_argument("LyapunovExponent needs exactly one handle per exponent.");

        if (buffer._cols != bufferColumns(sequence, numExponents))
            throw std::invalid_argument("LyapunovExponent buffer must have bufferColumns() columns.");

        return numExponents;
    }
};

#endif // LYAPUNOV_EXPONENT_H
