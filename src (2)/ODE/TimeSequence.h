/**
 * @file TimeSequence.h
 * @brief Abstract GPU state sequence: a stepping rule, plus iterators that walk paths through it.
 */

#ifndef CUDABANDED_TIME_SEQUENCE_H
#define CUDABANDED_TIME_SEQUENCE_H

#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/Vec.h"

template<typename Real> class LyapunovExponent;

/**
 * @brief A stepping rule for a system whose state lives on the GPU.
 *
 * A derived class implements nextPoint(), for example with RK4. The
 * sequence itself holds no path, no stream and no scratch memory, so any
 * number of paths can walk the same system at once, each on its own
 * Handle, through iterator().
 *
 * Anything a step needs besides the state -- stage vectors for RK4, for
 * instance -- is supplied by the caller as scratch, because scratch
 * shared between paths on different streams would be written
 * concurrently. scratchColumns() says how many state-length columns a
 * path needs.
 *
 * @tparam Real State element type.
 */
template<typename Real>
class TimeSequence {
public:

    /**
     * @brief One path through a TimeSequence, advanced on one Handle.
     *
     * Holds the current point, the handle every step is enqueued on, and
     * the scratch those steps use. Stepping is in place: point() always
     * refers to the same storage, holding the latest state.
     *
     * Copying an Iterator copies the references, not the path: the copy
     * steps the same point on the same stream.
     */
    class Iterator {
    public:
        /**
         * @param sequence The system being walked; must outlive this.
         * @param point    The path's current state, advanced in place.
         * @param hand     Handle whose stream every step runs on; must
         *                 outlive this.
         * @param scratch  sequence.scratchColumns() columns, each as long
         *                 as the state, used by every step of this path
         *                 and by no other path.
         */
        Iterator(const TimeSequence& sequence, Vec<Real> point, Handle& hand, Mat<Real> scratch)
            : sequence_(sequence), point_(point), hand_(hand), scratch_(scratch) {}

        /** Advances the path by one step. Enqueued, not synchronized. */
        void next() {
            sequence_.nextPoint(point_, point_, hand_, scratch_);
        }

        /** Advances the path by @p steps steps. Enqueued, not synchronized. */
        void advance(size_t steps) {
            for (size_t stepIndex = 0; stepIndex < steps; ++stepIndex)
                next();
        }

        /** The path's current state, as a shallow view of its storage. */
        Vec<Real> point() const { return point_; }

        /** The handle this path runs on. */
        Handle& handle() const { return hand_; }

    private:
        const TimeSequence& sequence_;
        Vec<Real> point_;
        Handle& hand_;
        Mat<Real> scratch_;
    };

    virtual ~TimeSequence() = default;

    /**
     * @brief How many state-length scratch columns one path needs.
     *
     * Zero by default. ODE needs three for its RK4 stages.
     */
    virtual size_t scratchColumns() const { return 0; }

    /**
     * @brief Computes one successor state.
     * @param currentPoint Input state; must not be modified.
     * @param nextPoint    Destination, completely overwritten.
     * @param hand         Handle whose stream all work is enqueued on.
     * @param scratch      scratchColumns() columns of workspace, used by
     *                     this call alone for its duration.
     *
     * @pre currentPoint and nextPoint have equal lengths, and may be the
     *      same storage: Iterator steps in place, so every
     *      implementation must support that.
     * @note No synchronization is performed.
     */
    virtual void nextPoint(
        const Vec<Real>& currentPoint,
        Vec<Real> nextPoint,
        Handle& hand,
        Mat<Real> scratch
    ) const = 0;

    /**
     * @brief Starts a path at @p startingPoint, to be advanced on @p hand.
     * @param startingPoint The path's state; advanced in place, so the
     *                      caller loses its original contents.
     * @param hand          Handle the path runs on.
     * @param scratch       See Iterator.
     */
    Iterator iterator(Vec<Real> startingPoint, Handle& hand, Mat<Real> scratch) const {
        return Iterator(*this, startingPoint, hand, scratch);
    }

    /**
     * @brief Fills a trajectory beginning with the state in column zero.
     * @param points  Matrix containing one state per column.
     * @param hand    Handle the steps run on.
     * @param scratch See Iterator.
     *
     * Column zero is preserved. Each subsequent column contains the
     * successor of the previous column. Zero or one column requires no
     * stepping.
     */
    void trajectory(Mat<Real>& points, Handle& hand, Mat<Real> scratch) const {
        for (size_t col = 1; col < points._cols; ++col) {
            auto current = points.col(col - 1);
            auto destination = points.col(col);
            nextPoint(current, destination, hand, scratch);
        }
    }

    /**
     * @brief Computes the state after t steps without retaining a trajectory.
     * @param t            Number of steps; zero copies the initial state.
     * @param initialPoint Starting state, preserved.
     * @param result       Caller-owned destination for the final state.
     * @param hand         Handle the steps run on.
     * @param scratch      See Iterator.
     *
     * @pre initialPoint and result have equal lengths and disjoint storage.
     */
    void pointAt(
        size_t t,
        const Vec<Real>& initialPoint,
        Vec<Real> result,
        Handle& hand,
        Mat<Real> scratch
    ) const {
        result.set(initialPoint, hand);
        iterator(result, hand, scratch).advance(t);
    }

protected:

    /**
     * @brief The one external collaborator allowed to reach projection().
     *
     * Nothing else gets this access; projection() stays non-public for
     * everyone else.
     */
    friend class LyapunovExponent<Real>;

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
     * @param v    Deviation vector to project, modified in place.
     * @param hand Handle whose stream the work is enqueued on.
     */
    virtual void projection(Vec<Real>& v, Handle& hand) const {}

};

#endif // CUDABANDED_TIME_SEQUENCE_H
