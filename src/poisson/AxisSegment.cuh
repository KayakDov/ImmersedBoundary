/**
 * @file AxisSegment.cuh
 * @brief Defines axis-aligned segment helpers for Poisson boundary logic.
 * @ingroup poisson
 *
 * @details
 * The Poisson module assembles structured operators, boundary metadata, and solver-facing data structures for grid-based elliptic solves.
 */

#ifndef CUDABANDED_AXISSEGMENT_CUH
#define CUDABANDED_AXISSEGMENT_CUH

#include "deviceArrays/headers/SimpleArray.h"
#include "poisson/BoundaryCondition.cuh"

template<typename Real>
struct Delta1d {
    const DeviceData1d<Real> deltaVar;
    const double val;

    /**
     *
     * @param delta set to 0 if the variable delta will be used.  If the distance between nodes is uniform, then this is that distance.
     * @param deltaVar set to empty set if the distance between nodes is uniform.  If the distance is variable, this will be used.
     * The first value is the distance from the first boundary to the first node.  If there are n nodes, then the n + 1 value is
     * the distance from the last node to the end boundary.
     */
    __device__ Delta1d(Real delta, const DeviceData1d<Real>& deltaVar)
    : deltaVar(deltaVar), val(delta) {}

    __device__ Real operator[](size_t i) const{
        return val > 0 ? val : deltaVar[i];
    }
};


template<typename Real, typename SpacingTypeStart, typename SpacingTypeEnd>
class AxisSegment {
public:
    /**
     * The number of nodes in this dimension.
     */
    size_t numNodes;

    /** @brief The condition applied at the start (lower index) of the segment. */
    const SpacingTypeStart start;

    /** @brief The condition applied at the end (higher index) of the segment. */
    const SpacingTypeEnd end;

    /**
     * returns the start or the end.
     * @param isEnd true if you want the end boundary condition, false if you want the start.
     * @return the desired boundary condition.
     */
    __host__ __device__ const BoundaryCondition<Real>& operator[](bool isEnd) const {
        return isEnd ? end : start;
    }

    AxisSegment(size_t numNodes, const SpacingTypeStart& start, const SpacingTypeEnd& end) :
        numNodes(numNodes),
        start(start),
        end(end){}

    /**
     * @brief Apply boundary condition contribution to system.
     *
     * Modifies the matrix L for a grid point adjacent to the boundary.
     * @return true if a modification is made, false otherwise.
     */
    __device__ bool setBoundaryL(Real& mainDiagVal, Real& leftDiagonal, Real& rightDiagonal, const size_t indexInLine) const {
        if (indexInLine == 0) this->start.setL(mainDiagVal, rightDiagonal);
        else if (indexInLine == this->numNodes - 1) this->end.setL(mainDiagVal, leftDiagonal);
        else return false;
        return true;
    }


    /**
     * @brief Apply boundary condition contribution to RHS only.
     *
     * Adds the boundary-condition-induced contribution to the right-hand side
     * vector, assuming the operator (L) is assembled independently.
     *
     * @param rhs Right-hand side vector (modisfied in place)
     * @param indexInLine  The index of this point in this dimension.
     * @return true if a modification is made, false otherwise.
     */
    __device__ bool setBoundaryRHS1d(DeviceData1d<Real>& rhs, const size_t indexInLine) const {
        if (indexInLine == 0) this->start.setBoundaryRHS(rhs[0]);
        else if (indexInLine == this->numNodes - 1) this->end.setBoundaryRHS(rhs[this->numNodes - 1]);
        else return false;
        return true;
    }

    /**
     * @brief Apply boundary condition contribution to system.
     *
     * Modifies the matrix L and RHS vector b for a grid point adjacent to the boundary.
     * @return true if a modification is made, false otherwise.
     * @return true if a modification is made, false otherwise.
     */
    __device__ bool setLAndRHS(Real& mainDiagVal, Real& startDiagVal, Real& endDiagonalVal, DeviceData1d<Real> rhs, const size_t indexInLine) const {
        setBoundaryL(mainDiagVal, startDiagVal, endDiagonalVal, indexInLine);
        return setBoundaryRHS1d(rhs, indexInLine);
    }


    __host__ __device__ bool isUndefined() const {
        return numNodes == static_cast<size_t>(0);
    }


    /**
     * Are both conditions Neumann resulting in singular 1d matrix?
     * @return true if both conditions are neumann.
     */
    __host__ bool bothNeumann() const {
        return start.isNeumann && end.isNeumann;
    }

};

template<typename T>
class VariableSegment : public AxisSegment<T, VariableBoundary<T, false>, VariableBoundary<T, true>> {
public:
    const DeviceData1d<T> delta;

    VariableSegment(bool beginIsNeumann, bool endIsNeumann, T beginVal, T endVal, const SimpleArray<T>& delta) :
        AxisSegment<T, VariableBoundary<T, false>, VariableBoundary<T, true>>(
            delta.size() - 1,
            VariableBoundary<T, false>(beginIsNeumann, beginVal, delta.subArray(0, 2)),
            VariableBoundary<T, true>(endIsNeumann, endVal, delta.subArray(delta.size() - 2, 2))
        ),
        delta(delta) {}

    __device__ void setInteriorL(T& mainDiagVal, T& prevVal, T& nextVal, const size_t indexInLine) const {

        T distTerm = 2/(delta[indexInLine] + delta[indexInLine + 1]);
        prevVal = distTerm/delta[indexInLine];
        nextVal = distTerm/delta[indexInLine + 1];
        mainDiagVal -= 2 / (delta[indexInLine + 1] * delta[indexInLine]);

    }

    Delta1d<T> getDelta() const{
        return Delta1d<T>(false, delta);
    }

    __host__ __device__ friend bool operator==(const VariableSegment& lhs,
                                               const VariableSegment& rhs) {
        return (lhs.start == rhs.start) && (lhs.end == rhs.end) && (lhs.numNodes == rhs.numNodes) && lhs.delta.data == rhs.delta.data;
    }

    __host__ __device__ friend bool operator!=(const VariableSegment& lhs,
                                               const VariableSegment& rhs) {
        return !(lhs == rhs);
    }
};

/**
 * @class UniformSegment
 * @brief Encapsulates the boundary conditions for both ends of a 1D grid segment.
 *
 * This class stores the physical and numerical constraints for the start and end
 * of a dimension (e.g., Left/Right, Top/Bottom). By storing these by value,
 * the pair maintains its own valid copy of the conditions, preventing dangling
 * references during kernel execution.
 * * @tparam T The floating-point type (float or double).
 */
template<typename T>
class UniformSegment : public AxisSegment<T, UniformBoundary<T>, UniformBoundary<T>> {
public:
    const T inverseDeltaSq;
    const T delta;

    /**
    * @brief Construct by providing raw parameters for both boundaries.
     * This constructor initializes the internal BoundaryCondition objects directly.
     *
     * @param beginIsNeumann Is the beggin condition Neuman.  Set to false for Dirichlet.
     * @param endIsNeumann Is the end condition Neumann.  Set to false for Dirichlet.
     * @param beginVal The value at the beginning condition.
     * @param endVal The value at the end condition.
     * @param isStaggered True if the grid is staggered, false if it's node centered.
     * @param delta The distance between grid points.
     * @param numNodes The numver of grid points in this dimension.
     */
    __device__ __host__ UniformSegment(
        bool beginIsNeumann,
        bool endIsNeumann,
        T beginVal,
        T endVal,
        bool isStaggered,
        T delta,
        size_t numNodes
    ) :
        AxisSegment<T, UniformBoundary<T>, UniformBoundary<T>>(
            numNodes,
            UniformBoundary<T>(beginVal, delta, beginIsNeumann, isStaggered),
            UniformBoundary<T>((endIsNeumann ? -endVal : endVal), delta, endIsNeumann, isStaggered)
        ),
        inverseDeltaSq(1/(delta * delta)),
        delta(delta)
    {}

    /**
     * @brief Equality operator for condition pairs.
     * Required for deduplication in spectral or multi-dimensional setups.
     */
    __host__ __device__ friend bool operator==(const UniformSegment& lhs,
                                               const UniformSegment& rhs) {
        return (lhs.start == rhs.start) && (lhs.end == rhs.end) && (lhs.numNodes == rhs.numNodes);
    }

    __host__ __device__ friend bool operator!=(const UniformSegment& lhs,
                                               const UniformSegment& rhs) {
        return !(lhs == rhs);
    }

    __device__ void setInteriorL(T& mainDiagVal, T& prevVal, T& nextVal, const size_t indexInAxisSegment) const {
        prevVal = nextVal = inverseDeltaSq;
        mainDiagVal -= 2 * inverseDeltaSq;
    }

    Delta1d<T> getDelta() const {
        return Delta1d<T>(delta, SimpleArray<T>::empty());
    }
};

// Base declaration
template<typename SegmentType>
class AxisSegmentHost;

// --- SPECIALIZATION FOR UNIFORM SEGMENTS ---
template<typename Real>
class AxisSegmentHost<UniformSegment<Real>> {

    BoundaryConditionHost<Real> start, end;
    bool isStaggered;
    Real delta;
    size_t numNodes;

public:
    using SegmentType = UniformSegment<Real>;

    AxisSegmentHost(BoundaryConditionHost<Real> start, BoundaryConditionHost<Real> end, bool isStaggered, Real delta, size_t numNodes)
        : start(start), end(end), isStaggered(isStaggered), delta(delta), numNodes(numNodes) {}

    UniformSegment<Real> forDevice() const {
        return UniformSegment<Real>(start.isNeumann, end.isNeumann,
                                     start.value, end.value,
                                     isStaggered, delta, numNodes);
    }
};

// --- SPECIALIZATION FOR VARIABLE SEGMENTS ---
template<typename Real>
class AxisSegmentHost<VariableSegment<Real>> {


    BoundaryConditionHost<Real> start, end;
    SimpleArray<Real> varDelta; // Manages host lifetime / prevents premature cudaFree!

public:

    using SegmentType = VariableSegment<Real>;

    AxisSegmentHost(BoundaryConditionHost<Real> start, BoundaryConditionHost<Real> end, SimpleArray<Real> deviceArray)
        : start(start), end(end), varDelta(deviceArray) {}

    VariableSegment<Real> forDevice() const {
        return VariableSegment<Real>(start.isNeumann, end.isNeumann,
                                      start.value, end.value,
                                      varDelta); // Implicitly decays to DeviceData1d
    }
};





template<typename T>
__host__ __device__
bool operator==(const UniformSegment<T>&,
                const VariableSegment<T>&) {
    return false;
}

template<typename T>
__host__ __device__
bool operator==(const VariableSegment<T>&,
                const UniformSegment<T>&) {
    return false;
}

template<typename T>
__host__ __device__
bool operator!=(const UniformSegment<T>& lhs,
                const VariableSegment<T>& rhs) {
    return !(lhs == rhs);
}

template<typename T>
__host__ __device__
bool operator!=(const VariableSegment<T>& lhs,
                const UniformSegment<T>& rhs) {
    return !(lhs == rhs);
}


#endif //CUDABANDED_AXISSEGMENT_CUH
