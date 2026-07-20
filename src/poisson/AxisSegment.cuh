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

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include "deviceArrays/headers/SimpleArray.h"
#include "poisson/BoundaryCondition.cuh"

/**
 * @brief Host-side reconstruction of finite-volume cell widths from deltas.
 *
 * For a FluxLapl axis with n unknowns, the n+1 deltas are
 * {wall-to-centre_0, centre-to-centre, ..., centre_{n-1}-to-wall}, and the
 * n cell widths satisfy d_0 = W_0/2, d_i = (W_{i-1}+W_i)/2, d_n = W_{n-1}/2.
 * Solving front-to-back gives the recursion below; the leftover equation
 * W_{n-1} == 2 d_n is a consistency check that the deltas really came from
 * cell centres of cells tiling the segment wall-to-wall.
 *
 * @throws std::invalid_argument if the deltas are inconsistent with a
 *         wall-to-wall cell tiling, or any reconstructed width is <= 0.
 */
template<typename Real>
std::vector<Real> makeFvmWidths(const std::vector<Real>& d) {
    if (d.size() < 2) throw std::invalid_argument("FluxLapl axis needs at least 2 deltas (1 unknown).");
    const size_t n = d.size() - 1;
    std::vector<Real> w(n);
    w[0] = 2*d[0];
    for (size_t i = 1; i < n; ++i) w[i] = 2*d[i] - w[i-1];

    Real scale = 0;
    for (size_t i = 0; i <= n; ++i) scale = std::max(scale, std::abs(d[i]));
    if (std::abs(w[n-1] - 2*d[n]) > Real(1e-10) * scale)
        throw std::invalid_argument(
            "FluxLapl deltas are inconsistent: the reconstructed last cell width "
            "does not equal twice the last delta. The n+1 deltas must be the "
            "wall/centre distances of n cells tiling the segment wall-to-wall.");
    for (size_t i = 0; i < n; ++i)
        if (!(w[i] > 0)) throw std::invalid_argument("FluxLapl reconstructed a non-positive cell width.");
    return w;
}

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

    /**
     * Diagonal of D^2 in the symmetrizing similarity S = -D L D^{-1}
     * used by the eigendecomposition (Eigen.cu).  L = M^{-1} K with K
     * symmetric (K offdiag = 1/delta) and M_i proportional to this value,
     * so D = sqrt(symmScale) makes S symmetric.  Any positive constant
     * factor cancels; this keeps the historical convention d_i + d_{i+1}.
     */
    __device__ T symmScale(const size_t i) const {
        return delta[i] + delta[i + 1];
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
 * @class FluxLaplacian
 * @brief Conservative finite-volume Laplacian axis with variable spacing.
 *
 * Unknowns are cell averages at the centres of n cells that tile the segment
 * wall-to-wall.  Interior row i is the flux difference over cell i:
 *     ( (u_{i+1}-u_i)/d_{i+1} - (u_i-u_{i-1})/d_i ) / W_i
 * Summing W_i * row_i telescopes interior fluxes, so the operator is
 * discretely conservative and equals div(grad) built from the same faces --
 * the property required inside projection methods.  Contrast VariableSegment,
 * which is the pointwise 3-point formula at nodes (faces implicitly at
 * midpoints); the two coincide on locally uniform interiors and differ on
 * stretched meshes and at wall-adjacent rows.
 */
template<typename T>
class FluxLaplacian : public AxisSegment<T, FluxBoundary<T, false>, FluxBoundary<T, true>> {
public:
    /** n+1 centre/wall distances (same layout as VariableSegment's delta). */
    const DeviceData1d<T> delta;

    /** n reconstructed cell widths (see makeFvmWidths). */
    const DeviceData1d<T> width;

    FluxLaplacian(bool beginIsNeumann, bool endIsNeumann, T beginVal, T endVal,
                  const SimpleArray<T>& delta, const SimpleArray<T>& width) :
        AxisSegment<T, FluxBoundary<T, false>, FluxBoundary<T, true>>(
            delta.size() - 1,
            FluxBoundary<T, false>(beginIsNeumann, beginVal, delta.subArray(0, 2), width.subArray(0, 1)),
            FluxBoundary<T, true>(endIsNeumann, endVal, delta.subArray(delta.size() - 2, 2), width.subArray(width.size() - 1, 1))
        ),
        delta(delta),
        width(width) {}

    __device__ void setInteriorL(T& mainDiagVal, T& prevVal, T& nextVal, const size_t indexInLine) const {

        T W = width[indexInLine];
        prevVal = 1 / (W * delta[indexInLine]);
        nextVal = 1 / (W * delta[indexInLine + 1]);
        mainDiagVal -= prevVal + nextVal;

    }

    /**
     * Diagonal of D^2 in the symmetrizing similarity S = -D L D^{-1}
     * (see VariableSegment::symmScale).  The flux operator is
     * L = W^{-1} K with the SAME symmetric K, so the scale is the cell
     * width; the factor 2 keeps the same convention as VariableSegment
     * (whose scale equals twice the midpoint control volume) and cancels
     * in the similarity anyway.
     */
    __device__ T symmScale(const size_t i) const {
        return 2 * width[i];
    }

    Delta1d<T> getDelta() const{
        return Delta1d<T>(false, delta);
    }

    __host__ __device__ friend bool operator==(const FluxLaplacian& lhs,
                                               const FluxLaplacian& rhs) {
        return (lhs.start == rhs.start) && (lhs.end == rhs.end) && (lhs.numNodes == rhs.numNodes)
            && lhs.delta.data == rhs.delta.data && lhs.width.data == rhs.width.data;
    }

    __host__ __device__ friend bool operator!=(const FluxLaplacian& lhs,
                                               const FluxLaplacian& rhs) {
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



// --- SPECIALIZATION FOR FLUX (FINITE-VOLUME) SEGMENTS ---
template<typename Real>
class AxisSegmentHost<FluxLaplacian<Real>> {

    BoundaryConditionHost<Real> start, end;
    SimpleArray<Real> varDelta; // Manages host lifetime / prevents premature cudaFree!
    SimpleArray<Real> fvWidth;  // Reconstructed cell widths; same lifetime rules.

public:

    using SegmentType = FluxLaplacian<Real>;

    /**
     * Builds the widths itself from the host deltas, so makeFvmWidths (and
     * its consistency checks) run exactly where the FluxLapl concept lives.
     */
    AxisSegmentHost(BoundaryConditionHost<Real> start, BoundaryConditionHost<Real> end,
                    const std::vector<Real>& hostDelta, cudaStream_t stream)
        : start(start), end(end),
          varDelta(SimpleArray<Real>::create(hostDelta, stream)),
          fvWidth(SimpleArray<Real>::create(makeFvmWidths(hostDelta), stream)) {}

    FluxLaplacian<Real> forDevice() const {
        return FluxLaplacian<Real>(start.isNeumann, end.isNeumann,
                                    start.value, end.value,
                                    varDelta, fvWidth); // Implicitly decay to DeviceData1d
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

template<typename T>
__host__ __device__
bool operator==(const FluxLaplacian<T>&, const UniformSegment<T>&) { return false; }

template<typename T>
__host__ __device__
bool operator==(const UniformSegment<T>&, const FluxLaplacian<T>&) { return false; }

template<typename T>
__host__ __device__
bool operator==(const FluxLaplacian<T>&, const VariableSegment<T>&) { return false; }

template<typename T>
__host__ __device__
bool operator==(const VariableSegment<T>&, const FluxLaplacian<T>&) { return false; }

template<typename T>
__host__ __device__
bool operator!=(const FluxLaplacian<T>& lhs, const UniformSegment<T>& rhs) { return !(lhs == rhs); }

template<typename T>
__host__ __device__
bool operator!=(const UniformSegment<T>& lhs, const FluxLaplacian<T>& rhs) { return !(lhs == rhs); }

template<typename T>
__host__ __device__
bool operator!=(const FluxLaplacian<T>& lhs, const VariableSegment<T>& rhs) { return !(lhs == rhs); }

template<typename T>
__host__ __device__
bool operator!=(const VariableSegment<T>& lhs, const FluxLaplacian<T>& rhs) { return !(lhs == rhs); }


#endif //CUDABANDED_AXISSEGMENT_CUH