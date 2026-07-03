/**
 * @file BoundaryConfig.cuh
 * @brief Defines boundary-configuration containers for Poisson solves.
 * @ingroup poisson
 *
 * @details
 * The Poisson module assembles structured operators, boundary metadata, and solver-facing data structures for grid-based elliptic solves.
 */

#ifndef CUDABANDED_BOUNDARYCONFIG_CUH
#define CUDABANDED_BOUNDARYCONFIG_CUH

#include <functional>
#include "deviceArrays/headers/sparse/BandedMat.h"
#include "poisson/AxisSegment.cuh"
#include "solvers/Event.h"

/**
 * @struct BoundaryConfig
 * @brief Stores boundary condition configuration for a 3D domain.
 *
 * This version forces the user to explicitly define the segment type 
 * for each axis (X, Y, and Z), accommodating any mixture of uniform 
 * and variable boundary conditions.
 */
template<typename Real, typename SegX, typename SegY, typename SegZ>
struct BoundaryConfig {

    /** @brief Boundary segment handler for the X-axis */
    SegX x;

    /** @brief Boundary segment handler for the Y-axis */
    SegY y;

    /** @brief Boundary segment handler for the Z-axis */
    SegZ z;

    /**
     * @brief Explicit constructor requiring all 3 axis segments.
     * * This forces the user to instantiate the segments (Uniform or Variable)
     * beforehand and pass them in directly.
     */
    __host__ __device__ BoundaryConfig(const SegX& axisX, const SegY& axisY, const SegZ& axisZ)
        : x(axisX), y(axisY), z(axisZ) {}



    /**
     * @brief Deduces the full 3D grid dimensions from the segments themselves.
     * @return The dimensions of the grid.
     */
    __host__ __device__ GridDim dim() const {
        return GridDim(this->y.numNodes, this->x.numNodes, this->z.numNodes); // [cite: 257]
    }


    XYZ<Delta1d<Real>> delta() const {
        return {x.getDelta(), y.getDelta(), z.getDelta()};
    }

    /**
     * Checks if all the boundary conditions are Neumann, which results in a singular Laplacian.
     * @return True if all axes have Neumann conditions on both ends.
     */
    bool allNeumann() const {
        return this->x.bothNeumann() &&
               this->y.bothNeumann() &&
               (this->z.bothNeumann() || this->z.numNodes <= 1); // [cite: 254]
    }
};

template<typename Real, typename SegX, typename SegY, typename SegZ>
class BoundaryConfigHost {

public:
    const AxisSegmentHost<SegX> x;
    const AxisSegmentHost<SegY> y;
    const AxisSegmentHost<SegZ> z;

    BoundaryConfigHost(const AxisSegmentHost<SegX>& axisX, const AxisSegmentHost<SegY>& axisY, const AxisSegmentHost<SegZ>& axisZ)
        : x(axisX), y(axisY), z(axisZ) {}

    // Deterministic return type! Flawless parameter passing for kernels.
    auto forDevice() const {
        return BoundaryConfig<Real, SegX, SegY, SegZ>(x.forDevice(), y.forDevice(), z.forDevice());
    }
};

/**
 * @brief A factory that deduces boundary types at runtime and injects the
 * strongly-typed BoundaryConfig into a provided callback.
 * * @param dim The dimensions of the grid (rows=Y, cols=X, layers=Z).
 * @param deltas XYZ struct containing the delta vectors for each axis.
 * @param startIsNeumann XYZ struct of Neumann flags for the start boundaries.
 * @param endIsNeumann XYZ struct of Neumann flags for the end boundaries.
 * @param startVal XYZ struct of boundary values for the start boundaries.
 * @param endVal XYZ struct of boundary values for the end boundaries.
 * @param isStaggered True if using a staggered grid discretization.
 * @param stream CUDA stream used for asynchronous GPU allocations.
 * @param launchParams The lambda callback to execute once types are deduced.  This is a lambda expression that takes in a single boundaryConfig, and does something with it.
 */
template<typename Real, typename Callback>
void buildBoundaryConfigAndLaunch(
    const GridDim& dim,
    const XYZ<std::vector<Real>>& deltas,
    const XYZ<bool>& startIsNeumann,
    const XYZ<bool>& endIsNeumann,
    const XYZ<Real>& startVal,
    const XYZ<Real>& endVal,
    bool isStaggered,
    cudaStream_t stream,
    Callback&& launchParams
) {
    auto dispatchZ = [&](const auto& segHostX, const auto& segHostY) {
        using SegX = typename std::decay_t<decltype(segHostX)>::SegmentType;
        using SegY = typename std::decay_t<decltype(segHostY)>::SegmentType;
        if (deltas.z.size() == 1) {
            AxisSegmentHost<UniformSegment<Real>> segHostZ(
                {startVal.z, startIsNeumann.z},
                {endVal.z, endIsNeumann.z},
                isStaggered,
                deltas.z[0],
                dim.layers
            );
            launchParams(
                BoundaryConfigHost<Real, SegX, SegY, UniformSegment<Real>>(segHostX, segHostY, segHostZ)
            );
        } else {
            SimpleArray<Real> arrayZ = SimpleArray<Real>::create(deltas.z, stream);
            AxisSegmentHost<VariableSegment<Real>> segHostZ(
                {startVal.z, startIsNeumann.z},
                {endVal.z, endIsNeumann.z},
                arrayZ
            );
            launchParams(
                BoundaryConfigHost<Real, SegX, SegY, VariableSegment<Real>>(segHostX, segHostY, segHostZ)
            );
        }
    };

    auto dispatchY = [&](const auto& segHostX) {
        if (deltas.y.size() == 1) {
            AxisSegmentHost<UniformSegment<Real>> segHostY(
                {startVal.y, startIsNeumann.y},
                {endVal.y, endIsNeumann.y},
                isStaggered,
                deltas.y[0],
                dim.rows
            );
            dispatchZ(segHostX, segHostY);
        } else {
            SimpleArray<Real> arrayY = SimpleArray<Real>::create(deltas.y, stream);
            AxisSegmentHost<VariableSegment<Real>> segHostY(
                {startVal.y, startIsNeumann.y},
                {endVal.y, endIsNeumann.y},
                arrayY
            );
            dispatchZ(segHostX, segHostY);
        }
    };

    if (deltas.x.size() == 1) {
        AxisSegmentHost<UniformSegment<Real>> segHostX(
                {startVal.x, startIsNeumann.x},
                {endVal.x, endIsNeumann.x},
                isStaggered,
                deltas.x[0],
                dim.cols
            );
        dispatchY(segHostX);
    } else {
        SimpleArray<Real> arrayX = SimpleArray<Real>::create(deltas.x, stream);
        AxisSegmentHost<VariableSegment<Real>> segHostX(
                {startVal.x, startIsNeumann.x},
                {endVal.x, endIsNeumann.x},
                arrayX
            );
        dispatchY(segHostX);
    }
}

/**
     * @brief Backwards compatible constructor for uniform-everywhere configurations.
     *
     * This allows you to keep using the old initialization logic. It initializes
     * x, y, and z axes using the uniform parameters provided.
     */
template<typename T>
static auto makeUniformBoundaryConfigHost(
    const XYZ<bool>& startIsNeumann, const XYZ<bool>& endIsNeumann,
    const XYZ<T>& startVal, const XYZ<T>& endVal,
    const Real3d& delta, const GridDim& dim, bool isStaggered
) {
    return BoundaryConfigHost<T, UniformSegment<T>, UniformSegment<T>, UniformSegment<T>>(
        AxisSegmentHost<UniformSegment<T>>({startVal.x, startIsNeumann.x}, {endVal.x, endIsNeumann.x}, isStaggered, delta.x, dim.cols),
        AxisSegmentHost<UniformSegment<T>>({startVal.y, startIsNeumann.y}, {endVal.y, endIsNeumann.y}, isStaggered, delta.y, dim.rows),
        AxisSegmentHost<UniformSegment<T>>({startVal.z, startIsNeumann.z}, {endVal.z, endIsNeumann.z}, isStaggered, delta.z, dim.layers)
    );
}



#ifndef INSTANTIATION_MACROS_H
#define INSTANTIATION_MACROS_H

// Applies a given MACRO_NAME to all 8 segment combinations for a specific Real type
#define APPLY_TO_ALL_SEGMENT_COMBOS(Real, MACRO_NAME) \
MACRO_NAME(Real, UniformSegment<Real>,  UniformSegment<Real>,  UniformSegment<Real>)  \
MACRO_NAME(Real, UniformSegment<Real>,  UniformSegment<Real>,  VariableSegment<Real>) \
MACRO_NAME(Real, UniformSegment<Real>,  VariableSegment<Real>, UniformSegment<Real>)  \
MACRO_NAME(Real, UniformSegment<Real>,  VariableSegment<Real>, VariableSegment<Real>) \
MACRO_NAME(Real, VariableSegment<Real>, UniformSegment<Real>,  UniformSegment<Real>)  \
MACRO_NAME(Real, VariableSegment<Real>, UniformSegment<Real>,  VariableSegment<Real>) \
MACRO_NAME(Real, VariableSegment<Real>, VariableSegment<Real>, UniformSegment<Real>)  \
MACRO_NAME(Real, VariableSegment<Real>, VariableSegment<Real>, VariableSegment<Real>)

#endif // INSTANTIATION_MACROS_H
#endif // CUDABANDED_BOUNDARYCONFIG_CUH