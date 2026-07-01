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
     * Checks if all the boundary conditions are Neumann, which results in a singular Laplacian.
     * @return True if all axes have Neumann conditions on both ends.
     */
    __host__ bool allNeumann() const {
        return this->x.bothNeumann() && 
               this->y.bothNeumann() && 
               (this->z.bothNeumann() || this->z.numNodes <= 1); // [cite: 254]
    }

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
    auto dispatchZ = [&](const auto& segX, const auto& segY) {
        if (deltas.z.size() == 1) {
            UniformSegment<Real> segZ(startIsNeumann.z, endIsNeumann.z, startVal.z, endVal.z, isStaggered, deltas.z[0], dim.layers);
            launchParams(BoundaryConfig<Real, std::decay_t<decltype(segX)>, std::decay_t<decltype(segY)>, UniformSegment<Real>>(segX, segY, segZ));
        } else {
            SimpleArray<Real> arrayZ = SimpleArray<Real>::create(deltas.z, stream);
            VariableSegment<Real> segZ(startIsNeumann.z, endIsNeumann.z, startVal.z, endVal.z, arrayZ);
            launchParams(BoundaryConfig<Real, std::decay_t<decltype(segX)>, std::decay_t<decltype(segY)>, VariableSegment<Real>>(segX, segY, segZ));
        }
    };

    auto dispatchY = [&](const auto& segX) {
        if (deltas.y.size() == 1) {
            UniformSegment<Real> segY(startIsNeumann.y, endIsNeumann.y, startVal.y, endVal.y, isStaggered, deltas.y[0], dim.rows);
            dispatchZ(segX, segY);
        } else {
            SimpleArray<Real> arrayY = SimpleArray<Real>::create(deltas.y, stream);
            VariableSegment<Real> segY(startIsNeumann.y, endIsNeumann.y, startVal.y, endVal.y, arrayY);
            dispatchZ(segX, segY);
        }
    };

    if (deltas.x.size() == 1) {
        UniformSegment<Real> segX(startIsNeumann.x, endIsNeumann.x, startVal.x, endVal.x, isStaggered, deltas.x[0], dim.cols);
        dispatchY(segX);
    } else {
        SimpleArray<Real> arrayX = SimpleArray<Real>::create(deltas.x, stream);
        VariableSegment<Real> segX(startIsNeumann.x, endIsNeumann.x, startVal.x, endVal.x, arrayX);
        dispatchY(segX);
    }
}

/**
     * @brief Backwards compatible constructor for uniform-everywhere configurations.
     *
     * This allows you to keep using the old initialization logic. It initializes
     * x, y, and z axes using the uniform parameters provided.
     */
template<typename T>
static auto makeUniformBoundaryConfig(
    const XYZ<bool>& startIsNeumann, const XYZ<bool>& endIsNeumann,
    const XYZ<T>& startVal, const XYZ<T>& endVal,
    const Real3d& delta, const GridDim& dim, bool isStaggered
) {
    return BoundaryConfig<T, UniformSegment<T>, UniformSegment<T>, UniformSegment<T>>(
        UniformSegment<T>(startIsNeumann.x, endIsNeumann.x, startVal.x, endVal.x, isStaggered, delta.x, dim.cols),
        UniformSegment<T>(startIsNeumann.y, endIsNeumann.y, startVal.y, endVal.y, isStaggered, delta.y, dim.rows),
        UniformSegment<T>(startIsNeumann.z, endIsNeumann.z, startVal.z, endVal.z, isStaggered, delta.z, dim.layers)
    );
}
#endif // CUDABANDED_BOUNDARYCONFIG_CUH