
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
 * Holds pointers to boundary condition objects for each face of the domain.
 * Faces can be null if not applicable (e.g., in 2D problems).
 */
template<typename Real>
struct BoundaryConfig : public XYZ<UniformSegment<Real>>{

    /**
     * A simplified constructor that creates uniform boundary conditions.
     * @param isNeumann
     * @param isStaggered
     * @param dim
     */
    __host__ __device__ BoundaryConfig(bool isNeumann, bool isStaggered, const GridDim& dim, const Real3d delta = Real3d(1, 1, 1)):
        BoundaryConfig(isNeumann, isNeumann,isStaggered, dim, delta){}

    /**
     * A simplified constructor that creates identical boundary conditions for each dimensions.
     * @param startIsNeumann
     * @param endIsNeumann
     * @param dim
     * @param isStaggered
     */
    __host__  BoundaryConfig( bool startIsNeumann,  const bool endIsNeumann, bool isStaggered, const GridDim& dim, const Real3d delta = Real3d(1, 1, 1)):
        BoundaryConfig(
            XYZ<bool>::fill(startIsNeumann),
            XYZ<bool>::fill(endIsNeumann),
            XYZ<Real>::fill(startIsNeumann),
            XYZ<Real>::fill(endIsNeumann),
            delta,
            dim, isStaggered)
    {}

    /**
     *
     * @param startIsNeumann The condition for the bginning of each dimesnion, true for Neumann and false for Dirichlet.
     * @param endIsNeumann The condition for the end of each dimension.
     * @param startVal The value for the condition at the beginning of each dimension: Neumann -> d/d_delta at boundary, Dirichlet -> const value for all boundary
     * @param endVal The valued for the condition at the end of each dimension.
     * @param delta The distance between grid points.
     * @param dim The shape of the grid.
     * @param isStaggered True for staggered grids, false for node centered grids.
     */
    __host__ __device__ BoundaryConfig(
        const XYZ<bool>& startIsNeumann,  const XYZ<bool>& endIsNeumann,
        const XYZ<Real>& startVal, const XYZ<Real>& endVal,
        const Real3d& delta,
        const GridDim& dim,
        bool isStaggered
    ):XYZ<UniformSegment<Real>>(
        UniformSegment<Real>(startIsNeumann.x, endIsNeumann.x, startVal.x, endVal.x, isStaggered, delta.x, dim.cols),
        UniformSegment<Real>(startIsNeumann.y, endIsNeumann.y, startVal.y, endVal.y, isStaggered, delta.y, dim.rows),
        UniformSegment<Real>(startIsNeumann.z, endIsNeumann.z, startVal.z, endVal.z, isStaggered, delta.z, dim.layers)
    ){}

    /**
     * Checks if all the boundary oncdiitons are Neumann resulting in a singular laplacian.
     * @return True if all the boundary conditions are Neumann.
     */
    __host__ bool allNeumann() const {
        return this->x.bothNeumann() && this->y.bothNeumann() && (this->z.bothNeumann() || this->z.numNodes <= 1);
    }

    /**
     *
     * @return The dimensions of the grid.
     */
    __host__ __device__ GridDim dim() const {
        return GridDim(this->y.numNodes, this->x.numNodes, this->z.numNodes);
    }

};
#endif //CUDABANDED_BOUNDARYCONFIG_CUH
