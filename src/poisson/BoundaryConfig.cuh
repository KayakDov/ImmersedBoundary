
#ifndef CUDABANDED_BOUNDARYCONFIG_CUH
#define CUDABANDED_BOUNDARYCONFIG_CUH
#include <functional>
#include "deviceArrays/headers/sparse/BandedMat.h"
#include "poisson/BoundaryCondition.cuh"

/**
 * @struct BoundaryConfig
 * @brief Stores boundary condition configuration for a 3D domain.
 *
 * Holds pointers to boundary condition objects for each face of the domain.
 * Faces can be null if not applicable (e.g., in 2D problems).
 */
template<typename Real>
struct BoundaryConfig : public XYZ<BoundaryPair<Real>>{

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
    ):XYZ<BoundaryPair<Real>>(
        BoundaryPair(startIsNeumann.x, endIsNeumann.x, startVal.x, endVal.x, isStaggered, delta.x, dim.cols),
        BoundaryPair(startIsNeumann.y, endIsNeumann.y, startVal.y, endVal.y, isStaggered, delta.y, dim.rows),
        BoundaryPair(startIsNeumann.z, endIsNeumann.z, startVal.z, endVal.z, isStaggered, delta.z, dim.layers)
    ){}


    /**
     * True if i is the first index that these conditions appear at.
     * @param i The index to be checked.
     * @return the value of the index repeated, or -1 if this is the first appearence.
     */
    __host__ int repeat(int i) const {
        for (size_t j = 0; j < i; ++j) if ((*this)[j] == (*this)[i]) return j;
        return -1;

    }

    /**
     * @brief Efficiently creates unique objects based on 1D Laplacian boundary conditions.
     *
     * This utility iterates through the X, Y, and Z dimensions of a Laplacian setup.
     * If the boundary conditions (dimensions, spacing, and types) for two dimensions
     * are identical, the function reuses the existing object by sharing ownership
     * via std::shared_ptr. This prevents redundant memory allocations and unnecessary
     * re-computation of Laplacian matrices or eigenvalue vectors.
     *
     * @tparam ResultType The type of object to be created (e.g., Mat<T> or Vec<T>).

     * @param[out] outputs  An array of 3 shared_ptrs to be populated with the results.
     * @param[in]  factory  A factory function: `ResultType factory(const LaplacianConditions<T>&)`.
     * * @note Because ResultType is wrapped in a shared_ptr, ResultType does not need
     * to be assignable, which is critical for classes with const data members.
     */
    template <typename ResultType>
    __host__ void createUnique(std::shared_ptr<ResultType> (&outputs)[3], std::function<ResultType(const BoundaryPair<Real>&)> factory) const {
        size_t numDim = dim().numDims();
        for (size_t i = 0; i <  numDim; ++i) {
            int repeatInd = repeat(i);
            if (repeatInd == -1) outputs[i] = std::make_shared<ResultType>(factory((*this)[i]));
            else outputs[i] = outputs[repeatInd];
        }
        if (numDim == 2) outputs[2] = nullptr;
    }


    /**
     * Generates, including memory allocation, eigen values and vectors.  The matrices pointed to, that are retruned,
     * hold the values in the last column, and the vectors in the first nxn cells.
     * @param hands3 Used to create the different vectors in parrallel.  The number of handles should be equal to the number of dimentisons.
     * @param events The number of events should be equal to the number of dimesnions.
     * @param preAllocatedForL_iX3
     * @return pointers to matrices containing the eigen values and vectors.
     */
    __host__ void generateEigen(Handle *hands3, Event *events, std::shared_ptr<Mat<Real>> (&preAllocatedForL_iX3)[3]) const{

        createUnique<Mat<Real>>(preAllocatedForL_iX3, [](const BoundaryPair<Real>& c) {
            return Mat<Real>::create(c.dimLength, c.dimLength + 1);
        });

        (*this)[0].generateEigen(hands3[0], *(preAllocatedForL_iX3[0]));

        for (size_t i = 1; i < dim().numDims(); ++i)
            if (repeat(i) < 0){
                (*this)[i].generateEigen(hands3[i], *(preAllocatedForL_iX3[i]));
                events[i - 1].record(hands3[i]);
                events[i - 1].hold(hands3[0]);
            }

    }
    /**
     * Checks if all the boundary oncdiitons are Neumann resulting in a singular laplacian.
     * @return True if all the boundary conditions are Neumann.
     */
    __host__ bool allNeumann() const {
        return this->x.bothNeumann() && this->y.bothNeumann() && this->z.bothNeumann();
    }

    /**
     *
     * @return The dimensions of the grid.
     */
    __host__ __device__ GridDim dim() const {
        return GridDim(this->y.dimLength, this->x.dimLength, this->z.dimLength);
    }

};
#endif //CUDABANDED_BOUNDARYCONFIG_CUH
