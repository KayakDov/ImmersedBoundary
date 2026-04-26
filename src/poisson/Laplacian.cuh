
#ifndef CUDABANDED_POISSONLHS_H
#define CUDABANDED_POISSONLHS_H
#include <array>
#include <vector>

#include "../deviceArrays/headers/sparse/BandedMat.h"
#include "math/Real3d.h"
#include "deviceArrays/headers/SquareMat.h"
#include "math/XYZ.h"
#include "poisson/BoundaryCondition.cuh"
#include "solvers/Event.h"
#include "poisson//LaplacianKernels.cuh"

constexpr size_t numDiagonals3d = 7;
constexpr size_t numDiagonals2d = 5;


/**
 * The one dimensional Laplacians for a 2 or 3d grid.  Note, if a pair of Laplacians have the same dimension size and
 * boundary conditions, then the pointers may point to the same memory.
 * @tparam T
 */
template<typename T>
class Laplacian1d {
    Vec<int32_t> inds;
    const BoundaryConfig<T> boundary;

public:
    const XYZ<Mat<T>> rawBanded;
    /**
     *
     * @param boundary The boundary for the 1d laplacians.
     * @param hand The context.
     */
    Laplacian1d(const BoundaryConfig<T> &boundary, Handle &hand);
    /**
     * Selects one of the laplacian 1d operators.
     * @param dim 0 for the x dimension, 1 for the y dimesnion, and 2 for the z dimension.
     * @return a 1d operator.
     */
    BandedMat<T> banded(size_t dim);

    /**
     * The square matrix of the given dimension.
     * @param dim 0 for x, 1 for y, 2 for z.
     * @param hand
     * @return A square 1d laplacian matrix.
     */
    SquareMat<T> dense(size_t dim, Handle &hand);
};



template<typename T> class LaplacianEigen;

template<typename T>
class LaplacianEigen {
    LaplacianEigen(const XYZ<Vec<T>>& vals, const XYZ<SquareMat<T>>& vecs);
public:
    const XYZ<Vec<T>> vals;
    const XYZ<SquareMat<T>> vecs;

    /**
     * Generates the eigenvector matrices.
     * @param boundary The boundary conditions.
     * @param hands A handle for each dimension.
     * @param events an event for each dimension - 1
     * @param hands
     * @return The Laplacian's Eigen vector matrices.
     */
    static LaplacianEigen make(const BoundaryConfig<T> &boundary, Handle *hands, Event *events);

    GridDim dim() const;
};

template<typename T>
class Laplacian {
protected:
    const AdjacencyPatern adjacencys;
    const GridDim dim;
    const Real3d delta;
    const BoundaryConfig<T> boundary;

    std::unique_ptr<BandedMat<T>> bandedL = nullptr;
    std::unique_ptr<Vec<T>> rhsBC = nullptr;

public:

    /**
     * Creates the LHS matrix of the linear system used for solving the Poisson equation.
     * @param dim The dimensions of the Poisson grid.
     * @param delta Distance between grid points.
     * @param boundary The boundary configuration.
     */
    Laplacian(const GridDim& dim, const Real3d& delta, const BoundaryConfig<T>& boundary);

    /**
    * @brief Assemble and store the discrete Laplacian operator and boundary contribution.
    *
    *This method allocates its own memory.
    *
    * Builds the matrix representation of the staggered-grid Laplacian and the
    * corresponding right-hand-side contribution induced by boundary conditions.
    * The resulting operator is stored internally in banded form for reuse across solves.
    *
    * @param stream CUDA stream on which all operations will be enqueued.
    *               The caller is responsible for stream synchronization if needed.
    *
    * @note
    * - All buffers must be allocated prior to calling this function.
    * - The contents of the provided buffers are overwritten.
    * - The assembled operator and RHS contribution are retained internally
    *   and can be reused for multiple solves with different physical RHS terms.
    */
    void setOperation(cudaStream_t stream);

    /**
    * @brief Assemble and store the discrete Laplacian operator and boundary contribution.
    *
    * Builds the matrix representation of the staggered-grid Laplacian and the
    * corresponding right-hand-side contribution induced by boundary conditions.
    * The resulting operator is stored internally in banded form for reuse across solves.
    *
    * @param stream CUDA stream on which all operations will be enqueued.
    *               The caller is responsible for stream synchronization if needed.
    *
    * @param preAllocatedForL Preallocated matrix buffer that will be overwritten
    *        with the assembled Laplacian operator. Must have the correct size.
    *
    * @param preAllocatedForIndices Preallocated buffer that will be filled with
    *        index mappings required for banded matrix construction.
    *
    * @param rhsModifier Preallocated vector that will be overwritten with the
    *        boundary-condition contribution to the right-hand side (b_bc).
    *
    * @note
    * - All buffers must be allocated prior to calling this function.
    * - The contents of the provided buffers are overwritten.
    * - The assembled operator and RHS contribution are retained internally
    *   and can be reused for multiple solves with different physical RHS terms.
    */
    void setOperation(cudaStream_t stream, Mat<T> &preAllocatedForL, Vec<int32_t> &preAllocatedForIndices, Vec<T> &rhsModifier);

    /**
    * @brief Compute boundary-condition contributions to the right-hand side vector.
    *
    *This method allocates its own memory.
    *
    * Assembles the RHS modifications due to boundary conditions on all six faces.
    * Contributions from multiple faces (at corners and edges) accumulate safely via
    * atomic operations.
    *
    * @param[in] stream      CUDA stream for kernel execution.
    */
    void setRhsBC(cudaStream_t stream);

    /**
    * @brief Compute boundary-condition contributions to the right-hand side vector.
    *
    * Assembles the RHS modifications due to boundary conditions on all six faces.
    * Contributions from multiple faces (at corners and edges) accumulate safely via
    * atomic operations.
    *
    * @param[in] stream      CUDA stream for kernel execution.
    * @param[in,out] rhsModifier Vector to be filled with boundary-condition RHS contributions.
    */
    void setRhsBC(cudaStream_t stream, Vec<T> &rhsModifier);

    /**
     * Gets the banded matrix with the given dimension 0 -> X, 1 -> Y, and 2 -> Z.
     * @param dim The desired dimension of the 1d operator.
     * @return The pointer to the desired matrix.  This will be null if the matrix is not set yet.
     */
    std::unique_ptr<BandedMat<T>>& get1d(int dim);

    /**
     * Gets the value of L.
     * @param stream used to build BandedL if it doesn't exist.
     * @return BandedL.
     */
    BandedMat<T>& banded(cudaStream_t stream = nullptr);

    /**
     * The dense version of this matrix.
     * @param handle
     * @return The dense version of this matrix.
     */
    SquareMat<T> dense(Handle &handle);
};



#endif //CUDABANDED_POISSONLHS_H
