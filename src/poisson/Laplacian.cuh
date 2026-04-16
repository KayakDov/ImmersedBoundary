
#ifndef CUDABANDED_POISSONLHS_H
#define CUDABANDED_POISSONLHS_H
#include <array>
#include <vector>

#include "../deviceArrays/headers/sparse/BandedMat.h"
#include "math/Real3d.h"
#include "deviceArrays/headers/SquareMat.h"
#include "poisson/BoundaryCondition.hpp"
#include "solvers/Event.h"
#include "poisson//LaplacianKernels.cuh"

constexpr size_t numDiagonals3d = 7;
constexpr size_t numDiagonals2d = 5;

struct AdjacencyInd {
    /**
     * The column in the banded matrix.
     */
    const size_t col;
    /**
     * The index of the diagonal that is held by that column.
     */
    const int32_t diag;
    __device__ __host__ AdjacencyInd(const size_t col, const int32_t diag) : col(col), diag(diag) {
    }
};

template<typename T>
class Laplacian1dManager {
public:
    std::unique_ptr<BandedMat<T>> Lx = nullptr;
    std::unique_ptr<BandedMat<T>> Ly = nullptr;
    std::unique_ptr<BandedMat<T>> Lz = nullptr;

    std::unique_ptr<BandedMat<T>>& operator[](size_t dim);
};

/**
 * How the adjacent grid cells are stored in the laplacian. *
 */
class AdjacencyPatern {
public:

    AdjacencyInd here, up, down, left, right, front, back;
    /**
     *
     * @param dim The dimensions of the grid.
     */
    __host__ __device__ AdjacencyPatern(GridDim dim);

    void loadMapRowToDiag(Vec<int32_t>& diags, cudaStream_t stream) const;

    static void loadMapRowToDiag(Vec<int32_t> &diags, std::vector<AdjacencyInd> indices, cudaStream_t stream);


};

template<typename T>
class Laplacian {
protected:
    const AdjacencyPatern adjacncies;
    const GridDim dim;
    const Real3d delta;
    const BoundaryConfig<T> boundary;

    std::unique_ptr<BandedMat<T>> bandedL = nullptr;
    std::unique_ptr<Vec<T>> rhsBC = nullptr;


public:

    Laplacian1dManager<T> _1d;

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
     * @brief Build a 1D banded Laplacian operator for a single dimension.
     *
     * Constructs the 1D finite difference matrix for dimension `dim` with appropriate
     * boundary conditions and diagonal storage layout. The resulting banded matrix is
     * stored internally for later use.
     *
     * @param[in] stream                    CUDA stream for kernel execution.
     * @param[in,out] preAllocatedForL_i    Banded matrix (size(dim) × 3) to fill with coefficients.
     * @param[in,out] preAllocatedForIndices Vector of length 3 for diagonal indices.
     * @param[in] dim                       Dimension: 0=row/y, 1=col/x, 2=layer/z.
     */
    void setL_i(cudaStream_t stream, SquareMat<T> &preAllocatedForL_i, Vec<int32_t> &preAllocatedForIndices, size_t dim);

    /**
     * @brief Effectively calls setL_i for each dimension.
     */
    void setL_iAll(cudaStream_t stream, SquareMat<T> *preAllocatedForL_iX3, Vec<int32_t> &preAllocatedForIndices);

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
 * @brief Build a 1D banded Laplacian operator for a single dimension.
 *
 * Allocates memory and constructs the 1D finite difference matrix for dimension `dim`.
 *
 * @param[in] stream CUDA stream for kernel execution.
 * @param[in] dim    Dimension: 0=row/y, 1=col/x, 2=layer/z.
 */
    void setL_i(cudaStream_t stream, size_t dim);

    /**
     * @brief Build 1D banded Laplacian operators for all three dimensions.
     *
     * Allocates memory and constructs all three 1D finite difference matrices simultaneously.
     *
     * @param[in] stream CUDA stream for kernel execution.
     */
    void setL_iAll(cudaStream_t stream);
};



#endif //CUDABANDED_POISSONLHS_H
