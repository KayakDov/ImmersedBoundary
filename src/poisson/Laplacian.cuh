
#ifndef CUDABANDED_POISSONLHS_H
#define CUDABANDED_POISSONLHS_H
#include <array>

#include "../deviceArrays/headers/sparse/BandedMat.h"
#include "math/Real3d.h"
#include "deviceArrays/headers/SquareMat.h"
#include "poisson/BoundaryCondition.hpp"
#include "solvers/Event.h"

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
};

template<typename T>
class Laplacian {
protected:
    const AdjacencyPatern adjacncies;
    const GridDim dim;
    const Real3d delta;
    const BoundaryConfig<T> boundary;

    std::unique_ptr<BandedMat<T>> banded = nullptr;
    std::unique_ptr<Vec<T>> rhsBC = nullptr;

public:
    /**
     * Creates the LHS matrix of the linear system used for solving the Poisson equation.
     * @param dim The dimensions of the Poisson grid.
     * @param delta Distance between gird points.
     * @param boundary The boundary configuration.
     */
    Laplacian(const GridDim& dim, const Real3d& delta, const BoundaryConfig<T>& boundary);

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

};



#endif //CUDABANDED_POISSONLHS_H
