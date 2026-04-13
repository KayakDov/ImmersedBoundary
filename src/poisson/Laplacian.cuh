
#ifndef CUDABANDED_POISSONLHS_H
#define CUDABANDED_POISSONLHS_H
#include <array>

#include "../deviceArrays/headers/sparse/BandedMat.h"
#include "math/Real3d.h"
#include "deviceArrays/headers/SquareMat.h"

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

    void loadMapRowToDiag(Vec<int32_t> diags, cudaStream_t stream);
};


/**
 * Types of boundaries.
 */
enum class BCType {Neumann, Dirichlet, NA};
struct BoundaryConfig {
    BCType left, right, top, bottom, front, back;
    __host__ __device__ BoundaryConfig(BCType left, BCType right, BCType top, BCType bottom, BCType front = BCType::NA, BCType back = BCType::NA);
};


template<typename T>
class Laplacian {
protected:
    const AdjacencyPatern adjacncies;
    const GridDim dim;
    const Real3d delta;
    const BoundaryConfig config;

public:
    /**
     * Creates the LHS matrix of the linear system used for solving the Poisson equation.
     * @param dim The dimensions of the Poisson grid.
     */
    Laplacian(GridDim dim, Real3d delta = Real3d(1.0, 1.0, 1.0));

    /**
     * Sets the values into the laplacian
     * @param stream
     * @param preAlocatedForA This matrix should be height * width * depth X (5 if 2d grid or 7 if 3d grid).
     * The laplacian will be placed here.
     * @param preAlocatedForIndices This vector will store the indices of the diagonals in A.
     * If the grid is 2d there should be 5 values here, if the grid is 3d there should be 7.
     * @return
     */
    virtual BandedMat<T> setL(cudaStream_t stream, Mat<T> &preAlocatedForA, Vec<int32_t> &preAlocatedForIndices) = 0;

    /**
     * Creates a vector that should be added to the rhs when solving L x = rhs, to account for the boundary conditions.
     * @param stream
     * @param rhs An empty vector that will be overwritten with the modifications that should be made to the rhs.
     */
    virtual void setRHS(cudaStream_t stream, Vec<T> &rhs) const = 0;

};

template<typename T>
class LaplacianNodeCentered : public Laplacian<T> {
public:
    LaplacianNodeCentered(GridDim dim, Real3d delta = Real3d(1.0, 1.0, 1.0));
    /** @inheritdoc */
    BandedMat<T> setL(cudaStream_t stream, Mat<T> &preAlocatedForA, Vec<int32_t> &preAlocatedForIndices) override;

    /**
     * Allocates memory for, and creates, a laplacian.
     * @param dim The dimensions of the laplacian's grid.
     * @param hand
     * @return A new Laplacian.
     */
    static BandedMat<T> L(const GridDim &dim, Handle &hand, Real3d delta = Real3d(1, 1, 1));

    /**
     * Prints a laplacian.
     * @param dim
     * @param hand
     */
    static void printL(const GridDim &dim, Handle &hand, Real3d delta = Real3d(1, 1, 1));

    /** @inheritdoc */
    void setRHS(cudaStream_t stream, Vec<T> &rhs) const override;
};
template<typename T>
class LaplacianStagared : public Laplacian<T> {

public:
    /**
     *
     * @param dim The number of dimensions of the grid.
     * @param delta The space between nodes of the grid.  Note that only have this space exists between nodes and
     * boundary conditions.
     */
    LaplacianStagared(GridDim dim, Real3d delta = Real3d(1.0, 1.0, 1.0));

    /** @inheritdoc */
    BandedMat<T> setL(cudaStream_t stream, Mat<T> &preAlocatedForA, Vec<int32_t> &preAlocatedForIndices) override;
    /** @inheritdoc */
    void setRHS(cudaStream_t stream, Vec<T> &rhs) const override;
};

#endif //CUDABANDED_POISSONLHS_H
