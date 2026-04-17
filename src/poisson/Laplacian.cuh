
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

/**
 * Used to check of two 1d Laplacians are equal.
 * @tparam T
 */
template<typename T>
struct LaplcianConditions {
    const BoundaryConditionPair<T> boundary;
    const size_t dim;
    LaplcianConditions(const size_t length, const BoundaryConditionPair<T>& condition);

    /**
     * @brief Equality operator implemented as a hidden friend.
     */
    friend bool operator==(const LaplcianConditions& lhs, const LaplcianConditions& rhs) {
        return (lhs.dim == rhs.dim) && (lhs.boundary == rhs.boundary);
    }

    /**
     * @brief Inequality operator (C++20 will generate this automatically if you only
     * provide operator==, but for older standards it is good practice to include).
     */
    friend bool operator!=(const LaplcianConditions& lhs, const LaplcianConditions& rhs) {
        return !(lhs == rhs);
    }
};


/**
 * The one dimensional Laplacians for a 2 or 3d grid.  Note, if a pair of Laplacians have the same dimension size and
 * boundary conditions, then the pointers may point to the same memory.
 * @tparam T
 */
template<typename T>
class Laplacian1d {
public:

    std::unique_ptr<BandedMat<T>> Lx = nullptr;
    std::unique_ptr<BandedMat<T>> Ly = nullptr;
    std::unique_ptr<BandedMat<T>> Lz = nullptr;

    /**
     * The boundary.
     */
    const BoundaryConfig<T> boundary;
    /**
     * Selects one of the laplacian 1d operators.
     * @param dim 0 for the x dimension, 1 for the y dimesnion, and 2 for the z dimension.
     * @return a 1d operator.
     */
    std::unique_ptr<BandedMat<T>>& operator[](size_t dim);


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
    void set(cudaStream_t stream, Mat<T> &preAllocatedForL_i, Vec<int32_t> &preAllocatedForIndices, size_t dim);

    /**
     * @brief Effectively calls setL_i for each dimension.
     */
    void set(cudaStream_t stream, Mat<T> *preAllocatedForL_iX3, Vec<int32_t> &preAllocatedForIndices);


    /**
     * @brief Build a 1D banded Laplacian operator for a single dimension.
     *
     * Allocates memory and constructs the 1D finite difference matrix for dimension `dim`.
     *
     * @param[in] stream CUDA stream for kernel execution.
     * @param n The number of elements in this dimension.  The vector length.
     * @param[in] dim    Dimension: 0=row/y, 1=col/x, 2=layer/z.
     */
    void set(cudaStream_t stream, size_t n, size_t dim);

    /**
     * @brief Build 1D banded Laplacian operators for all three dimensions.
     *
     * Ensures that if two Laplacians have the same dimension size and memory, then only one object in memory is created.
     *
     * Allocates memory and constructs all three 1D finite difference matrices simultaneously.
     *
     * @param[in] stream CUDA stream for kernel execution.
     * @param dim The dimension of the underlying 2 or 3d grid.
     */
    void set(cudaStream_t stream, const GridDim& dim);
};

/**
 * The matrices of eigen vectors for a 2d or 3d laplacian.
 * @tparam T
 */
template<typename T>
class LaplacianEigenVec {
private:
    LaplacianEigenVec(const SquareMat<T>& eVecX, const SquareMat<T>& eVecY, const SquareMat<T>& eVecZ);
public:
    /**
     * The eigen vectors for L_x
     */
    const SquareMat<T> eVecX;
    /**
     * The eigen vectors for L_y
     */
    const SquareMat<T> eVecY;
    /**
     * The eigen vectors for L_z, if the laplacian is 3d.  Otherwise a null pointer.
     */
    const SquareMat<T> eVecZ;

    /**
     * Generates the eigenvector matrices.
     * @param boundary The boundary conditions.
     * @param dim The dimensions of the grid.
     * @param hand The handle.
     * @return The Laplacian's Eigen vector matrices.
     */
    LaplacianEigenVec factory(BoundaryConfig<T> boundary, const GridDim& dim, Handle& hand);
};

template<typename T>
struct LaplacianEigenVal {
    SquareMat<T> eVecX;
    SquareMat<T> eVecY;
    SquareMat<T> eVecZ;
};

template<typename T>
class LaplacianEigen {
    LaplacianEigenVal<T> vals;
    LaplacianEigenVal<T> vecs;

};

template<typename T>
class Laplacian {
protected:
    const AdjacencyPatern adjacencies;
    const GridDim dim;
    const Real3d delta;
    const BoundaryConfig<T> boundary;

    std::unique_ptr<BandedMat<T>> bandedL = nullptr;
    std::unique_ptr<Vec<T>> rhsBC = nullptr;

public:

    Laplacian1d<T> _1d;

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
};



#endif //CUDABANDED_POISSONLHS_H
