//
// Created by usr on 12/24/25.
//

#include "poisson/Laplacian.cuh"

#include <vector>

#include "deviceArrays/headers/Support/Streamable.h"
#include "math/Real3dDevice.hpp"


/**
 * @class RhsBCSetter1d
 * @brief Applies boundary-condition contributions to the RHS along a single dimension.
 *
 * This class provides a lightweight mechanism for adding boundary-condition-induced
 * contributions to the right-hand side vector without modifying the system operator.
 * It is intended for use in matrix-free or preassembled-operator workflows where only
 * the RHS must reflect boundary conditions.
 *
 * Each instance operates on a single grid point, identified by its flattened index,
 * and is typically invoked once per spatial dimension.
 *
 * @tparam T Floating-point type (float or double).
 */
template<typename T>
class RhsBCSetter1d {
protected:
    DeviceData1d<T> &rhs;   ///< Reference to the right-hand side vector
public:
    const size_t flat;            ///< Flattened grid index of the current grid point
    /**
     * @brief Constructs a DimensionSetter for a given grid point.
     *
     * @param[in,out] rhs The right-hand side vector.
     * @param[in] flat    The flattened index of the current grid point.
     */
    RhsBCSetter1d(DeviceData1d<T> & rhs, size_t flat) : rhs(rhs), flat(flat) {}

    /**
     * @brief Applies boundary-condition contributions to the RHS along one dimension.
     *
     * This method mirrors the boundary handling of the 1D Laplacian stencil but only
     * updates the boundary-condition contribution to the right-hand side vector.
     * No modifications to the operator (L) are performed.
     *
     * @param[in] gridIndex Index of the current grid point along this dimension (0 to end-1).
     * @param[in] end       Size of the grid in this dimension (rows, cols, or layers).
     * @param[in] left      Boundary condition at gridIndex == 0.
     * @param[in] right     Boundary condition at gridIndex == end - 1.
     */
    __device__ void setRHSIn1d(const size_t gridIndex, const size_t end, const BoundaryCondition<T>& left, const BoundaryCondition<T>& right) {
        if (gridIndex == 0) left.setBoundaryRHSContribution(flat, rhs);
        else if (gridIndex == end - 1) right.setBoundaryRHSContribution(flat, rhs);
    }
};

/**
 * @class DimensionSetter
 * @brief Helper class to set 1D Laplacian stencil coefficients for each spatial dimension.
 *
 * This class facilitates the application of 1D finite difference stencils along each
 * dimension of a 3D grid, accumulating contributions to the banded system matrix L
 * and modifying the RHS vector to account for boundary conditions.
 *
 * The staggered grid Laplacian is built by calling laplacianStaggered1d() three times—
 * once for each dimension (row, column, layer)—with coefficients accumulating on the
 * system matrix.
 *
 * @tparam T Floating-point type (float or double).
 */
template<typename T>
class DimensionSetter : public RhsBCSetter1d<T> {
    DeviceData2d<T> &L;     ///< Reference to the banded system matrix L
public:
    /**
     * @brief Constructs a DimensionSetter for a given grid point.
     *
     * @param[in,out] L   The banded system matrix (7 or 5 diagonals for 3D or 2D).
     * @param[in,out] rhs The right-hand side vector.
     * @param[in] flat    The flattened index of the current grid point.
     */
    DimensionSetter(DeviceData2d<T>& L, DeviceData1d<T> & rhs, size_t flat) : RhsBCSetter1d<T>(rhs, flat),L(L) {}

    /**
     * @brief Applies the 1D Laplacian stencil along one dimension with boundary condition handling.
     *
     *
     * The method accounts for banded matrix storage where:
     * - The main diagonal (index 0) is stored in column `primaryDiagColInd` at row `flat`.
     * - The positive diagonal is stored in column `rightDiagColInd` at row `flat`.
     * - The negative diagonal is stored in column `leftDiagColInd` at row `flat - diagOffset`,
     *   where `diagOffset` is the absolute value of the diagonal index.
     *
     * @param[in] gridIndex    Index of the current grid point along this dimension (0 to end-1).
     * @param[in] end          Size of the grid in this dimension (rows, cols, or layers).
     * @param[in] left         Boundary condition at gridIndex == 0.
     * @param[in] right        Boundary condition at gridIndex == end - 1.
     * @param[in] inverseDeltaSq Precomputed 1/delta^2 for this dimension.
     * @param[in] diagOffset   Absolute value of the negative diagonal index (row offset for
     *                         the negative diagonal in banded storage).
     * @param[in] primraryDiagColInd Column index in L where the main diagonal is stored.
     * @param[in] rightDiagColInd    Column index in L where the positive diagonal is stored.
     * @param[in] leftDiagColInd     Column index in L where the negative diagonal is stored.
     */
    __device__ void setRowInBanded1d(
        const size_t gridIndex, const size_t end,
        const BoundaryCondition<T> left, const BoundaryCondition<T> right,
        const T inverseDeltaSq,
        const size_t diagOffset, const size_t primraryDiagColInd, const size_t rightDiagColInd, const size_t leftDiagColInd
    ) {
        if (gridIndex == 0) left.setLAndRHS(L, this->flat, primraryDiagColInd, rightDiagColInd, this->rhs);
        else if (gridIndex == end - 1) right.setLAndRHS(L, this->flat - diagOffset, primraryDiagColInd, leftDiagColInd, this->rhs);
        else {
            L(this->flat, primraryDiagColInd) -= 2 * inverseDeltaSq;
            L(this->flat, rightDiagColInd) = L(this->flat - diagOffset, leftDiagColInd) = inverseDeltaSq;
        }
    }
};

/**
 * @brief CUDA kernel to set up the staggered grid Laplacian matrix and apply boundary conditions.
 *
 * @param[in,out] L        Banded system matrix (dim.size() × numDiagonals); coefficients are accumulated.
 * @param[in] dim          Grid dimensions.
 * @param[in] boundary     Boundary conditions for all six faces.
 * @param[in] ap           Adjacency pattern specifying diagonal storage layout.
 * @param[in,out] rhs      Right-hand side vector; modified by boundary conditions.
 * @param[in] invDeltaSq   Precomputed 1/delta^2 for each dimension.
 */
template<typename T>
__global__ void buildLaplacianKernel(DeviceData2d<T> L, const GridDim dim, const BoundaryConfig<T> boundary, const AdjacencyPatern ap, DeviceData1d<T> rhs, const Real3dDevice<T> invDeltaSq) {
    GridInd3d gridInd;
    if (gridInd >= dim) return;

    DimensionSetter<T> ds(L, rhs, dim[gridInd]);

    L(ap.here, ds.flat) = rhs[ds.flat] = 0;

    ds.setRowInBanded1d(
        gridInd.row, dim.rows,
        boundary.top, boundary.bottom,
        invDeltaSq.y,
        ap.down.diag, ap.here.col, ap.down.col, ap.up.col
    );
    ds.setRowInBanded1d(
        gridInd.col, dim.cols,
        boundary.left, boundary.right,
        invDeltaSq.x,
        ap.right.diag, ap.here.col, ap.right.col, ap.left.col
    );
    if (dim.layers > 1)
        ds.setRowInBanded1d(
            gridInd.layer, dim.layers,
            boundary.front, boundary.back,
            invDeltaSq.z,
            ap.back.diag, ap.here.col, ap.back.col, ap.front.col
        );
}

/**
 * @brief CUDA kernel to assemble boundary-condition contributions to the RHS on a staggered grid.
 *
 * Computes only the boundary-condition-induced contribution to the right-hand side
 * vector (rhsBC), without modifying the operator (L). This is intended for use with
 * preassembled or matrix-free Laplacian operators.
 *
 * @param[in] dim        Grid dimensions.
 * @param[in] boundary   Boundary conditions for all six faces.
 * @param[in] ap         Adjacency pattern specifying indexing layout (used for consistency).
 * @param[in,out] rhs    Right-hand side vector; overwritten with boundary contributions.
 */
template<typename T>
__global__ void buildRhsBCKernel(
    const GridDim dim,
    const BoundaryConfig<T> boundary,
    const AdjacencyPatern ap,
    DeviceData1d<T> rhs
) {
    GridInd3d gridInd;
    if (gridInd >= dim) return;

    RhsBCSetter1d<T> ds(rhs, dim[gridInd]);

    rhs[ds.flat] = 0;

    ds.setRHSIn1d(gridInd.row, dim.rows, boundary.top, boundary.bottom);

    ds.setRHSIn1d(gridInd.col, dim.cols, boundary.left, boundary.right);

    if (dim.layers > 1) ds.setRHSIn1d(gridInd.layer, dim.layers,boundary.front, boundary.back);

}

AdjacencyPatern::AdjacencyPatern(GridDim dim):
    here(0, 0),
    up(1, -1),
    down(2, 1),
    left(3, -dim.rows * dim.layers),
    right(4, dim.rows * dim.layers),
    front (5, -dim.rows),
    back(6, dim.rows)
    {

}


template<typename T>
Laplacian<T>::Laplacian(const GridDim& dim, const Real3d& delta, const BoundaryConfig<T>& boundary) :
    dim(dim),
    delta(delta),
    boundary(boundary),
    adjacncies(dim) {
}

template <typename T>
T invSq(T x) {
    return 1/(x*x);
}

void AdjacencyPatern::loadMapRowToDiag(Vec<int32_t>& diags, const cudaStream_t stream) const{
    std::vector<int32_t> diagsCpu(diags.size(), 0);
    diagsCpu[here.col] = here.diag;
    diagsCpu[up.col] = up.diag;
    diagsCpu[down.col] = down.diag;
    diagsCpu[left.col] = left.diag;
    diagsCpu[right.col] = right.diag;
    if (diagsCpu.size() > numDiagonals2d) {
        diagsCpu[front.col] = front.diag;
        diagsCpu[back.col] = back.diag;
    }
    diags.set(diagsCpu.data(), stream);
}

template<typename T>
void Laplacian<T>::setOperation(cudaStream_t stream, Mat<T> &preAllocatedForL, Vec<int32_t> &preAllocatedForIndices, Vec<T>& rhsModifier) {
    KernelPrep kp = this->dim.kernelPrep();
    buildLaplacianKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAllocatedForL,
        this->dim, this->boundary,
        this->adjacncies,
        rhsModifier,
        Real3dDevice<T>(
            1/(this->delta.x * this->delta.x),
            1/(this->delta.y * this->delta.y),
            1/(this->delta.z * this->delta.z))
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    this->adjacncies.loadMapRowToDiag(preAllocatedForIndices, stream);

    banded = std::make_unique<BandedMat<T>>(preAllocatedForL, preAllocatedForIndices);
    rhsBC = std::make_unique<Vec<T>>(rhsModifier);
}

template<typename T>
void Laplacian<T>::setRhsBC(cudaStream_t stream, Mat<T> &preAllocatedForL, Vec<int32_t> &preAllocatedForIndices, Vec<T>& rhsModifier) {

}

template class Laplacian<float>;
template class Laplacian<double>;
