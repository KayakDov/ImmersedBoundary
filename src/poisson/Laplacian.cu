//
// Created by usr on 12/24/25.
//

#include "poisson/Laplacian.cuh"

#include <vector>

#include "deviceArrays/headers/Support/Streamable.h"
#include "math/Real3dDevice.hpp"


/**
 * holds a flat index.
 */
class FlatInd {
public:
    size_t flat;            ///< Flattened grid index of the current grid point

    /**
     *  A constructor.
     * @param flat The index
     */
    __device__ FlatInd(const size_t flat) : flat(flat) {}

    /**
     * Sets the flat index
     * @param flat The new flat index.
     */
    __device__ void setFlat(size_t flat) {this->flat = flat;}
};

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
class RhsBCSetter1d : virtual public FlatInd{
protected:
    DeviceData1d<T> &rhs;   ///< Reference to the right-hand side vector
public:
    /**
     * @brief Constructs a DimensionSetter for a given grid point.
     *
     * @param[in,out] rhs The right-hand side vector.
     * @param[in] flat    The flattened index of the current grid point.
     */
    __device__ RhsBCSetter1d(DeviceData1d<T> & rhs, size_t flat) : rhs(rhs), FlatInd(flat) {}

    /**
     * @brief Applies boundary-condition contributions to the RHS along one dimension.
     *
     * This method mirrors the boundary handling of the 1D Laplacian stencil but only
     * updates the boundary-condition contribution to the right-hand side vector.
     * No modifications to the operator (L) are performed.
     *
     * @param[in] indexInLine Index of the current grid point along this dimension (0 to end-1).
     * @param[in] end       Size of the grid in this dimension (rows, cols, or layers).
     * @param[in] left      Boundary condition at gridIndex == 0.
     * @param[in] right     Boundary condition at gridIndex == end - 1.
     */
    __device__ void setRHSIn1d(const size_t indexInLine, const size_t end, const BoundaryCondition<T>& left, const BoundaryCondition<T>& right) {
        if (indexInLine == 0) left.setBoundaryRHSContribution(flat, rhs);
        else if (indexInLine == end - 1) right.setBoundaryRHSContribution(flat, rhs);
    }
};

/**
 * @class LSetter
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
class LSetter : virtual public FlatInd {
    DeviceData2d<T> &L;     ///< Reference to the banded system matrix L
    size_t diagOffset, primraryDiagColInd, rightDiagColInd, leftDiagColInd;
public:
    /**
     * @brief Constructs a DimensionSetter for a given grid point.
     *
     * @param[in,out] L   The banded system matrix (7 or 5 diagonals for 3D or 2D).
     * @param[in] flat    The flattened index of the current grid point.
     */
    __device__ LSetter(DeviceData2d<T>& L, size_t flat) : FlatInd(flat),L(L) {}

    /**
     *
     * @param L The banded laplacian to be set.
     * @param flat The row of the laplacian that is to be set.
     * @param diagOffset The number of elements from the primary diagonal to the nex diagonal that will be changed.
     * This is also the index of the secondary diagonal.
     * @param primraryDiagColInd The column index of the primary diagonal in the banded matrix.
     * @param rightDiagColInd The index of the righ diagonal's column.
     * @param leftDiagColInd The index of the left diagonal's column.
     */
    __device__ LSetter(DeviceData2d<T>& L, size_t flat, size_t diagOffset, size_t primraryDiagColInd, size_t rightDiagColInd, size_t leftDiagColInd) :
        FlatInd(flat),L(L),
        diagOffset(diagOffset),
        primraryDiagColInd(primraryDiagColInd),
        rightDiagColInd(rightDiagColInd),
        leftDiagColInd(leftDiagColInd)
        {}

    /**
     * Sets the laplacian the this dimension setter will modify on its subsequent calls.
     * @param L The new laplacian to be held here.
     */
    __device__ void setL(DeviceData2d<T>& L) {
        this->L = L;
    }

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
     * @param[in] indexInLine    Index of the current grid point along this dimension (0 to end-1).
     * @param[in] lineLength          Size of the grid in this dimension (rows, cols, or layers).
     * @param[in] lineStart         Boundary condition at gridIndex == 0.
     * @param[in] lineEnd        Boundary condition at gridIndex == end - 1.
     * @param[in] diagOffset   Absolute value of the negative diagonal index (row offset for
     *                         the negative diagonal in banded storage).
     * @param[in] primraryDiagColInd Column index in L where the main diagonal is stored.
     * @param[in] rightDiagColInd    Column index in L where the positive diagonal is stored.
     * @param[in] leftDiagColInd     Column index in L where the negative diagonal is stored.
     */
    __device__ void setRowInBanded1d(
        const size_t indexInLine, const size_t lineLength,
        const BoundaryCondition<T> lineStart, const BoundaryCondition<T> lineEnd,
        const size_t diagOffset, const size_t primraryDiagColInd, const size_t rightDiagColInd, const size_t leftDiagColInd
    ) {
        if (indexInLine == 0) lineStart.setL(L, this->flat, primraryDiagColInd, rightDiagColInd);
        else if (indexInLine == lineLength - 1) lineEnd.setL(L, this->flat - diagOffset, primraryDiagColInd, leftDiagColInd);
        else {
            L(this->flat, primraryDiagColInd) -= 2 * lineStart.inverseDeltaSquared;
            L(this->flat, rightDiagColInd) = L(this->flat - diagOffset, leftDiagColInd) = lineStart.inverseDeltaSquared;
        }
    }

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
     *   Note, this method assumes that the index in line is the same as the flat index.
     *
     * @param[in] lineLength          Size of the grid in this dimension (rows, cols, or layers).
     * @param[in] lineStart         Boundary condition at gridIndex == 0.
     * @param[in] lineEnd        Boundary condition at gridIndex == end - 1.
     */
    __device__ void setRowInBanded1d(const size_t lineLength, const BoundaryCondition<T> lineStart, const BoundaryCondition<T> lineEnd) {
        setRowInBanded1d(this->flat, lineLength, lineStart, lineEnd, diagOffset, primraryDiagColInd, rightDiagColInd, leftDiagColInd);
    }
};

template<typename T>
class LAndRhsSetter : public LSetter<T>, public RhsBCSetter1d<T>{
    __device__ LAndRhsSetter(DeviceData2d<T>& L, DeviceData1d<T>& rhs, size_t flat)
        : FlatInd(flat), LSetter<T>(L, flat), RhsBCSetter1d<T>(rhs, flat) {}

    /**
     * @brief Applies the 1D Laplacian stencil along one dimension with boundary condition handling.
     *
     * Sets both the system matrix L and the RHS vector contributions for a single row,
     * accounting for boundary conditions and banded matrix storage.
     *
     * @param[in] indexInLine    Index of the current grid point along this dimension (0 to end-1).
     * @param[in] lineLength     Size of the grid in this dimension.
     * @param[in] lineStart      Boundary condition at gridIndex == 0.
     * @param[in] lineEnd        Boundary condition at gridIndex == end - 1.
     * @param[in] diagOffset     Absolute value of the negative diagonal index.
     * @param[in] primraryDiagColInd Column index for the main diagonal.
     * @param[in] rightDiagColInd    Column index for the positive diagonal.
     * @param[in] leftDiagColInd     Column index for the negative diagonal.
     */
    __device__ void setRowInBanded1dAndRhs(
        const size_t indexInLine, const size_t lineLength,
        const BoundaryCondition<T> lineStart, const BoundaryCondition<T> lineEnd,
        const size_t diagOffset, const size_t primraryDiagColInd, const size_t rightDiagColInd, const size_t leftDiagColInd
    ) {
        LSetter<T>::setRowInBanded1d(indexInLine, lineLength, lineStart, lineEnd, diagOffset, primraryDiagColInd, rightDiagColInd, leftDiagColInd);
        RhsBCSetter1d<T>::setRHSIn1d(indexInLine, lineLength, lineStart, lineEnd);
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
__global__ void buildLaplacianKernel(DeviceData2d<T> L, const GridDim dim, const BoundaryConfig<T> boundary, const AdjacencyPatern ap, DeviceData1d<T> rhs) {
    GridInd3d gridInd;
    if (gridInd >= dim) return;

    LAndRhsSetter<T> ds(L, rhs, dim[gridInd]);

    L(ap.here.col, ds.flat) = rhs[ds.flat] = 0;

    ds.setRowInBanded1d(
        gridInd.row, dim.rows,
        boundary.top, boundary.bottom,
        ap.down.diag,
        ap.here.col, ap.down.col, ap.up.col
    );
    ds.setRowInBanded1d(
        gridInd.col, dim.cols,
        boundary.left, boundary.right,
        ap.right.diag,
        ap.here.col, ap.right.col, ap.left.col
    );
    if (dim.layers > 1)
        ds.setRowInBanded1d(
            gridInd.layer, dim.layers,
            boundary.front, boundary.back,
            ap.back.diag,
            ap.here.col, ap.back.col, ap.front.col
        );
}

/**
 * @brief CUDA kernel to assemble boundary-condition contributions to the RHS.
 *
 * Uses a 2D thread grid where each thread handles up to 6 boundary faces.
 * Threads are launched with dimensions max(rows, cols, layers) to efficiently
 * cover all faces without underutilization. Atomic operations safely handle
 * corners/edges where contributions from multiple faces accumulate.
 *
 * @param[in] dim        Grid dimensions.
 * @param[in] boundary   Boundary conditions for all six faces.
 * @param[in,out] rhs    Right-hand side vector; accumulates boundary contributions.
 */
template<typename T>
__global__ void buildRhsBCKernel(
    const GridDim dim,
    const BoundaryConfig<T> boundary,
    DeviceData1d<T> rhs
) {
    GridInd2d ind;
    GridInd3d ind3d;

    if (dim.layers > 1 && ind.row < dim.rows && ind.col < dim.cols) {
        ind3d.set(ind.row, ind.col, 0);
        boundary.front.setBoundaryRHSContribution(dim[ind3d], rhs);
        ind3d.layer = dim.layers - 1;
        boundary.back.setBoundaryRHSContribution(dim[ind3d], rhs);
    }
    if (ind.row < dim.rows && ind.col < dim.layers) {
        ind3d.set(ind.row, 0, ind.col);
        boundary.left.setBoundaryRHSContribution(dim[ind3d], rhs);
        ind3d.col = dim.cols - 1;
        boundary.right.setBoundaryRHSContribution(dim[ind3d], rhs);
    }
    if (ind.row < dim.layers && ind.col < dim.cols) {
        ind3d.set(0, ind.col, ind.row);
        boundary.top.setBoundaryRHSContribution(dim[ind3d], rhs);
        ind3d.row = dim.rows - 1;
        boundary.bottom.setBoundaryRHSContribution(dim[ind3d], rhs);
    }
}

/**
 * @brief CUDA kernel to build a 1D banded Laplacian operator.
 *
 * Each thread handles one grid point along the dimension, setting the stencil
 * coefficients (u_{i-1} - 2*u_i + u_{i+1})/delta^2 for the 1D finite difference.
 *
 * @tparam T Floating-point type (float or double).
 *
 * @param[in,out] bandedL_i       Banded matrix for this dimension (n × 3).
 * @param[in] start               Boundary condition at i == 0.
 * @param[in] end                 Boundary condition at i == n - 1.
 * @param[in] primary             Adjacency info for the main diagonal.
 * @param[in] prev                Adjacency info for the negative diagonal (u_{i-1}).
 * @param[in] next                Adjacency info for the positive diagonal (u_{i+1}).
 */
template <typename T>
__global__ void buildL1dKernel(DeviceData2d<T> bandedL_i, const BoundaryCondition<T> start, const BoundaryCondition<T> end, const AdjacencyInd primary, const AdjacencyInd prev, const AdjacencyInd next) {
    size_t i = idx();
    if (i >= bandedL_i.rows) return;

    LSetter<T> ds(bandedL_i, i);
    ds.setRowInBanded1d(i, bandedL_i.rows, start, end, next.diag, primary.col, next.col, prev.col);
}

template <typename T>
__global__ void buildAllL1dKernel(DeviceData2d<T> bandedL_x, DeviceData2d<T> bandedL_y, DeviceData2d<T> bandedL_z, const BoundaryConfig<T> boundary, const AdjacencyInd primary, const AdjacencyInd prev, const AdjacencyInd next) {
    size_t i = idx();

    LSetter<T> ds(bandedL_x, i, next.diag, primary.col, next.col, prev.col);
    if (i < bandedL_x.rows) ds.setRowInBanded1d(bandedL_x.rows, boundary.left, boundary.right);
    if (i < bandedL_y.rows) {
        ds.setL(bandedL_y);
        ds.setRowInBanded1d(bandedL_y.rows, boundary.top, boundary.bottom);
    }
    if (i < bandedL_z.rows) {
        ds.setL(bandedL_z);
        ds.setRowInBanded1d(bandedL_z.rows, boundary.front, boundary.back);
    }
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
    loadMapRowToDiag(diags, {here, up, down, left, right, front, back}, stream);
}

void AdjacencyPatern::loadMapRowToDiag(Vec<int32_t> &diags, std::vector<AdjacencyInd> indices, cudaStream_t stream) {
    std::vector<int32_t> diagsCpu(diags.size(), 0);
    for (AdjacencyInd ind : indices) diagsCpu[ind.col] = ind.diag;
    diags.set(diagsCpu.data(), stream);
}

template<typename T>
void Laplacian<T>::setOperation(cudaStream_t stream) {
    size_t numDiags =  dim.layers > 1 ? numDiagonals3d : numDiagonals2d;
    Vec<int32_t> preAllocatedForIndices = Vec<int32_t>::create(numDiags, stream);
    Mat<T> rhsAndL = Mat<T>::create(dim.size(), numDiags + 1);
    Mat<T> preAllocatedForL = rhsAndL.col(numDiags);
    Vec<T> rhsModifier = preAllocatedForL.col(rhsAndL.subMat(0,0,dim.size(), numDiags));

    setOperation(stream, preAllocatedForL, preAllocatedForIndices, rhsModifier);
}

template<typename T>
void Laplacian<T>::setOperation(cudaStream_t stream, Mat<T> &preAllocatedForL, Vec<int32_t> &preAllocatedForIndices, Vec<T>& rhsModifier) {
    KernelPrep kp = this->dim.kernelPrep();
    buildLaplacianKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAllocatedForL,
        this->dim, this->boundary,
        this->adjacncies,
        rhsModifier
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    this->adjacncies.loadMapRowToDiag(preAllocatedForIndices, stream);

    bandedL = std::make_unique<BandedMat<T>>(preAllocatedForL, preAllocatedForIndices);
    rhsBC = std::make_unique<Vec<T>>(rhsModifier);
}

template<typename T>
void Laplacian<T>::setL_i(cudaStream_t stream, SquareMat<T> &preAllocatedForL_i, Vec<int32_t> &preAllocatedForIndices, size_t dim) {

    AdjacencyInd prev(1, -1), next(2, 1), primary(0, 0);

    KernelPrep kp(dim);
    CHECK_CUDA_ERROR(buildL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAllocatedForL_i,
        this->boundary(dim, true), this->boundary(dim, false),
        primary, prev, next
    ));

    AdjacencyPatern::loadMapRowToDiag(preAllocatedForIndices, {primary, prev, next}, stream);

    get1d(dim) = std::make_unique<BandedMat<T>>(preAllocatedForL_i, preAllocatedForIndices);
}

template<typename T>
void Laplacian<T>::setL_iAll(cudaStream_t stream, SquareMat<T>* preAllocatedForL_iX3, Vec<int32_t> &preAllocatedForIndices) {

    AdjacencyInd prev(1, -1), next(2, 1), primary(0, 0);

    KernelPrep kp(this->dim.maxDim());

    CHECK_CUDA_ERROR(buildAllL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAllocatedForL_iX3[0], preAllocatedForL_iX3[1], preAllocatedForL_iX3[2],
        this->boundary,
        primary, prev, next
    ));

    AdjacencyPatern::loadMapRowToDiag(preAllocatedForIndices, {primary, prev, next}, stream);

    for (size_t i = 0; i < 3; i++)
        get1d(i) = std::make_unique<BandedMat<T>>(preAllocatedForL_iX3[i], preAllocatedForIndices);
}

template<typename T>
void Laplacian<T>::setL_i(cudaStream_t stream, size_t dim) {
    Vec<int32_t> preAllocatedForIndices = Vec<int32_t>::create(3, stream);
    Mat<T> preAllocatedForL_i = SquareMat<T>::create(this->dim[dim], 3);
    setL_i(stream, preAllocatedForL_i, preAllocatedForIndices, dim);
}

template<typename T>
void Laplacian<T>::setL_iAll(cudaStream_t stream) {
    Vec<int32_t> preAllocatedForIndices = Vec<int32_t>::create(3, stream);
    Mat<T> preAllocatedForL_iX3[3] = {
        SquareMat<T>::create(this->dim[0], 3),
        SquareMat<T>::create(this->dim[1], 3),
        SquareMat<T>::create(this->dim[2], 3)
    };
    setL_iAll(stream, preAllocatedForL_iX3, preAllocatedForIndices);
}

template<typename T>
void Laplacian<T>::setRhsBC(cudaStream_t stream) {
    Vec<T> rhsModifier = SimpleArray<T>::create(dim.size());
    setRhsBC(stream, rhsModifier);
}

template<typename T>
void Laplacian<T>::setRhsBC(cudaStream_t stream, Vec<T>& rhsModifier) {
    KernelPrep kp(std::max(dim.rows, dim.layers), std::max(dim.layers, dim.cols));
    buildRhsBCKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(dim, boundary, rhsModifier);
    CHECK_CUDA_ERROR(cudaGetLastError());
    rhsBC = std::make_unique<Vec<T>>(rhsModifier);
}

template<typename T>
std::unique_ptr<BandedMat<T>>& Laplacian<T>::get1d(int32_t dim) {
    switch (dim) {
        case 0: return Lx;
        case 1: return Ly;
        case 2: return Lz;
        default: throw std::out_of_range("Invalid dimension: must be 0, 1, or 2");
    }
}

template class Laplacian<float>;
template class Laplacian<double>;
