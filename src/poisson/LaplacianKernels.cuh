
#ifndef CUDABANDED_LAPLACIANKERNELS_CUH
#define CUDABANDED_LAPLACIANKERNELS_CUH
#include "deviceArrays/headers/DeviceData.cuh"

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
    __host__ __device__ AdjacencyPatern(GridDim dim):
        here(0, 0),
        up(1, -1),
        down(2, 1),
        left(3, -dim.rows * dim.layers),
        right(4, dim.rows * dim.layers),
        front (5, -dim.rows),
        back(6, dim.rows)
    {

    };

    __host__ void loadMapRowToDiag(Vec<int32_t>& diags, cudaStream_t stream) const;

    __host__ static void loadMapRowToDiag(Vec<int32_t> &diags, std::vector<AdjacencyInd> indices, cudaStream_t stream);
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
class LSetter {

public:
    DeviceData2d<T>* L;
    size_t rowL;
    /**
     * @brief Constructs a DimensionSetter for a given grid point.
     */
    __device__ LSetter(DeviceData2d<T>& L, size_t rowL) : L(&L), rowL(rowL) {}

    /**
     * @brief Set coefficients for a 1D row in the banded Laplacian.
     *
     * This method handles all cases:
     * - Boundary at start (index == 0)
     * - Boundary at end   (index == lineLength - 1)
     * - Interior node     (otherwise)
     *
     * Diagonal and off-diagonal structure are set according
     * to the supplied adjacency pattern and boundary conditions.
     *
     * @param indexInLine       Grid point index along this dimension.
     * @param lineLength        Number of grid points in this dimension.
     * @param lineStart         Boundary condition at start   (index == 0).
     * @param lineEnd           Boundary condition at end     (index == lineLength-1).
     * @param primary          The value at the main diagoanl.
     * @param left          The value at the left diagonal on the row.
     * @param right         The value at the right diagonal on the row.

     */
    __device__ void setRowInBanded1d(
        const size_t indexInLine, const size_t lineLength,
        const BoundaryConditionPair<T>& boundaries,
        const AdjacencyInd& primary, const AdjacencyInd& left, const AdjacencyInd& right
    ) {
        T& mainDiag = (*L)(rowL, primary);
        T& rightDiag = (*L)(rowL, right);
        T& leftDiag = (*L)(rowL, left);

        if (!boundaries.setL(mainDiag, rightDiag)){
            mainDiag -= 2 * boundaries.start.inverseDeltaSquared;
            leftDiag = rightDiag = boundaries.start.inverseDeltaSquared;
        }
    }
};

/**
 * This class does the same as LSetter, but stores primary, right, and left values.
 * @tparam T
 */
template<typename T>
class LSetter1d {
    LSetter<T> lSetter;
    const AdjacencyInd& primary;
    const AdjacencyInd& right;
    const AdjacencyInd& left;
public:
    /**
     *
     * @param L The banded laplacian to be set.
     * @param rowL The row of the laplacian that is to be set.
     * @param primary The column index of the primary diagonal in the banded matrix.
     * @param right The index of the righ diagonal's column.
     * @param left The index of the left diagonal's column.
     */
    __device__ LSetter1d(DeviceData2d<T>& L, size_t rowL, const AdjacencyInd& primary, const AdjacencyInd& left, const AdjacencyInd& right) :
        lSetter(L, rowL),
        primary(primary),
        right(right),
        left(left)
    {}


    /**
     * @brief Set coefficients for a 1D row in the banded Laplacian.
     *
     * This method handles all cases:
     * - Boundary at start (index == 0)
     * - Boundary at end   (index == lineLength - 1)
     * - Interior node     (otherwise)
     *
     * Diagonal and off-diagonal structure are set according
     * to the supplied adjacency pattern and boundary conditions.
     *
     * @param lineStart         Boundary condition at start   (index == 0).
     * @param lineEnd           Boundary condition at end     (index == lineLength-1).
     */
    __device__ void setRowInBanded1d(
        DeviceData2d<T>& L,
        const BoundaryConditionPair<T>& boundary
    ) {
        lSetter.L = &L;
        lSetter.setRowInBanded1d(lSetter.rowL, L.rows, boundary, primary, left, right);
    }
};

/**
 * Sets both the laplacian and the BC modifications for the rhs.
 * @tparam T
 */
template<typename T>
class LAndRhsSetter : public LSetter<T>{
    DeviceData1d<T> rhs;
public:
    __device__ LAndRhsSetter(DeviceData2d<T>& L, DeviceData1d<T>& rhs, size_t flatInd)
        : rhs(rhs), LSetter<T>(L, flatInd) {}

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
     * @param primary          The value on the main diagonal.
     * @param left          The value on the left diagonal.
     * @param right          The value on the right diagonal.
     *
     */
    __device__ void setRowInBanded1dAndRhs(
        const size_t indexInLine, const size_t lineLength,
        const BoundaryConditionPair<T>& condition,
        const AdjacencyInd& primary, const AdjacencyInd& left, const AdjacencyInd& right
    ) {
        LSetter<T>::setRowInBanded1d(indexInLine, lineLength, condition, primary, left, right);
        condition.setBoundaryRHS(rhs, indexInLine, lineLength);

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

    ds.setRowInBanded1dAndRhs(
        gridInd.row, dim.rows,
        boundary.top, boundary.bottom,
        ap.here, ap.up, ap.down
    );
    ds.setRowInBanded1dAndRhs(
        gridInd.col, dim.cols,
        boundary.left, boundary.right,
        ap.here, ap.left, ap.right
    );
    if (dim.layers > 1)
        ds.setRowInBanded1dAndRhs(
            gridInd.layer, dim.layers,
            boundary.front, boundary.back,
            ap.here, ap.back, ap.front
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
__global__ void buildRhsBCKernel(const GridDim dim, const BoundaryConfig<T> boundary, DeviceData1d<T> rhs) {
    GridInd2d ind;
    GridInd3d ind3d;

    T& rhsVal = rhs[dim[ind3d]];

    if (dim.layers > 1 && ind.row < dim.rows && ind.col < dim.cols) {
        ind3d.set(ind.row, ind.col, 0);
        boundary.front.setBoundaryRHS(rhsVal);
        ind3d.layer = dim.layers - 1;
        boundary.back.setBoundaryRHS(rhsVal);
    }
    if (ind.row < dim.rows && ind.col < dim.layers) {
        ind3d.set(ind.row, 0, ind.col);
        boundary.left.setBoundaryRHS(rhsVal);
        ind3d.col = dim.cols - 1;
        boundary.right.setBoundaryRHS(rhsVal);
    }
    if (ind.row < dim.layers && ind.col < dim.cols) {
        ind3d.set(0, ind.col, ind.row);
        boundary.top.setBoundaryRHS(rhsVal);
        ind3d.row = dim.rows - 1;
        boundary.bottom.setBoundaryRHS(rhsVal);
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
__global__ void buildL1dKernel(
    DeviceData2d<T> bandedL_i,
    const BoundaryConditionPair<T> condition,
    const AdjacencyInd primary,
    const AdjacencyInd prev,
    const AdjacencyInd next
) {
    size_t i = idx();
    if (i >= bandedL_i.rows) return;

    LSetter<T> ds(bandedL_i, i);
    ds.setRowInBanded1d(i, bandedL_i.rows, condition, primary, next, prev);
}

template <typename T>
__global__ void buildAllL1dKernel(DeviceData2d<T> bandedL_x, DeviceData2d<T> bandedL_y, DeviceData2d<T> bandedL_z, const BoundaryConfig<T> boundary, const AdjacencyInd primary, const AdjacencyInd prev, const AdjacencyInd next) {
    size_t i = idx();

    LSetter1d<T> ds(bandedL_x, i, primary, next, prev);
    if (i < bandedL_x.rows) ds.setRowInBanded1d(bandedL_x, boundary.left, boundary.right);
    if (i < bandedL_y.rows) ds.setRowInBanded1d(bandedL_y, boundary.top, boundary.bottom);
    if (i < bandedL_z.rows) ds.setRowInBanded1d(bandedL_z, boundary.front, boundary.back);
}



#endif //CUDABANDED_LAPLACIANKERNELS_CUH
