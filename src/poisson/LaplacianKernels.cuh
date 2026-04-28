
#ifndef CUDABANDED_LAPLACIANKERNELS_CUH
#define CUDABANDED_LAPLACIANKERNELS_CUH
#include "deviceArrays/headers/DeviceData.cuh"



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
    DeviceData2d<T>* laplacian;
    size_t rowL;
    /**
     * @brief Constructs a DimensionSetter for a given grid point.
     */
    __device__ LSetter(DeviceData2d<T>& L, size_t rowL) : laplacian(&L), rowL(rowL) {}

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
     * @param boundaries    The boundary conditions in the dimesnion being worked on.
     * @param primary          The value at the main diagoanl.
     * @param leftRight          The value at the left and right diagonals on the row.


     */
    __device__ void setRowInBanded1d(
        const size_t indexInLine,
        const BoundaryPair<T>& boundaries,
        const AdjacencyInd& primary, const AdjacencyIndPair& leftRight
    ) {
        T& mainDiag = (*laplacian)(rowL, primary);
        T& rightDiag = (*laplacian)(rowL, leftRight.getRight());
        T& leftDiag = (*laplacian)(rowL, leftRight.getLeft());

        if (!boundaries.setL(mainDiag, leftDiag, rightDiag, indexInLine)) {
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
    const AdjacencyIndPair& leftRight;
public:
    /**
     *
     * @param L The banded laplacian to be set.
     * @param rowL The row of the laplacian that is to be set.
     * @param primary The column index of the primary diagonal in the banded matrix.
     * @param leftRight
     */
    __device__ LSetter1d(DeviceData2d<T>& L, size_t rowL, const AdjacencyInd& primary, const AdjacencyIndPair& leftRight) :
        lSetter(L, rowL),
        primary(primary),
        leftRight(leftRight) {

    }


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
     */
    __device__ void setRowInBanded1d(
        DeviceData2d<T>& laplacian,
        const BoundaryPair<T>& boundary
    ) {
        lSetter.laplacian = &laplacian;
        laplacian(lSetter.rowL, primary) = 0;
        lSetter.setRowInBanded1d(lSetter.rowL, boundary, primary, leftRight);
    }
};


/**
 * @brief CUDA kernel to set up the staggered grid Laplacian matrix and apply boundary conditions.
 *
 * @param[in,out] L        Banded system matrix (dim.size() × numDiagonals); coefficients are accumulated.
 * @param[in] dim          Grid dimensions.
 * @param[in] boundary     Boundary conditions for all six faces.
 * @param[in] ap           Adjacency pattern specifying diagonal storage layout.
 */
template<typename T>
__global__ void buildLaplacianKernel(DeviceData2d<T> L, const GridDim dim, const BoundaryConfig<T> boundary, const AdjacencyPatern ap) {
    GridInd3d gridInd;
    if (gridInd >= dim) return;

    size_t rowIndex = dim[gridInd];
    LSetter<T> ds(L, rowIndex);
    L(rowIndex, ap.here) = 0;

    ds.setRowInBanded1d(gridInd.row, boundary.topBottom, ap.here, ap.upDown);
    ds.setRowInBanded1d(gridInd.col, boundary.leftRight, ap.here, ap.leftRight);
    if (dim.layers > 1) ds.setRowInBanded1d(gridInd.layer, boundary.frontBack, ap.here, ap.frontBack);
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

    if (dim.layers > 1 && ind.row < dim.rows && ind.col < dim.cols) {
        ind3d.set(ind.row, ind.col, 0);
        boundary.frontBack[0].setBoundaryRHS(rhs[dim[ind3d]]);
        ind3d.layer = dim.layers - 1;
        boundary.frontBack[1].setBoundaryRHS(rhs[dim[ind3d]]);
    }
    if (ind.row < dim.rows && ind.col < dim.layers) {
        ind3d.set(ind.row, 0, ind.col);
        boundary.leftRight[0].setBoundaryRHS(rhs[dim[ind3d]]);
        ind3d.col = dim.cols - 1;
        boundary.leftRight[1].setBoundaryRHS(rhs[dim[ind3d]]);
    }
    if (ind.row < dim.layers && ind.col < dim.cols) {
        ind3d.set(0, ind.col, ind.row);
        boundary.topBottom[0].setBoundaryRHS(rhs[dim[ind3d]]);
        ind3d.row = dim.rows - 1;
        boundary.topBottom[1].setBoundaryRHS(rhs[dim[ind3d]]);
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
 * @param[in] prevNext            The indices of the previouse and next element.

 */
template <typename T>
__global__ void buildL1dKernel(
    DeviceData2d<T> bandedL_i,
    const BoundaryPair<T> condition,
    const AdjacencyInd primary,
    const AdjacencyIndPair prevNext
) {
    size_t i = idx();
    if (i >= bandedL_i.rows) return;

    LSetter<T> ds(bandedL_i, i);
    ds.setRowInBanded1d(i, bandedL_i.rows, condition, primary, prevNext);
}

template <typename T>
__global__ void buildAllL1dKernel(XYZ<DeviceData2d<T>> bandedL, const BoundaryConfig<T> boundary, const AdjacencyInd primary, const AdjacencyIndPair prevNext) {
    size_t i = idx();

    LSetter1d<T> ds(bandedL.x, i, primary, prevNext);

    if (i < bandedL.x.rows) ds.setRowInBanded1d(bandedL.x, boundary.leftRight);
    if (i < bandedL.y.rows) ds.setRowInBanded1d(bandedL.y, boundary.topBottom);
    if (bandedL.z.size() > 1 && i < bandedL.z.rows) ds.setRowInBanded1d(bandedL.z, boundary.frontBack);
}



#endif //CUDABANDED_LAPLACIANKERNELS_CUH
