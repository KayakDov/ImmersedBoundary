
#ifndef CUDABANDED_LAPLACIANKERNELS_CUH
#define CUDABANDED_LAPLACIANKERNELS_CUH
#include "deviceArrays/headers/DeviceData.cuh"
#include "poisson/Laplacian1d.cuh"



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
    template<typename AxisSegmentT>
    __device__ void setRowInBanded1d(
        const size_t indexInLine,
        const AxisSegmentT& boundaries,
        const AdjacencyInd& primary, const AdjacencyIndPair& leftRight
    ) {
        T& mainDiag = (*laplacian)[primary.bandedInd(rowL)];
        T& rightDiag = (*laplacian)[leftRight.right.bandedInd(rowL)];
        T& leftDiag = (*laplacian)[leftRight.left.bandedInd(rowL)];

        if (indexInLine == 0) {
            if (rowL + leftRight.left.diag < laplacian->rows) leftDiag = 0;
            boundaries.start.setL(mainDiag, rightDiag);
        } else if (indexInLine == boundaries.numNodes - 1) {
            if (rowL + leftRight.right.diag < laplacian->rows) rightDiag = 0;
            boundaries.end.setL(mainDiag, leftDiag);

        } else boundaries.setInteriorL(mainDiag, leftDiag, rightDiag, indexInLine);

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
    template<typename SegmentType>
    __device__ void setRowInBanded1d(
        DeviceData2d<T>& laplacian,
        const SegmentType& boundary
    ) {
        lSetter.laplacian = &laplacian;
        laplacian[primary.bandedInd(lSetter.rowL)] = 0;
        lSetter.setRowInBanded1d(lSetter.rowL, boundary, primary, leftRight);
    }
};

template<typename T, typename BoundaryConfigT>
__global__ void buildLaplacianKernel(DeviceData2d<T> bandedL, const GridDim dim, const BoundaryConfigT boundary, const AdjacencyPatern ap) {
    GridInd3d gridInd;
    if (gridInd >= dim) return;

    size_t rowIndex = dim[gridInd];

    LSetter<T> ds(bandedL, rowIndex);
    bandedL[ap.here.bandedInd(rowIndex)] = 0;

    size_t n = dim.size();

    ds.setRowInBanded1d(gridInd.row, boundary.y, ap.here, ap.y);
    ds.setRowInBanded1d(gridInd.col, boundary.x, ap.here, ap.x);
    if (dim.layers > 1) ds.setRowInBanded1d(gridInd.layer, boundary.z, ap.here, ap.z);
}


template<typename T, typename BoundaryConfigT>
__global__ void buildRhsBoundaryCorrectionKernel(const GridDim dim, const BoundaryConfigT boundary, DeviceData1d<T> rhs) {
    GridInd2d ind;
    GridInd3d ind3d(0, 0, 0);

    if (dim.layers > 1 && ind.row < dim.rows && ind.col < dim.cols) {
        ind3d.set(ind.row, ind.col, 0);
        boundary.z.start.setBoundaryRHS(rhs[dim[ind3d]]);
        ind3d.layer = dim.layers - 1;
        boundary.z.end.setBoundaryRHS(rhs[dim[ind3d]]);
    }
    if (ind.row < dim.rows && ind.col < dim.layers) {
        ind3d.set(ind.row, 0, ind.col);
        boundary.x.start.setBoundaryRHS(rhs[dim[ind3d]]);
        ind3d.col = dim.cols - 1;
        boundary.x.end.setBoundaryRHS(rhs[dim[ind3d]]);
    }
    if (ind.row < dim.layers && ind.col < dim.cols) {
        ind3d.set(0, ind.col, ind.row);
        boundary.y.start.setBoundaryRHS(rhs[dim[ind3d]]);
        ind3d.row = dim.rows - 1;
        boundary.y.end.setBoundaryRHS(rhs[dim[ind3d]]);
    }
}


template <typename T, typename AxisSegmentT>
__global__ void buildL1dKernel(
    DeviceData2d<T> bandedL_i,
    const AxisSegmentT condition,
    const AdjacencyInd primary,
    const AdjacencyIndPair prevNext
) {
    size_t i = idx();
    if (i >= bandedL_i.rows) return;

    LSetter<T> ds(bandedL_i, i);
    ds.setRowInBanded1d(i,  condition, primary, prevNext);
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

template <typename T, typename BoundaryConfigT>
__global__ void buildAllL1dKernel(XYZ<DeviceData2d<T>> bandedL, const BoundaryConfigT boundary, const AdjacencyInd primary, const AdjacencyIndPair prevNext) {
    size_t i = idx();

    LSetter1d<T> ds(bandedL.x, i, primary, prevNext);


    if (i < bandedL.x.rows) ds.setRowInBanded1d(bandedL.x, boundary.x);
    if (i < bandedL.y.rows) ds.setRowInBanded1d(bandedL.y, boundary.y);
    if (i < bandedL.z.rows) ds.setRowInBanded1d(bandedL.z, boundary.z);
}

template <typename T>
__global__ void setSymetrizationMatrix(XYZ<DeviceData1d<T>> symnetrizationBand, XYZ<DeviceData1d<T>> inv, XYZ<Delta1d<T>> delta) {
    size_t i = idx();
    for (int32_t dim = 0; dim < 3; ++dim)
        if (dim < symnetrizationBand[dim]._cols) {
            symnetrizationBand[dim][i] = sqrt(delta[dim][i] + delta[dim][i + 1]);
            inv[dim][i] = 1/symnetrizationBand[dim][i];
        }
}


#endif //CUDABANDED_LAPLACIANKERNELS_CUH
