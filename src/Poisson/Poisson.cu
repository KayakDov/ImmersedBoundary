
#include "Poisson.h"

#include <optional>

#include "../deviceArrays/headers/GridDim.cuh"

/**
 * @brief Map a linear index to coordinates on the front or back faces of the 3D grid.
 *
 * This function determines the (layer, row, col) coordinates corresponding
 * to the front (layer = 0) or back (layer = depth - 1) boundary faces.
 *
 * @param[out] layer The computed depth index (0 or depth - 1).
 * @param[out] row The computed row index in the grid.
 * @param[out] col The computed column index in the grid.
 * @param[in] height The interior grid height (without boundary).
 * @param[in] width The interior grid width (without boundary).
 * @param[in] depth The interior grid depth (without boundary).
 * @param[in] idx The linear thread index, offset for front/back faces.
 *
 */
__device__ GridInd3d setIndicesFrontBackFaces(const GridDim& g, size_t idx) {
    return {idx % g.rows, (idx / g.rows) % g.cols, idx < g.layerSize ? 0 : g.layers - 1};
}

/**
 * @brief Map a linear index to coordinates on the left or right faces of the 3D grid.
 *
 * This function determines the (layer, row, col) coordinates corresponding
 * to the left (col = 0) or right (col = width - 1) boundary faces.
 *
 * The function excludes overlap with front/back faces by checking if
 * the layer is at the boundary (0 or depth - 1). In those cases, it returns false.
 *
 * @param success will be set to true if the indices are set, and false otherwise.
 * @param[in] idx The linear thread index, offset for left/right faces.
 * @return true if the index corresponds to a valid left/right face location, false otherwise.
 */
__device__ GridInd3d setIndicesLeftRightFaces(const GridDim& g, bool& success, size_t idx) {
    idx -= 2 * g.layerSize;
    const size_t layer = g.layers - 1 - (idx / g.rows) % g.layers;
    if ((success = layer < g.layers - 1 && layer > 0))
        return {idx % g.rows, idx < g.rows * g.layers ? 0 : g.cols - 1, layer};

}
/**
 * @brief Map a linear index to coordinates on the top or bottom faces of the 3D grid.
 *
 * This function determines the (layer, row, col) coordinates corresponding
 * to the top (row = 0) or bottom (row = height - 1) boundary faces.
 *
 * The function excludes overlap with front/back and left/right faces by checking
 * if the layer is at the boundary (0 or depth - 1) or if the column is at the
 * left/right boundary (0 or width - 1). In those cases, it returns false.
 *
 * @param success will be set to true if an index is returned, and false otherwise.
 * @param[in] idx The linear thread index, offset for top/bottom faces.
 * @return true if the index corresponds to a valid top/bottom face location, false otherwise.
 */
__device__ GridInd3d setIndicesTopBottomFaces(const GridDim& g, bool& success, size_t idx) {
    idx -= 2 * g.layerSize + 2 * g.rows * g.layers;
    const size_t layer = g.layers - 1 - (idx % g.layers);
    const size_t col = (idx / g.layers) % g.cols;
    if ((success = layer < g.layers - 1 && layer > 0 && col < g.cols - 1 && col > 0))
        return {idx < g.layers * g.cols ? 0 : g.rows - 1, col, layer};

}

/**
 * Computes the value to set b to.
 * @tparam T The type of data
 * @param ind The index in the grid.
 * @param g The dimensions of the grid.
 * @param topBottom The boundary.
 * @param leftRight The boundary.
 * @param frontBack The boundary.
 * @return the new value to be placed in b.
 */
template <typename T>
__device__ T bVal(const GridInd3d& ind, const GridDim& g, const DeviceData2d<T>& topBottom, const DeviceData2d<T>& leftRight, const DeviceData2d<T>& frontBack) {
    return (ind.row == 0            ? topBottom(g.layers - 1 - ind.layer, ind.col)           : 0)
        + (ind.row == g.rows - 1  ? topBottom(g.layers - 1 - ind.layer + g.layers, ind.col)  : 0)
        + (ind.col == 0            ? leftRight(ind.row, g.layers - 1 - ind.layer)           : 0)
        + (ind.col == g.cols - 1   ? leftRight(ind.row + g.rows, g.layers -1 - ind.layer)  : 0)
        + (ind.layer == 0          ? frontBack(ind.row, ind.col)                              : 0)
        + (ind.layer == g.layers - 1 ? frontBack(ind.row + g.rows, ind.col)                    : 0);
}

/**
 * @brief CUDA kernel to apply boundary conditions to the right-hand side of a 3D Poisson problem.
 *
 * This kernel updates the RHS vector `b` with contributions from the six boundary faces
 * (front, back, left, right, top, bottom) of the 3D domain. Each thread corresponds to one
 * boundary element and adds its boundary contribution to the correct position in `b`.
 *
 * @tparam T Floating-point type (float or double).
 * @param[out] b The right-hand side vector of the linear system. Stored in column-major order.
 *               Initially holds the RHS for the interior, and is incremented by boundary contributions.
 * @param[in] frontBack Concatenated arrays holding the boundary values for the front (layer = 0)
 *                      and back (layer = depth - 1) faces.
 * @param fbSize The size of the front plus the size of the back is the size of frontBack

 * @param[in] leftRight Concatenated arrays holding the boundary values for the left (col = 0)
 *                      and right (col = width - 1) faces.
 * @param lrAndFbSize The sum of the sizes of the front, back, left, and right.
 * @param[in] topBottom Concatenated arrays holding the boundary values for the top (row = 0)
 *                      and bottom (row = height - 1) faces.
 * @param boundarySize The size of all the faces.
 * @param grid the dimensions of the grid
 */
template <typename T>
__global__ void setRHSKernel3D(DeviceData1d<T> b,
                               const DeviceData2d<T> frontBack, const size_t fbSize,
                               const DeviceData2d<T> leftRight, const size_t lrAndFbSize,
                               const DeviceData2d<T> topBottom, const size_t boundarySize,
                               const GridDim grid) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool success;

    if (idx < fbSize) {
        GridInd3d ind = setIndicesFrontBackFaces(grid, idx);
        b[grid[ind]] -= bVal(ind, grid, topBottom, leftRight, frontBack);
    }
    else if (idx < lrAndFbSize) {
        auto ind = setIndicesLeftRightFaces(grid, success, idx);
        if (success) b[grid[ind]] -= bVal(ind, grid, topBottom, leftRight, frontBack);
    }
    else if (idx < boundarySize){
        auto ind = setIndicesTopBottomFaces(grid, success, idx);
        if (success)
            b[grid[ind]] -= bVal(ind, grid, topBottom, leftRight, frontBack);
    }
}

template <typename T>
size_t Poisson<T>::size() const {
    return dim.size();
}

template <typename T>
void Poisson<T>::setB(const CubeBoundary<T>& boundary, cudaStream_t stream) {

    KernelPrep kp(boundary.size());

    setRHSKernel3D<<<kp.gridDim, kp.blockDim, 0, stream>>>(
        _b.toKernel1d(),
        boundary.frontBack.toKernel2d(), boundary.frontBack.size(),
        boundary.leftRight.toKernel2d(), boundary.frontBack.size() + boundary.leftRight.size(),
        boundary.topBottom.toKernel2d(), boundary.size(),
        dim);

    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename T>
Poisson<T>::Poisson(const CubeBoundary<T>& boundary, Vec<T>& f, const cudaStream_t stream) :
    dim(boundary.frontBack._rows/2, boundary.frontBack._cols, boundary.topBottom._rows/2),
    _b(f) {
    setB(boundary, stream);
}


template class Poisson<float>;
template class Poisson<double>;