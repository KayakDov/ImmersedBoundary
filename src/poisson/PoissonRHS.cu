
#include "PoissonRHS.h"

#include <iostream>
#include <optional>

#include "../deviceArrays/headers/Support/GridDim.hpp"
#include "../deviceArrays/headers/Support/Streamable.h"


template <typename T>
__global__ void setRHSKernel(DeviceData1d<T> b, const DeviceData2d<T> topBottom, const DeviceData2d<T> leftRight, const DeviceData2d<T> frontBack, const GridDim grid) {

    const GridInd2d ind;

    if (ind < frontBack) {
        const size_t row = ind.row % grid.rows;
        const size_t layer = ind.row >= grid.rows ? grid.layers - 1 : 0;
        const size_t col = ind.col;
        atomicAdd(&b[grid(row, col, layer)], -frontBack[ind]);
    }

    if (ind < leftRight) {
        const size_t row = ind.row % grid.rows;
        const size_t col = ind.row >= grid.rows ? grid.cols - 1 : 0;
        const size_t layer = grid.layers - 1 - ind.col;
        atomicAdd(&b[grid(row, col, layer)], -leftRight[ind]);
    }

    if (ind < topBottom) {
        const size_t row = ind.row >= grid.layers ? grid.rows - 1 : 0;
        const size_t col = ind.col;
        const size_t layer = grid.layers - 1 - (ind.row % grid.layers);
        atomicAdd(&b[grid(row, col, layer)], -topBottom[ind]);
    }
}


template <typename T>
PoissonRHS<T>::PoissonRHS(const CubeBoundary<T>& boundary, Vec<T>& f, const cudaStream_t stream):_b(f), dim(boundary.frontBack._rows/2, boundary.frontBack._cols, boundary.topBottom._rows/2){

    cudaDeviceSynchronize();

    KernelPrep kp(std::max(dim.cols, dim.layers), 2 * std::max(dim.rows, dim.layers));
    setRHSKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        f.toKernel1d(),
        boundary.topBottom.toKernel2d(),
        boundary.leftRight.toKernel2d(),
        boundary.frontBack.toKernel2d(),
        dim
        );
    CHECK_CUDA_ERROR(cudaGetLastError());

    cudaDeviceSynchronize();
}


template class PoissonRHS<float>;
template class PoissonRHS<double>;






// template <typename T>
// __device__ T bVal(const GridInd3d& ind, const GridDim& g, const DeviceCubeBoundary<T>& dcb) {
//     size_t distFromBack = g.layers - 1 - ind.layer;
//     return (ind.row == 0             ? dcb.bottom(distFromBack, ind.col) : 0)
//         + (ind.row == g.rows - 1     ? dcb.top(distFromBack, ind.col)    : 0)
//         + (ind.col == 0              ? dcb.left(ind.row, distFromBack)   : 0)
//         + (ind.col == g.cols - 1     ? dcb.right(ind.row, distFromBack)  : 0)
//         + (ind.layer == 0            ? dcb.front(ind.row, ind.col)           : 0)
//         + (ind.layer == g.layers - 1 ? dcb.back(ind.row, ind.col)            : 0);
// }
//
// template <typename T>
// __global__ void setRHSKernel3D(DeviceData1d<T> b, const DeviceCubeBoundary<T> dcb, GridDim grid) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//     size_t bound;
//     GridInd3d ind;
//     if (idx < (bound = 2 * dcb.front.size())) {
//         size_t layer;
//         if (idx >= dcb.front.size()) {
//             idx -= dcb.front.size();
//             layer = grid.layers - 1;
//         } else layer = 0;
//         ind = GridInd3d(dcb.front.row(idx), dcb.front.col(idx), layer);
//     }
//     else if ( (idx -= bound) < (bound = 2 * dcb.left.size())) {
//         size_t col;
//         if (idx >= dcb.left.size()) {
//             idx -= dcb.left.size();
//             col = grid.cols - 1;
//         } else col = 0;
//
//         ind = GridInd3d(dcb.left.row(idx), col, dcb.left.col(idx));
//     }
//     else if ((idx -= bound) < (bound = 2* dcb.top.size())) {
//         size_t row;
//         if (idx >= dcb.top.size()) {
//             idx -= dcb.top.size();
//             row = 0;
//         } else row = grid.rows - 1;
//         ind = GridInd3d(row, dcb.top.col(idx), dcb.top.row(idx));
//     }
//     else return;
//     printf("idx = %d, index (row, col, layer) = (%lu, %lu, %lu)\n", blockIdx.x * blockDim.x + threadIdx.x, ind.row, ind.col, ind.layer);
//     b[grid[ind]] -= bVal(ind, grid, dcb);
// }