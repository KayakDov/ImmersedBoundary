
#include "Poisson.h"

#include <optional>

#include "../deviceArrays/headers/GridDim.cuh"

//TODO: half can be passed as a paramater.  There also seems to be a lot of redundant code between methods.
template <typename T>
__global__ void setRHSKernelFrontBack(DeviceData1d<T> b, const DeviceData2d<T> faces, const GridDim grid) {
    GridInd2d ind;
    if (ind >= faces) return;

    const size_t half = faces.rows / 2;

    // printf("front back: (%lu, %lu) -> (%lu, %lu, %lu)\n", blockIdx.y * blockDim.y + threadIdx.y, ind.col, ind.row, ind.col, layer);

    b[grid(ind.row % half, ind.col, ind.row >= half ? grid.layers - 1 : 0)] -= faces[ind];
}
template <typename T>
__global__ void setRHSKernelLeftRight(DeviceData1d<T> b, const DeviceData2d<T> faces, const GridDim grid) {
    GridInd2d ind;
    if (ind >= faces) return;

    const size_t half = faces.rows / 2;

    // printf("left right: (%lu, %lu) -> (%lu, %lu, %lu)\n", blockIdx.y * blockDim.y + threadIdx.y, ind.col, ind.row, col, grid.layers - 1 - ind.col);

    b[grid(ind.row % half, ind.row >= half ? grid.cols - 1 : 0, grid.layers - 1 - ind.col)] -= faces[ind];
}
template <typename T>
__global__ void setRHSKernelTopBottom(DeviceData1d<T> b, const DeviceData2d<T> faces, const GridDim grid) {
    GridInd2d ind;
    if (ind >= faces) return;

    const size_t half = faces.rows / 2;

    // printf("top botton: (%lu, %lu) -> (%lu, %lu, %lu)\n", blockIdx.y * blockDim.y + threadIdx.y, ind.col, row, ind.col, grid.layers - 1 - ind.row);

    b[grid(ind.row >= half ? grid.rows - 1 : 0, ind.col, grid.layers - 1 - (ind.row % half))] -= faces[ind];
}


template <typename T>
void Poisson<T>::setB(const CubeBoundary<T>& boundary, cudaStream_t stream) {

    KernelPrep kpTB = boundary.topBottom.kernelPrep();
    setRHSKernelTopBottom<<<kpTB.gridDim, kpTB.blockDim, 0, stream>>>(_b.toKernel1d(), boundary.topBottom.toKernel2d(), dim);
    CHECK_CUDA_ERROR(cudaGetLastError());

    KernelPrep kpLR = boundary.leftRight.kernelPrep();
    setRHSKernelLeftRight<<<kpLR.gridDim, kpLR.blockDim, 0, stream>>>(_b.toKernel1d(), boundary.leftRight.toKernel2d(), dim);
    CHECK_CUDA_ERROR(cudaGetLastError());

    KernelPrep kpFB = boundary.frontBack.kernelPrep();
    setRHSKernelFrontBack<<<kpFB.gridDim, kpFB.blockDim, 0, stream>>>(_b.toKernel1d(), boundary.frontBack.toKernel2d(), dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename T>
size_t Poisson<T>::size() const {
    return dim.size();
}


template <typename T>
Poisson<T>::Poisson(const CubeBoundary<T>& boundary, Vec<T>& f, const cudaStream_t stream) :
    dim(boundary.frontBack._rows/2, boundary.frontBack._cols, boundary.topBottom._rows/2),
    _b(f) {
    setB(boundary, stream);
}


template class Poisson<float>;
template class Poisson<double>;






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