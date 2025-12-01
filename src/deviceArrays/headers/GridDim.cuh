//
// Created by dov on 11/4/25.
//

#ifndef BICGSTAB_GRIDDIM_H
#define BICGSTAB_GRIDDIM_H
#include "KernelPrep.cuh"

#ifndef __CUDACC__
#define __host__
#define __device__
#define __global__
#endif


#include <cstddef>
#include <device_launch_parameters.h>

__device__ inline size_t idx(){return blockIdx.x * blockDim.x + threadIdx.x;}
__device__ inline size_t idy(){return blockIdx.y * blockDim.y + threadIdx.y;}

class GridInd2d {
public:
    size_t row, col;
    __device__ GridInd2d(size_t row, size_t col);

    /**
     * The index of the thread.
     */
    __device__ inline GridInd2d(): GridInd2d(idy(), idx()){}
};

/**
 * A transposed 2d index of the thread.
 */
class GridInd2dT : public GridInd2d{
public:
    /**
     * The thread index transposed.  Should be used when the kernel prep is transposed.
     */
    __device__ GridInd2dT(): GridInd2d(idx(), idy()){}
};

class GridInd3d : public GridInd2d{
public:
    size_t layer;
    __device__ GridInd3d(size_t row, size_t col, size_t layer);
    __device__ inline GridInd3d(): GridInd2d(), layer(blockIdx.z * blockDim.z + threadIdx.z){}
};

class GridDim {
public:
    const size_t rows, cols, layers, layerSize;
    __host__ __device__ GridDim(size_t height, size_t width, size_t depth);

    /**
     *
     * @return The number of points in the grid.
     */
    __host__ __device__ [[nodiscard]] size_t size() const;

    /**
     * The flat index for the row, column, and height.  This does not account for leading dimension.
     * @param row
     * @param col
     * @param layer
     * @return
     */
    __device__ size_t operator()(size_t row, size_t col, size_t layer) const;

    __device__ [[nodiscard]] GridInd3d operator[](size_t idx) const;

    /**
     * The flat index of this GridInd.
     * @param ind
     * @return
     */
    __device__ [[nodiscard]] size_t operator[](const GridInd3d& ind) const;


    /**
     *
     * @return kernel threading for every node of this grid.
     */
    [[nodiscard]]KernelPrep kernelPrep() const;

};

/**
 * Are the indices within (non inclusive) the dimensions.
 * @param ind
 * @param dim
 * @return
 */
__device__ bool operator<(const GridInd3d& ind, const GridDim& dim);

/**
 * Are the indices outside the dimensions (inclusive).
 * @param ind
 * @param dim
 * @return
 */
__device__ bool operator>=(const GridInd3d& ind, const GridDim& dim);


#endif //BICGSTAB_GRIDDIM_H
