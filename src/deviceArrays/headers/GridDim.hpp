//
// Created by dov on 11/4/25.
//

#ifndef BICGSTAB_GRIDDIM_H
#define BICGSTAB_GRIDDIM_H
#include "KernelPrep.cuh"

//I have no idea why I need this.  TODO:delete this if everything seems to be working.
// #ifndef __CUDACC__
// #define __host__
// #define __device__
// #define __global__
// #endif


#include <cstddef>
#include <device_launch_parameters.h>

#include "math/Real3d.h"

__device__ inline size_t idx(){return blockIdx.x * blockDim.x + threadIdx.x;}
__device__ inline size_t idy(){return blockIdx.y * blockDim.y + threadIdx.y;}

class GridInd2d {
public:
    size_t row, col;
    __device__ inline GridInd2d(size_t row, size_t col):row(row), col(col){}

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
    __device__ inline GridInd2dT(): GridInd2d(idx(), idy()){}
};

class GridInd3d : public GridInd2d{
public:
    size_t layer;
    __device__ inline GridInd3d(size_t row, size_t col, size_t layer): GridInd2d(row, col), layer(layer) {
    }
    __device__ inline GridInd3d(): GridInd2d(), layer(blockIdx.z * blockDim.z + threadIdx.z){}
};

class GridDim {
public:
    const size_t rows, cols, layers, layerSize;
    __host__ __device__ inline  GridDim(size_t height, size_t width, size_t depth): cols(width), rows(height), layers(depth), layerSize(rows * cols) {}

    [[nodiscard]] size_t maxDim() const {
        return std::max(std::max(rows, cols),  layers);
    }
    [[nodiscard]] size_t minDim() const {
        return std::min(std::min(rows, cols),  layers);
    }

    [[nodiscard]] size_t numDims() const {
        return 2 + (layers > 1);
    }
    [[nodiscard]] size_t volume() const {
        return rows * cols * layers;
    }

    /**
     *
     * @return The number of points in the grid.
     */
    __host__ __device__ [[nodiscard]] inline size_t size() const {
        return layers * layerSize;
    }

    /**
     * The flat index for the row, column, and height.  This does not account for leading dimension.
     * @param row
     * @param col
     * @param layer
     * @return
     */
    __device__ size_t inline operator()(size_t row, size_t col, size_t layer) const{
        return layer * rows + col * rows * layers + row;
    }

    __device__ [[nodiscard]] inline GridInd3d operator[](size_t idx) const{
        return {idx % rows, (idx / rows) % cols, idx/(layerSize)};
    }

    /**
     * The flat index of this GridInd.
     * @param ind
     * @return
     */
    __device__ [[nodiscard]] size_t inline operator[](const GridInd3d& ind) const{
        return this->operator()(ind.row, ind.col, ind.layer);
    }


    /**
     *
     * @return kernel threading for every node of this grid.
     */
    [[nodiscard]]KernelPrep inline kernelPrep() const{
        return {cols, rows, layers};
    }

    [[nodiscard]] Real3d center() const{
        return {cols/2.0, rows/2.0, layers/2.0};
    }
};

inline std::ostream& operator<<(std::ostream& os, const GridDim& g) {
    return os << "(rows, cols, layers) = (" << g.rows << ", " << g.cols << ", " << g.layers << ")";
}

/**
 * Are the indices within (non inclusive) the dimensions.
 * @param ind
 * @param dim
 * @return
 */
__device__ bool inline operator<(const GridInd3d& ind, const GridDim& dim){
    return ind.row < dim.rows && ind.col < dim.cols && ind.layer < dim.layers;
}

/**
 * Are the indices outside the dimensions (inclusive).
 * @param ind
 * @param dim
 * @return
 */
__device__ bool inline operator>=(const GridInd3d& ind, const GridDim& dim){
    return !(ind < dim);
}


#endif //BICGSTAB_GRIDDIM_H
