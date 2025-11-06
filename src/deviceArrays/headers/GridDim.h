//
// Created by dov on 11/4/25.
//

#ifndef BICGSTAB_GRIDDIM_H
#define BICGSTAB_GRIDDIM_H


class GridInd2d {
public:
    const size_t row, col;
    __device__ GridInd2d(size_t row, size_t col);
    __device__ GridInd2d();
};

class GridInd3d : public GridInd2d{
public:
    const size_t layer;
    __device__ GridInd3d(size_t row, size_t col, size_t layer);
    __device__ GridInd3d();
    
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
     * The flat index for the row, column, and height.
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
