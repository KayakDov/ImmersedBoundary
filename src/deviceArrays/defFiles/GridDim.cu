
#include "../headers/GridDim.cuh"

__device__ GridInd2d::GridInd2d(const size_t row, const size_t col): row(row), col(col) {
}

__device__ GridInd3d::GridInd3d(size_t row, size_t col, size_t layer): GridInd2d(row, col), layer(layer) {
}

__device__ __host__ GridDim::GridDim(const size_t height, const size_t width, const size_t depth): cols(width), rows(height), layers(depth), layerSize(rows * cols) {}

__device__ __host__ size_t GridDim::size() const {
    return layers * layerSize;
}

__device__ size_t GridDim::operator()(const size_t row, const size_t col, const size_t layer) const {
    return layer * rows + col * rows * layers + row;
}

__device__ GridInd3d GridDim::operator[](size_t idx) const {
    return {idx % rows, (idx / rows) % cols, idx/(layerSize)};
}

__device__ size_t GridDim::operator[](const GridInd3d& ind) const {
    return this->operator()(ind.row, ind.col, ind.layer);
}

KernelPrep GridDim::kernelPrep() const {
    return {cols, rows, layers};
}

__device__ bool operator<(const GridInd3d &ind, const GridDim &dim) {
    return ind.row < dim.rows && ind.col < dim.cols && ind.layer < dim.layers;
}

__device__ bool operator>=(const GridInd3d &ind, const GridDim &dim) {
    return !(ind < dim);
}
