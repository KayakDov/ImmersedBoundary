//
// Created by usr on 12/24/25.
//

#ifndef CUDABANDED_POISSONLHS_H
#define CUDABANDED_POISSONLHS_H
#include <array>
#include <cuda_runtime.h>
#include "BandedMat.h"
#include "math/Real3d.h"

constexpr size_t numDiagonals3d = 7;
constexpr size_t numDiagonals2d = 5;

struct AdjacencyInd {//TODO: Account for distance between grid points not equal to 1.
    /**
     * The column in the banded matrix.
     */
    const size_t col;
    /**
     * The index of the diagonal that is held by that column.
     */
    const int32_t diag;
    __device__ __host__ AdjacencyInd(const size_t col, const int32_t diag) : col(col), diag(diag) {
    }
};

template<typename T>
class ToeplitzLaplacian {
    const std::array<AdjacencyInd, numDiagonals3d> adjInds; //here, up, down, left, right, back, front;
    const GridDim dim;

    void loadMapRowToDiag(Vec<int32_t> diags, cudaStream_t stream);

public:
    /**
     * Creates the LHS matrix of the linear system used for solving the Poisson equation.
     * @param dim The dimensions of the Poisson grid.
     */
    ToeplitzLaplacian(GridDim dim);

    /**
     * Sets the values into the laplacian
     * @param stream
     * @param preAlocatedForA This matrix should be height * width * depth X (5 if 2d grid or 7 if 3d grid).
     * The laplacian will be placed here.
     * @param preAlocatedForIndices This vector will store the indices of the diagonals in A.
     * If the grid is 2d there should be 5 values here, if the grid is 3d there should be 7.
     * @return
     */
    BandedMat<T> setA(cudaStream_t stream, Mat<T> &preAlocatedForA, Vec<int32_t> &preAlocatedForIndices, const Real3d& delta = Real3d(1, 1, 1));
};

#endif //CUDABANDED_POISSONLHS_H
