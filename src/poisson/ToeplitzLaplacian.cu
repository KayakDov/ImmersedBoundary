//
// Created by usr on 12/24/25.
//

#include "ToeplitzLaplacian.cuh"


/**
* @brief Device-side functor to set off-diagonal entries of the system matrix A to 0 or NAN.
*
* This is used inside the setAKernel3d kernel to handle the six neighbors for each interior
* grid point. It ensures that entries corresponding to boundary connections are either
* correctly set to 0 (non-existent internal connection) or marked as NAN (which typically
* signals an element outside the valid band storage).
*
* @tparam T Floating-point type (float or double).
*/
template<typename T>
class Set0 {
private:
    DeviceData2d<T> &a;
    const size_t idGrid;

public:
    /**
     * @brief Constructs the Set0 functor.
     *
     * @param[in,out] a Pointer to the banded matrix data on the device.
     * @param[in] idGrid The flat index of the current grid point (row in A).
     */
    __device__ Set0(DeviceData2d<T> &a, const size_t idGrid) : a(a), idGrid(idGrid) {
    }

    /**
     * @brief Sets the corresponding off-diagonal entry to 0 or NAN based on boundary condition logic.
     *
     * This operator is called to check if a specific off-diagonal entry (corresponding to a neighbor)
     * should be set to 0 (internal point) or NAN (outside band storage).
     *
     * @param[in] aInd The index of the diagonal corresponding to the neighbor being checked.
     */
    __device__ void operator()(const AdjacencyInd aInd) {
        const size_t rowInd = modPos(static_cast<int32_t>(idGrid) + min(aInd.diag, 0), static_cast<int32_t>(a.rows));
        if (rowInd < a.rows - abs(aInd.diag)) a(rowInd, aInd.col) = static_cast<T>(0);
        else a(rowInd, aInd.col) = NAN;
    }
};

/**
 * Sets the laplacian values for 2d.
 * @tparam T
 * @param a The A matrix.
 * @param ind The index of the value to be set.
 * @param g The dimesnsions of the grid.
 * @param up
 * @param down
 * @param left
 * @param right
 * @param set0
 */
template<typename T>
__device__ void setA2d(DeviceData2d<T> a,
                       const GridInd3d &ind,
                       const GridDim &g,
                       const AdjacencyInd &up, const AdjacencyInd &down,
                       const AdjacencyInd &left, const AdjacencyInd &right, Set0<T>& set0) {

    if (ind.row == 0) set0(up);
    else if (ind.row == g.rows - 1) set0(down);

    if (ind.col == 0) set0(left);
    else if (ind.col == g.cols - 1) set0(right);
}

/**
 * @brief CUDA kernel to set up the system matrix A for the 3D Poisson FDM problem.
 *
 * Each thread handles one unknown point $(gRow, gCol, gLayer)$ in the interior grid,
 * setting the main diagonal entry ($A_{i,i} = -6$) and using the Set0 functor to handle
 * the 6 off-diagonal entries (neighbors) and enforce boundary conditions by setting
 * unused band elements to NAN.
 *
 * @tparam T Floating-point type (float or double).
 *
 */
template<typename T>
__global__ void setAKernel2d(DeviceData2d<T> a,
                             const GridDim g,
                             const AdjacencyInd here, const AdjacencyInd up, const AdjacencyInd down,
                             const AdjacencyInd left, const AdjacencyInd right) {
    const GridInd3d ind;

    if (ind >= g) return;
    const size_t idGrid = g[ind];
    Set0<T> set0(a, idGrid);

    setA2d(a, ind, g, up, down, left, right, set0);
}

/**
 * @brief CUDA kernel to set up the system matrix A for the 3D Poisson FDM problem.
 *
 * Each thread handles one unknown point $(gRow, gCol, gLayer)$ in the interior grid,
 * setting the main diagonal entry ($A_{i,i} = -6$) and using the Set0 functor to handle
 * the 6 off-diagonal entries (neighbors) and enforce boundary conditions by setting
 * unused band elements to NAN.
 *
 * @tparam T Floating-point type (float or double).
 *
 */
template<typename T>
__global__ void setAKernel3d(DeviceData2d<T> a,
                             const GridDim g,
                             const AdjacencyInd here, const AdjacencyInd up, const AdjacencyInd down,
                             const AdjacencyInd left, const AdjacencyInd right, const AdjacencyInd front,
                             const AdjacencyInd back
) {
    const GridInd3d ind;

    if (ind >= g) return;

    const size_t idGrid = g[ind];
    Set0<T> set0(a, idGrid);

    setA2d(a, ind, g, up, down, left, right, set0);

    if (ind.layer == 0) set0(front);
    else if (ind.layer == g.layers - 1) set0(back);
}

template<typename T>
ToeplitzLaplacian<T>::ToeplitzLaplacian(GridDim dim) : dim(dim),
                                                       adjInds{
                                                           AdjacencyInd(0, 0),
                                                           AdjacencyInd(1, -1),
                                                           AdjacencyInd(2, 1),
                                                           AdjacencyInd(3, -dim.rows * dim.layers),
                                                           AdjacencyInd(4, dim.rows * dim.layers),
                                                           AdjacencyInd(5, -dim.rows),
                                                           AdjacencyInd(6, dim.rows)
                                                       } {
}

template<typename T>
void ToeplitzLaplacian<T>::loadMapRowToDiag(Vec<int32_t> diags, const cudaStream_t stream) {
    int32_t diagsCpu[numDiagonals3d];
    for (size_t i = 0; i < numDiagonals3d; i++) diagsCpu[adjInds[i].col] = adjInds[i].diag;
    diags.set(diagsCpu, stream);
}

template<typename T>
BandedMat<T> ToeplitzLaplacian<T>::setA(cudaStream_t stream, Mat<T> &preAlocatedForA,
                                        Vec<int32_t> &preAlocatedForIndices, const Real3d& delta) {
    T   denDx2 = 1 / (delta.x * delta.x),
        denDy2 = 1 / (delta.y * delta.y),
        denDz2 = 1 / (delta.z * delta.z);
    preAlocatedForA.col(0).fill(-2*(denDx2 + denDy2 + (dim.layers > 1 ? denDz2 : 0)), stream);
    preAlocatedForA.subMat(0,1,preAlocatedForA._rows, 2).fill(denDy2, stream);
    preAlocatedForA.subMat(0,3,preAlocatedForA._rows, 2).fill(denDx2, stream);
    if (delta.z > 1) preAlocatedForA.subMat(0,5,preAlocatedForA._rows, 2).fill(denDz2, stream);

    const KernelPrep kp = dim.kernelPrep();
    if (dim.layers > 1) setAKernel3d<T><<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAlocatedForA.toKernel2d(), dim,
        adjInds[0], adjInds[1], adjInds[2], adjInds[3], adjInds[4], adjInds[5], adjInds[6]
    );
    else setAKernel2d<T><<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAlocatedForA.toKernel2d(), dim,
        adjInds[0], adjInds[1], adjInds[2], adjInds[3], adjInds[4]
    );

    CHECK_CUDA_ERROR(cudaGetLastError());

    loadMapRowToDiag(preAlocatedForIndices, stream);

    return BandedMat<T>(preAlocatedForA, preAlocatedForIndices);
}

template class ToeplitzLaplacian<float>;
template class ToeplitzLaplacian<double>;
