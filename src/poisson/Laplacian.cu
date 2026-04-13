//
// Created by usr on 12/24/25.
//

#include "poisson/Laplacian.cuh"

#include <vector>

#include "deviceArrays/headers/Support/Streamable.h"

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
 * @param dim The dimesnsions of the grid.
 * @param up
 * @param down
 * @param left
 * @param right
 * @param set0
 */
template<typename T>
__device__ void setA2d(DeviceData2d<T> a, const GridInd3d &ind, const GridDim &dim, const AdjacencyPatern &ap, Set0<T>& set0) {

    if (ind.row == 0) set0(ap.up);
    else if (ind.row == dim.rows - 1) set0(ap.down);

    if (ind.col == 0) set0(ap.left);
    else if (ind.col == dim.cols - 1) set0(ap.right);
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
__global__ void setAKernel2d(DeviceData2d<T> a, const GridDim g, const AdjacencyPatern ap) {
    const GridInd3d ind;

    if (ind >= g) return;
    const size_t idGrid = g[ind];
    Set0<T> set0(a, idGrid);

    setA2d(a, ind, g, ap, set0);
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
__global__ void setAKernel3d(DeviceData2d<T> a, const GridDim g, AdjacencyPatern ap) {
    const GridInd3d ind;

    if (ind >= g) return;

    const size_t idGrid = g[ind];
    Set0<T> set0(a, idGrid);

    setA2d(a, ind, g, ap, set0);

    if (ind.layer == 0) set0(ap.front);
    else if (ind.layer == g.layers - 1) set0(ap.back);
}

AdjacencyPatern::AdjacencyPatern(GridDim dim):
    here(0, 0),
    up(1, -1),
    down(2, 1),
    left(3, -dim.rows * dim.layers),
    right(4, dim.rows * dim.layers),
    front (5, -dim.rows),
    back(6, dim.rows)
    {

}

template<typename T>
Laplacian<T>::Laplacian(GridDim dim, Real3d delta) :
    dim(dim),
    delta(delta),
    adjacncies(dim) {
}

template <typename T>
T invSq(T x) {
    return 1/(x*x);
}

template<typename T>
LaplacianNodeCentered<T>::LaplacianNodeCentered(GridDim dim, Real3d delta) : Laplacian<T>(dim, delta) {
}

void AdjacencyPatern::loadMapRowToDiag(Vec<int32_t> diags, const cudaStream_t stream) {
    std::vector<int32_t> diagsCpu(diags.size(), 0);
    diagsCpu[here.col] = here.diag;
    diagsCpu[up.col] = up.diag;
    diagsCpu[down.col] = down.diag;
    diagsCpu[left.col] = left.diag;
    diagsCpu[right.col] = right.diag;
    if (diagsCpu.size() > numDiagonals2d) {
        diagsCpu[front.col] = front.diag;
        diagsCpu[back.col] = back.diag;
    }
    diags.set(diagsCpu.data(), stream);
}

BoundaryConfig::BoundaryConfig(BCType left, BCType right, BCType top, BCType bottom, BCType front, BCType back):
    left(left), right(right), top(top), bottom(bottom), front(front), back(back) {
}


template<typename T>
BandedMat<T> LaplacianNodeCentered<T>::setL(cudaStream_t stream, Mat<T> &preAlocatedForA, Vec<int32_t> &preAlocatedForIndices) {

    T denDx2 = invSq(this->delta.x), denDy2 = invSq(this->delta.y), denDz2 = invSq(this->delta.z);

    preAlocatedForA.col(0).fill(-2*(denDx2 + denDy2 + (this->dim.layers > 1 ? denDz2 : 0)), stream);

    preAlocatedForA.subMat(0,1,preAlocatedForA._rows, 2).fill(denDy2, stream);
    preAlocatedForA.subMat(0,3,preAlocatedForA._rows, 2).fill(denDx2, stream);
    if (this->dim.layers > 1) preAlocatedForA.subMat(0,5,preAlocatedForA._rows, 2).fill(denDz2, stream);

    const KernelPrep kp = this->dim.kernelPrep();
    if (this->dim.layers > 1) setAKernel3d<T><<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAlocatedForA.toKernel2d(), this->dim,
        this->adjacncies
    );
    else setAKernel2d<T><<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAlocatedForA.toKernel2d(), this->dim,
        this->adjacncies
    );

    CHECK_CUDA_ERROR(cudaGetLastError());

    this->adjacncies.loadMapRowToDiag(preAlocatedForIndices, stream);

    return BandedMat<T>(preAlocatedForA, preAlocatedForIndices);
}


template<typename T>
BandedMat<T> LaplacianNodeCentered<T>::L(const GridDim &dim, Handle &hand, Real3d delta) {

    auto spaceForA = Mat<T>::create(dim.size(), numDiagonals3d);
    auto inds = SimpleArray<int32_t>::create(numDiagonals3d, hand);
    auto A = LaplacianNodeCentered<T>(dim, delta).setL(hand, spaceForA, inds);
    return A;
}


template<typename T>
void LaplacianNodeCentered<T>::printL(const GridDim &dim, Handle &hand, Real3d delta) {

    auto aDense = SquareMat<T>::create(dim.size());
    auto A = L(dim, hand, delta);
    A.getDense(aDense, &hand);
    std::cout << "L = \n" << GpuOut<T>(aDense, hand) << std::endl;
}

template<typename T>
LaplacianStagared<T>::LaplacianStagared(GridDim dim, Real3d delta) : Laplacian<T>(dim, delta) {}

template<typename T>
BandedMat<T> LaplacianStagared<T>::setL(cudaStream_t stream, Mat<T> &preAlocatedForA, Vec<int32_t> &preAlocatedForIndices) {

}

template class Laplacian<float>;
template class Laplacian<double>;

template class LaplacianNodeCentered<float>;
template class LaplacianNodeCentered<double>;

template class LaplacianStagared<float>;
template class LaplacianStagared<double>;
