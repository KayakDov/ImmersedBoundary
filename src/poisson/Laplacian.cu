//
// Created by usr on 12/24/25.
//

#include "poisson/Laplacian.cuh"

#include <vector>

#include "deviceArrays/headers/Support/Streamable.h"
#include "math/Real3dDevice.hpp"



template<typename T>
std::unique_ptr<BandedMat<T>> & Laplacian1dManager<T>::operator[](size_t dim) {
    switch (dim) {
        case 0: return Lx;
        case 1: return Ly;
        case 2: return Lz;
        default: throw std::out_of_range("Invalid dimension: must be 0, 1, or 2");
    }
}

template<typename T>
Laplacian<T>::Laplacian(const GridDim& dim, const Real3d& delta, const BoundaryConfig<T>& boundary) :
    dim(dim),
    delta(delta),
    boundary(boundary),
    adjacncies(dim) {
}

template <typename T>
T invSq(T x) {
    return 1/(x*x);
}

void AdjacencyPatern::loadMapRowToDiag(Vec<int32_t>& diags, const cudaStream_t stream) const{
    loadMapRowToDiag(diags, {here, up, down, left, right, front, back}, stream);
}

void AdjacencyPatern::loadMapRowToDiag(Vec<int32_t> &diags, std::vector<AdjacencyInd> indices, cudaStream_t stream) {
    std::vector<int32_t> diagsCpu(diags.size(), 0);
    for (AdjacencyInd ind : indices) diagsCpu[ind.col] = ind.diag;
    diags.set(diagsCpu.data(), stream);
}

template<typename T>
void Laplacian<T>::setOperation(cudaStream_t stream) {
    size_t numDiags =  dim.layers > 1 ? numDiagonals3d : numDiagonals2d;
    Vec<int32_t> preAllocatedForIndices = Vec<int32_t>::create(numDiags, stream);
    Mat<T> rhsAndL = Mat<T>::create(dim.size(), numDiags + 1);
    Mat<T> preAllocatedForL = rhsAndL.col(numDiags);
    Vec<T> rhsModifier = preAllocatedForL.col(rhsAndL.subMat(0,0,dim.size(), numDiags));

    setOperation(stream, preAllocatedForL, preAllocatedForIndices, rhsModifier);
}

template<typename T>
void Laplacian<T>::setOperation(cudaStream_t stream, Mat<T> &preAllocatedForL, Vec<int32_t> &preAllocatedForIndices, Vec<T>& rhsModifier) {
    KernelPrep kp = this->dim.kernelPrep();
    buildLaplacianKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAllocatedForL,
        this->dim, this->boundary,
        this->adjacncies,
        rhsModifier
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    this->adjacncies.loadMapRowToDiag(preAllocatedForIndices, stream);

    bandedL = std::make_unique<BandedMat<T>>(preAllocatedForL, preAllocatedForIndices);
    rhsBC = std::make_unique<Vec<T>>(rhsModifier);
}

template<typename T>
void Laplacian<T>::setL_i(cudaStream_t stream, SquareMat<T> &preAllocatedForL_i, Vec<int32_t> &preAllocatedForIndices, size_t dim) {

    AdjacencyInd prev(1, -1), next(2, 1), primary(0, 0);

    KernelPrep kp(dim);
    CHECK_CUDA_ERROR(
        buildL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
            preAllocatedForL_i, this->boundary(dim, true), this->boundary(dim, false), primary, prev, next
        )
    );

    AdjacencyPatern::loadMapRowToDiag(preAllocatedForIndices, {primary, prev, next}, stream);

    _1d[dim] = std::make_unique<BandedMat<T>>(preAllocatedForL_i, preAllocatedForIndices);
}

template<typename T>
void Laplacian<T>::setL_iAll(cudaStream_t stream, SquareMat<T>* preAllocatedForL_iX3, Vec<int32_t> &preAllocatedForIndices) {

    AdjacencyInd prev(1, -1), next(2, 1), primary(0, 0);

    KernelPrep kp(this->dim.maxDim());

    CHECK_CUDA_ERROR(buildAllL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAllocatedForL_iX3[0], preAllocatedForL_iX3[1], preAllocatedForL_iX3[2],
        this->boundary,
        primary, prev, next
    ));

    AdjacencyPatern::loadMapRowToDiag(preAllocatedForIndices, {primary, prev, next}, stream);

    for (size_t i = 0; i < 3; i++)
        _1d[i] = std::make_unique<BandedMat<T>>(preAllocatedForL_iX3[i], preAllocatedForIndices);
}

template<typename T>
void Laplacian<T>::setL_i(cudaStream_t stream, size_t dim) {
    Vec<int32_t> preAllocatedForIndices = Vec<int32_t>::create(3, stream);
    Mat<T> preAllocatedForL_i = SquareMat<T>::create(this->dim[dim], 3);
    setL_i(stream, preAllocatedForL_i, preAllocatedForIndices, dim);
}

template<typename T>
void Laplacian<T>::setL_iAll(cudaStream_t stream) {
    Vec<int32_t> preAllocatedForIndices = Vec<int32_t>::create(3, stream);
    Mat<T> preAllocatedForL_iX3[3] = {
        SquareMat<T>::create(this->dim[0], 3),
        SquareMat<T>::create(this->dim[1], 3),
        SquareMat<T>::create(this->dim[2], 3)
    };
    setL_iAll(stream, preAllocatedForL_iX3, preAllocatedForIndices);
}

template<typename T>
void Laplacian<T>::setRhsBC(cudaStream_t stream) {
    Vec<T> rhsModifier = SimpleArray<T>::create(dim.size());
    setRhsBC(stream, rhsModifier);
}

template<typename T>
void Laplacian<T>::setRhsBC(cudaStream_t stream, Vec<T>& rhsModifier) {
    KernelPrep kp(std::max(dim.rows, dim.layers), std::max(dim.layers, dim.cols));
    buildRhsBCKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(dim, boundary, rhsModifier);
    CHECK_CUDA_ERROR(cudaGetLastError());
    rhsBC = std::make_unique<Vec<T>>(rhsModifier);
}


template class Laplacian<float>;
template class Laplacian<double>;
