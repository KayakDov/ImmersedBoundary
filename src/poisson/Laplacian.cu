//
// Created by usr on 12/24/25.
//

#include "poisson/Laplacian.cuh"

#include <vector>

#include "deviceArrays/headers/Support/Streamable.h"
#include "math/Real3dDevice.hpp"

#include <memory>


template<typename T>
std::unique_ptr<BandedMat<T>> & Laplacian1d<T>::operator[](size_t dim) {
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
    adjacencys(dim) {
}

template <typename T>
T invSq(T x) {
    return 1/(x*x);
}

void AdjacencyPatern::loadMapRowToDiag(Vec<int32_t>& diags, const cudaStream_t stream) const{
    loadMapRowToDiag(diags, {here, upDown.getLeft(), upDown.getRight(), leftRight.getLeft(), leftRight.getRight(), frontBack.getLeft(), frontBack.getRight()}, stream);
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
    Mat<T> preAllocatedForL = rhsAndL.subMat(0,0,dim.size(), numDiags);
    SimpleArray<T> rhsModifier = rhsAndL.col(numDiags);

    setOperation(stream, preAllocatedForL, preAllocatedForIndices, rhsModifier);
}

template<typename T>
void Laplacian<T>::setOperation(cudaStream_t stream, Mat<T> &preAllocatedForL, Vec<int32_t> &preAllocatedForIndices, Vec<T>& rhsModifier) {

    KernelPrep kp = this->dim.kernelPrep();
    buildLaplacianKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAllocatedForL.toKernel2d(),
        this->dim, this->boundary,
        this->adjacencys,
        rhsModifier.toKernel1d()
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    this->adjacencys.loadMapRowToDiag(preAllocatedForIndices, stream);

    bandedL = std::make_unique<BandedMat<T>>(preAllocatedForL, preAllocatedForIndices);
    rhsBC = std::make_unique<Vec<T>>(rhsModifier);
}

template<typename T>
void Laplacian1d<T>::set(cudaStream_t stream, Mat<T> &preAllocatedForL_i, Vec<int32_t> &preAllocatedForIndices, size_t dim) {

    AdjacencyInd prev(1, -1), next(2, 1), primary(0, 0);

    KernelPrep kp(dim);

    buildL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAllocatedForL_i.toKernel2d(),
        this->boundary(dim, true), this->boundary(dim, false),
        primary, prev, next
    );

    CHECK_CUDA_ERROR(cudaGetLastError());

    AdjacencyPatern::loadMapRowToDiag(preAllocatedForIndices, {primary, prev, next}, stream);

    this->operator[](dim) = std::make_unique<BandedMat<T>>(preAllocatedForL_i, preAllocatedForIndices);
}

template<typename T>
void Laplacian1d<T>::set(cudaStream_t stream, Mat<T>* preAllocatedForL_iX3, Vec<int32_t> &preAllocatedForIndices) {

    AdjacencyInd prev(1, -1), next(2, 1), primary(0, 0);

    KernelPrep kp(std::max(preAllocatedForL_iX3[0].rows, std::max(preAllocatedForL_iX3[1].rows, preAllocatedForL_iX3[2].rows)));

    buildAllL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAllocatedForL_iX3[0].toKernel2d(), preAllocatedForL_iX3[1].toKernel2d(), preAllocatedForL_iX3[2].toKernel2d(),
        this->boundary,
        primary, prev, next
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    AdjacencyPatern::loadMapRowToDiag(preAllocatedForIndices, {primary, prev, next}, stream);

    for (size_t i = 0; i < 3; i++)
        this->operator[](i) = std::make_unique<BandedMat<T>>(preAllocatedForL_iX3[i], preAllocatedForIndices);
}

template<typename T>
void Laplacian1d<T>::set(cudaStream_t stream, size_t n, size_t dim) {
    Vec<int32_t> preAllocatedForIndices = Vec<int32_t>::create(3, stream);
    Mat<T> preAllocatedForL_i = Mat<T>::create(n, 3);
    set(stream, preAllocatedForL_i, preAllocatedForIndices, dim);
}

template<typename T>
void Laplacian1d<T>::set(cudaStream_t stream, const GridDim& dim) {
    Vec<int32_t> preAllocatedForIndices = Vec<int32_t>::create(3, stream);

    std::shared_ptr<Mat<T>> preAllocatedForL_iX3[3];

    createUnique(boundary, preAllocatedForL_iX3, [](const auto& c) {
        return Mat<T>::create(c.dim, 3);
    });

    Mat<T> mats[3] = {*preAllocatedForL_iX3[0], *preAllocatedForL_iX3[1], *preAllocatedForL_iX3[2]};
    set(stream, mats, preAllocatedForIndices);
}

template<typename T>
LaplacianEigenVec<T>::LaplacianEigenVec(
    const SquareMat<T>& eVecX,
    const SquareMat<T>& eVecY,
    const SquareMat<T>& eVecZ
) : eVecX(eVecX), eVecY(eVecY), eVecZ(eVecZ) {}

template<typename T>
LaplacianEigenVal<T>::LaplacianEigenVal(const Vec<T> &eVecX, const Vec<T> &eVecY, const Vec<T> &eVecZ) :
    eVecX(eVecX), eVecY(eVecY), eVecZ(eVecZ) {
}

template<typename T>
LaplacianEigen<T>::LaplacianEigen(const LaplacianEigenVal<T> &vals, const LaplacianEigenVec<T> &vecs) :
    vals(vals), vecs(vecs) {}


template<typename Real>
void BoundaryConfig<Real>::generateEigen(Handle *hands, Event *events, std::shared_ptr<Mat<Real>> (&preAllocatedForL_iX3)[3]) {


    createUnique(preAllocatedForL_iX3, [](BoundaryConditionPair<Real> c) {
        return Mat<Real>::create(c.dimLength, c.dimLength + 1);
    });

    for (size_t i = 0; i < 3; ++i)
        if (repeat(i) < 0) (*this)[i].generateEigen(hands[i], *(preAllocatedForL_iX3[i]));

    events[0].record(hands[1]);
    events[1].record(hands[2]);

}

template<typename T>
LaplacianEigen<T> LaplacianEigen<T>::make(const BoundaryConfig<T>& boundary, Handle* hands, Event* events) {

    std::shared_ptr<Mat<T>> eigen[3];
    boundary.generateEigen(hands, events, eigen);
    LaplacianEigenVal<T> vals(eigen[0]->lastCol(), eigen[1]->lastCol(), eigen[2]->lastCol());
    LaplacianEigenVec<T> vecs(eigen[0]->sqSubMatFirstBiggest(), eigen[1]->sqSubMatFirstBiggest(), eigen[2]->sqSubMatFirstBiggest());
    events[0].hold(hands[0]);
    events[1].hold(hands[0]);
    return LaplacianEigen<T>(vals, vecs);
}

template<typename T>
BandedMat<T> & Laplacian<T>::banded(cudaStream_t stream) {
    if (bandedL == nullptr) setOperation(stream);
    return *bandedL;
}

template<typename T>
void Laplacian<T>::setRhsBC(cudaStream_t stream) {
    Vec<T> rhsModifier = SimpleArray<T>::create(dim.size(), stream);
    setRhsBC(stream, rhsModifier);
}

template<typename T>
void Laplacian<T>::setRhsBC(cudaStream_t stream, Vec<T>& rhsModifier) {
    KernelPrep kp(std::max(dim.rows, dim.layers), std::max(dim.layers, dim.cols));
    buildRhsBCKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(dim, boundary, rhsModifier.toKernel1d());
    CHECK_CUDA_ERROR(cudaGetLastError());
    rhsBC = std::make_unique<Vec<T>>(rhsModifier);
}


#define INSTANTIATE_LAPLACIAN(T) \
template class Laplacian<T>; \
template class Laplacian1d<T>; \
template class LaplacianEigenVal<T>; \
template class LaplacianEigenVec<T>; \
template class LaplacianEigen<T>; \
template class BoundaryConfig<T>;

INSTANTIATE_LAPLACIAN(float)
INSTANTIATE_LAPLACIAN(double)