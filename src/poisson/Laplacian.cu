//
// Created by usr on 12/24/25.
//

#include "poisson/Laplacian.cuh"

#include <vector>

#include "deviceArrays/headers/Support/Streamable.h"
#include "math/Real3dDevice.hpp"

#include <memory>


template<typename T>
Laplacian1d<T>::Laplacian1d(const BoundaryConfig<T> &boundary, Handle& hand) :
    boundary(boundary),
    rawBanded(Mat<T>::create(boundary.leftRight.dimLength, 3), Mat<T>::create(boundary.topBottom.dimLength, 3), Mat<T>::create(boundary.frontBack.dimLength, 3)),
    inds(SimpleArray<int32_t>::create(3, hand)){

    AdjacencyIndPair prevNext(1, 1);
    AdjacencyInd primary(0, 0);

    KernelPrep kp(std::max(std::max(rawBanded.x._rows, rawBanded.y._rows), rawBanded.z._rows));

    buildAllL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        rawBanded.x.toKernel2d(), rawBanded.y.toKernel2d(), rawBanded.z.toKernel2d(),
        this->boundary,
        primary, prevNext
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    std::vector<AdjacencyInd> adjacencys = {primary, prevNext.getLeft(), prevNext.getRight()};
    AdjacencyPatern::loadMapRowToDiag(inds, adjacencys, hand);
}

template<typename T>
BandedMat<T> Laplacian1d<T>::banded(size_t dim) {
    return BandedMat<T>(rawBanded[dim], inds);
}

template<typename T>
SquareMat<T> Laplacian1d<T>::dense(size_t dim, Handle& hand) {
    auto banded = this->banded(dim);
    auto square = SquareMat<T>::create(banded._rows);
    banded.getDense(square, &hand);
    return square;
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
    std::vector<AdjacencyInd> adjacencies = {here, upDown.getLeft(), upDown.getRight(), leftRight.getLeft(), leftRight.getRight(), frontBack.getLeft(), frontBack.getRight()};
    loadMapRowToDiag(diags, adjacencies, stream);
}

void AdjacencyPatern::loadMapRowToDiag(Vec<int32_t> &diags, std::vector<AdjacencyInd>& indices, cudaStream_t stream) {
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
LaplacianEigen<T>::LaplacianEigen(const XYZ<Vec<T>> &vals, const XYZ<SquareMat<T>> &vecs) :
    vals(vals), vecs(vecs) {}


template<typename Real>
void BoundaryConfig<Real>::generateEigen(Handle *hands, Event *events, std::shared_ptr<Mat<Real>> (&preAllocatedForL_iX3)[3]) const{


    createUnique<Mat<Real>>(preAllocatedForL_iX3, [](const BoundaryConditionPair<Real>& c) {
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
    XYZ<Vec<T>> vals(eigen[0]->lastCol(), eigen[1]->lastCol(), eigen[2]->lastCol());
    XYZ<SquareMat<T>> vecs(eigen[0]->sqSubMatFirstBiggest(), eigen[1]->sqSubMatFirstBiggest(), eigen[2]->sqSubMatFirstBiggest());
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
SquareMat<T> Laplacian<T>::dense(Handle& handle) {
    BandedMat<T> banded = this->banded(handle);
    auto result = SquareMat<T>::create(banded._rows);
    banded.getDense(result, &handle);
    return result;
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
template class LaplacianEigen<T>; \
template class BoundaryConfig<T>;

INSTANTIATE_LAPLACIAN(float)
INSTANTIATE_LAPLACIAN(double)