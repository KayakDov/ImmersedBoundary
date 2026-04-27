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
    rawBanded(Mat<T>::create(boundary.leftRight.dimLength, 3), Mat<T>::create(boundary.topBottom.dimLength, 3), boundary.dim().numDims() == 3 ? Mat<T>::create(boundary.frontBack.dimLength, 3) : Mat<T>::empty()),
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

void AdjacencyPatern::loadMapRowToDiag(Vec<int32_t>& diags, cudaStream_t stream) const{
    std::vector<AdjacencyInd> adjacencies = {here, upDown.getLeft(), upDown.getRight(), leftRight.getLeft(), leftRight.getRight(), frontBack.getLeft(), frontBack.getRight()};
    loadMapRowToDiag(diags, adjacencies, stream);
}

void AdjacencyPatern::loadMapRowToDiag(Vec<int32_t> &diags, std::vector<AdjacencyInd>& indices, cudaStream_t stream) {
    std::vector<int32_t> diagsCpu(diags.size(), 0);
    for (AdjacencyInd ind : indices) diagsCpu[ind.col] = ind.diag;
    diags.set(diagsCpu.data(), stream);
    cudaStreamSynchronize(stream);//Don't want diagsCpu to be destroyed before the memory is passed.
}

template<typename T>
LaplacianEigen<T>::LaplacianEigen(const XYZ<Vec<T>> &vals, const XYZ<SquareMat<T>> &vecs) :
    vals(vals), vecs(vecs) {}


template<typename Real>
void BoundaryConfig<Real>::generateEigen(Handle *hands, Event *events, std::shared_ptr<Mat<Real>> (&preAllocatedForL_iX3)[3]) const{

    createUnique<Mat<Real>>(preAllocatedForL_iX3, [](const BoundaryPair<Real>& c) {
        return Mat<Real>::create(c.dimLength, c.dimLength + 1);
    });

    bool is3d = dim().numDims() == 3;

    for (size_t i = 0; i < 2 + is3d; ++i)
        if (repeat(i) < 0) (*this)[i].generateEigen(hands[i], *(preAllocatedForL_iX3[i]));

    events[0].record(hands[1]);
    if (is3d) events[1].record(hands[2]);
}

template<typename T>
LaplacianEigen<T> LaplacianEigen<T>::make(const BoundaryConfig<T>& boundary, Handle* hands, Event* events) {

    bool is3d = boundary.dim().numDims() == 3;
    std::shared_ptr<Mat<T>> eigen[3];
    boundary.generateEigen(hands, events, eigen);
    XYZ<Vec<T>> vals(eigen[0]->lastCol(), eigen[1]->lastCol(), is3d ? eigen[2]->lastCol() : SimpleArray<T>::empty());
    XYZ<SquareMat<T>> vecs(eigen[0]->sqSubMatFirstBiggest(), eigen[1]->sqSubMatFirstBiggest(), is3d ? eigen[2]->sqSubMatFirstBiggest() : SquareMat<T>::empty());
    events[0].hold(hands[0]);
    if (is3d) events[1].hold(hands[0]);
    return LaplacianEigen<T>(vals, vecs);
}

template<typename T>
BandedMat<T> laplacianLinear(const BoundaryConfig<T>& boundary, cudaStream_t stream){
    Vec<int32_t> indices =
        Vec<int32_t>::create(boundary.dim().numDims() == 3
                             ? numDiagonals3d
                             : numDiagonals2d,
                             stream);

    AdjacencyPatern ap(boundary.dim());
    ap.loadMapRowToDiag(indices, stream);

    Mat<T> data = Mat<T>::create(boundary.dim().size(), indices.size());

    KernelPrep kp = boundary.dim().kernelPrep();

    buildLaplacianKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        data.toKernel2d(),
        boundary.dim(),
        boundary,
        ap
    );

    CHECK_CUDA_ERROR(cudaGetLastError());

    return BandedMat<T>(data, indices);
}


template<typename T>
void laplacianRHS(const BoundaryConfig<T>& boundary, SimpleArray<T>& rhs, cudaStream_t stream)
{
    GridDim dim = boundary.dim();

    KernelPrep kp(
        std::max(dim.rows, dim.layers),
        std::max(dim.layers, dim.cols)
    );

    buildRhsBCKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        dim,
        boundary,
        rhs.toKernel1d()
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
}

template<typename T>
SimpleArray<T> laplacianRHS(const BoundaryConfig<T>& boundary, cudaStream_t stream){
    SimpleArray<T> rhs = SimpleArray<T>::create(boundary.dim().size(), stream);

    laplacianRHS(boundary, rhs, stream);

    return rhs;
}

#define INSTANTIATE_LAPLACIAN(T)                          \
template SimpleArray<T> laplacianRHS<T>(const BoundaryConfig<T>&, cudaStream_t); \
template BandedMat<T> laplacianLinear<T>(const BoundaryConfig<T>&, cudaStream_t); \
template class Laplacian1d<T>;                            \
template class LaplacianEigen<T>;

INSTANTIATE_LAPLACIAN(float)
INSTANTIATE_LAPLACIAN(double)