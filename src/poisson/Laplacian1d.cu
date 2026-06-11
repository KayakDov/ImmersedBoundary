//
// Created by usr on 6/11/26.
//

#include "Laplacian1d.cuh"
#include "BoundaryConfig.cuh"
#include "deviceArrays/headers/sparse/BandedMat.h"
#include "deviceArrays/headers/SquareMat.h"
#include "kronecker/KroneckerTriplet.h"
#include "solvers/Event.h"

template<typename T>
Laplacian1d<T>::Laplacian1d(const BoundaryConfig<T> &boundary, Handle& hand) :
    XYZ<TriDiagonal<T>>(
        {boundary.x.numNodes, hand},
        {boundary.y.numNodes, hand},
        {boundary.dim().numDims() == 3 ? Mat<T>::create(boundary.z.numNodes, 3) : Mat<T>::empty(), hand}
    ),
    boundary(boundary){

    KernelPrep kp(std::max(std::max(this->x._rows, this->y._rows), this->z._rows));

    buildAllL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        {this->x.toKernel2d(), this->y.toKernel2d(), this->z.toKernel2d()},
        this->boundary,
        TriDiagonal<T>::primary, TriDiagonal<T>::prevNext
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template<typename T>
SquareMat<T> Laplacian1d<T>::dense(size_t dim, Handle& hand) {
    auto square = SquareMat<T>::create((*this)[dim]._rows);
    dense(dim, square, hand);
    return square;
}

template<typename T>
void Laplacian1d<T>::dense(size_t dim, SquareMat<T>& denseGoesHere, Handle& hand) {
    this->banded(dim).getDense(denseGoesHere, hand);
}

template class Laplacian1d<float>;
template class Laplacian1d<double>;