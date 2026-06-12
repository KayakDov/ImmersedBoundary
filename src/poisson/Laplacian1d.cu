//
// Created by usr on 6/11/26.
//

#include "Laplacian1d.cuh"
#include "BoundaryConfig.cuh"
#include "deviceArrays/headers/SquareMat.h"
#include "poisson/LaplacianKernels.cuh"


template<typename T>
template<typename BoundaryConfigT>
Laplacian1d<T>::Laplacian1d(const BoundaryConfigT &boundary, Handle& hand) :
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
    (*this)[dim].getDense(denseGoesHere, hand);
}

template<typename T>
template<typename AxisSegmentT>
void Laplacian1d<T>::create(const AxisSegmentT &segment, TriDiagonal<T> mat, Handle &hand) {
    KernelPrep kp(3, mat._rows);
    buildL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(mat.toKernel2d(), segment, mat.primary, mat.prevNext);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template<typename T>
template<typename AxisSegmentT>
TriDiagonal<T> Laplacian1d<T>::create(const AxisSegmentT &segment, Handle &hand) {
    TriDiagonal<T> mat(segment.numNodes(), hand);
    create(segment, mat, hand);
    return mat;
}

template void Laplacian1d<float>::create<UniformSegment<float>>(const UniformSegment<float>&, TriDiagonal<float>, Handle&);
template void Laplacian1d<double>::create<UniformSegment<double>>(const UniformSegment<double>&, TriDiagonal<double>, Handle&);
template void Laplacian1d<float>::create<VariableSegment<float>>(const VariableSegment<float>&, TriDiagonal<float>, Handle&);
template void Laplacian1d<double>::create<VariableSegment<double>>(const VariableSegment<double>&, TriDiagonal<double>, Handle&);

template class Laplacian1d<float>;
template class Laplacian1d<double>;