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
    ){

    KernelPrep kp(std::max(std::max(this->x._rows, this->y._rows), this->z._rows));

    buildAllL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        XYZ<DeviceData2d<T>>(this->x.toKernel2d(), this->y.toKernel2d(), this->z.toKernel2d()),
        boundary,
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
    mat.col(mat.primary.colInBanded).fill(0, hand);
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

// 1. Explicitly instantiate the specific class methods
template void Laplacian1d<float>::create<UniformSegment<float>>(const UniformSegment<float>&, TriDiagonal<float>, Handle&);
template void Laplacian1d<double>::create<UniformSegment<double>>(const UniformSegment<double>&, TriDiagonal<double>, Handle&);
template void Laplacian1d<float>::create<VariableSegment<float>>(const VariableSegment<float>&, TriDiagonal<float>, Handle&);
template void Laplacian1d<double>::create<VariableSegment<double>>(const VariableSegment<double>&, TriDiagonal<double>, Handle&);
template void Laplacian1d<float>::create<FluxLaplacian<float>>(const FluxLaplacian<float>&, TriDiagonal<float>, Handle&);
template void Laplacian1d<double>::create<FluxLaplacian<double>>(const FluxLaplacian<double>&, TriDiagonal<double>, Handle&);

// 2. Explicitly instantiate the base class itself
template class Laplacian1d<float>;
template class Laplacian1d<double>;

// 3. Explicitly instantiate the templated constructor for all 8 config combinations
#define INSTANTIATE_LAPLACIAN1D_CTOR(Real, SegX, SegY, SegZ) \
template Laplacian1d<Real>::Laplacian1d(const BoundaryConfig<Real, SegX, SegY, SegZ>&, Handle&);

APPLY_TO_ALL_SEGMENT_COMBOS(double, INSTANTIATE_LAPLACIAN1D_CTOR)
APPLY_TO_ALL_SEGMENT_COMBOS(float,  INSTANTIATE_LAPLACIAN1D_CTOR)