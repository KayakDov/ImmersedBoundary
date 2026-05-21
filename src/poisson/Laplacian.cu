//
// Created by usr on 12/24/25.
//

#include "poisson/Laplacian.cuh"

#include <vector>

#include "deviceArrays/headers/Support/Streamable.h"


#include <memory>

#include "LaplacianKernels.cuh"


namespace poisson {
    template<typename T>
    Laplacian1d<T>::Laplacian1d(const BoundaryConfig<T> &boundary, Handle& hand) :
        boundary(boundary),
        rawBanded(Mat<T>::create(boundary.x.dimLength, 3), Mat<T>::create(boundary.y.dimLength, 3), boundary.dim().numDims() == 3 ? Mat<T>::create(boundary.z.dimLength, 3) : Mat<T>::empty()),
        inds(SimpleArray<int32_t>::create(3, hand)){

        AdjacencyIndPair prevNext(1, 1);
        AdjacencyInd primary(0, 0);

        KernelPrep kp(std::max(std::max(rawBanded.x._rows, rawBanded.y._rows), rawBanded.z._rows));

        buildAllL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
            {rawBanded.x.toKernel2d(), rawBanded.y.toKernel2d(), rawBanded.z.toKernel2d()},
            this->boundary,
            primary, prevNext
        );
        CHECK_CUDA_ERROR(cudaGetLastError());

        std::vector<AdjacencyInd> adjacencys = {primary, prevNext.left, prevNext.right};
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
        banded.getDense(square, hand);
        return square;
    }

    template<typename T>
    Eigen<T>::Eigen(const XYZ<Vec<T>> &vals, const XYZ<SquareMat<T>> &vecs) :
        vals(vals), vecs(vecs) {}


    template<typename T>
    Eigen<T> Eigen<T>::make(const BoundaryConfig<T>& boundary, Handle* hands3, Event* events) {

        bool is3d = boundary.dim().numDims() == 3;
        std::shared_ptr<Mat<T>> eigen[3];
        boundary.generateEigen(hands3, events, eigen);
        XYZ<Vec<T>> vals(eigen[0]->lastCol(), eigen[1]->lastCol(), is3d ? eigen[2]->lastCol() : SimpleArray<T>::empty());
        XYZ<SquareMat<T>> vecs(
            eigen[0]->sqSubMatFirstBiggest(),
            eigen[1]->sqSubMatFirstBiggest(),
            is3d ? eigen[2]->sqSubMatFirstBiggest() : SquareMat<T>::empty()//GPUConst<T>::get(0).matrix(1).sqSubMat(0,0,1)
        );
        return Eigen<T>(vals, vecs);
    }

    template<typename T>
    GridDim Eigen<T>::dim() const {
        return GridDim(vals.y.size(), vals.x.size(), vals.z.size());
    }

    template<typename T>
    BandedMat<T> laplacian(const BoundaryConfig<T>& boundary, Mat<T>& gridSizeXnumDiags, Vec<int32_t>& numDiags, cudaStream_t stream) {

        GridDim dimension = boundary.dim();

        AdjacencyPatern adjPat(dimension);
        adjPat.loadMapRowToDiag(numDiags, stream);

        KernelPrep kp = dimension.kernelPrep();

        buildLaplacianKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
            gridSizeXnumDiags.toKernel2d(),
            dimension,
            boundary,
            adjPat
        );

        CHECK_CUDA_ERROR(cudaGetLastError());
        return BandedMat<T>(gridSizeXnumDiags, numDiags);
    }

    template<typename T>
    BandedMat<T> laplacian(const BoundaryConfig<T>& boundary, cudaStream_t stream) {
        GridDim dim = boundary.dim();
        size_t numDiags = dim.numDims() == 3 ? numDiagonals3d : numDiagonals2d;
        auto mat = Mat<T>::create(dim.size(), numDiags);
        Vec<int32_t> indices = Vec<int32_t>::create(numDiags, stream);
        laplacian(boundary, mat, indices, stream);
        return BandedMat<T>(mat, indices);
    }

    template<typename T>
    void boundaryCorrection(const BoundaryConfig<T>& boundary, SimpleArray<T> correctionGoesHere, cudaStream_t stream) {
        GridDim dimension = boundary.dim();

        correctionGoesHere.fill(0, stream);

        KernelPrep kp(
            std::max(dimension.rows, dimension.layers),
            std::max(dimension.layers, dimension.cols)
        );

        buildRhsBCKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
            dimension,
            boundary,
            correctionGoesHere.toKernel1d()
        );

        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    template<typename T>
    SimpleArray<T> boundaryCorrection(const BoundaryConfig<T>& boundary, cudaStream_t stream) {
        SimpleArray<T> rhs = SimpleArray<T>::create(boundary.dim().size(), stream);

        boundaryCorrection(boundary, rhs, stream);

        return rhs;
    }
}

#define INSTANTIATE_LAPLACIAN(T)                          \
template SimpleArray<T> poisson::boundaryCorrection<T>(const BoundaryConfig<T>&, cudaStream_t); \
template void poisson::boundaryCorrection<T>(const BoundaryConfig<T>&, SimpleArray<T>, cudaStream_t); \
template BandedMat<T> poisson::laplacian<T>(const BoundaryConfig<T>&, cudaStream_t); \
template class poisson::Laplacian1d<T>; \
template class poisson::Eigen<T>;

INSTANTIATE_LAPLACIAN(float)
INSTANTIATE_LAPLACIAN(double)