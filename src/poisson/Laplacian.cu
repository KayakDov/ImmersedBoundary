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
        rawBanded(
            Mat<T>::create(boundary.x.dimLength, 3),
            Mat<T>::create(boundary.y.dimLength, 3),
            boundary.dim().numDims() == 3 ? Mat<T>::create(boundary.z.dimLength, 3) : Mat<T>::empty()
        ),
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
        auto square = SquareMat<T>::create(banded._rows);
        dense(dim, square, hand);
        return square;
    }

    template<typename T>
    void Laplacian1d<T>::dense(size_t dim, SquareMat<T>& denseGoesHere, Handle& hand) {
        this->banded(dim).banded.getDense(denseGoesHere, hand);
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
    void boundaryCorrection(const BoundaryConfig<T>& boundary, SimpleArray<T> rhsCorrectionGoesHere, cudaStream_t stream) {
        GridDim dimension = boundary.dim();

        rhsCorrectionGoesHere.fill(0, stream);

        KernelPrep kp(
            std::max(dimension.rows, dimension.layers),
            std::max(dimension.layers, dimension.cols)
        );

        buildRhsBCKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
            dimension,
            boundary,
            rhsCorrectionGoesHere.toKernel1d()
        );

        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    template<typename T>
    SimpleArray<T> boundaryCorrection(const BoundaryConfig<T>& boundary, cudaStream_t stream) {
        SimpleArray<T> rhs = SimpleArray<T>::create(boundary.dim().size(), stream);

        boundaryCorrection(boundary, rhs, stream);

        return rhs;
    }

    template<typename T>
    void generateEigen(Handle& hand, SquareMat<T> eVecs, Vec<T> eVals, const VariableSegment<T> &axisSegment) {

        auto buffer = Mat<T>::create(eVecs._rows, 3 + eVals.size());
        auto rawBanded = buffer.subMat(0,0,eVals.size(), 3);
        AdjacencyInd primary(0, 0);
        AdjacencyIndPair superSub(1, 1);
        KernelPrep kp(3, eVecs._rows);
        buildL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(rawBanded, axisSegment, primary, superSub);
        CHECK_CUDA_ERROR(cudaGetLastError());
        auto indices = SimpleArray<int32_t>::create(3, hand);

        std::vector<int32_t> indicesHost(3, 0);
        indicesHost[primary.colInBanded] = primary.diag;
        indicesHost[superSub.left.colInBanded] = superSub.left.diag;
        indicesHost[superSub.right.colInBanded] = superSub.right.diag;
        indices.set(indicesHost.data(), hand);

        BandedMat<T> banded(rawBanded, indices);

        auto dense = buffer.sqSubMat(0, 3, eVals.size());
        banded.getDense(dense, hand);

        dense.eigen(eVals, &eVecs, hand);
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