//
// Created by usr on 12/24/25.
//

#include "poisson/Poisson.cuh"

#include <vector>

#include "deviceArrays/headers/Support/Streamable.h"


#include <memory>

#include "LaplacianKernels.cuh"


namespace poisson {

    template<typename T, typename BoundaryConfigT>
    BandedMat<T> laplacian(const BoundaryConfigT& boundary, Mat<T>& gridSizeXnumDiags, Vec<int32_t>& numDiags, cudaStream_t stream) {

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

    template<typename T, typename BoundaryConfigT>
    BandedMat<T> laplacian(const BoundaryConfigT& boundary, cudaStream_t stream) {
        GridDim dim = boundary.dim();
        size_t numDiags = dim.numDims() == 3 ? numDiagonals3d : numDiagonals2d;
        auto mat = Mat<T>::create(dim.size(), numDiags);
        Vec<int32_t> indices = Vec<int32_t>::create(numDiags, stream);
        laplacian(boundary, mat, indices, stream);
        return BandedMat<T>(mat, indices);
    }

    template<typename T, typename BoundaryConfigT>
    void boundaryCorrection(const BoundaryConfigT& boundary, SimpleArray<T> rhsCorrectionGoesHere, cudaStream_t stream) {
        GridDim dimension = boundary.dim();

        rhsCorrectionGoesHere.fill(0, stream);


        KernelPrep kp(
            std::max(dimension.rows, dimension.layers),
            std::max(dimension.layers, dimension.cols)
        );

        buildRhsBoundaryCorrectionKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
            dimension,
            boundary,
            rhsCorrectionGoesHere.toKernel1d()
        );

        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    template<typename T, typename BoundaryConfigT>
    SimpleArray<T> boundaryCorrection(const BoundaryConfigT& boundary, cudaStream_t stream) {
        SimpleArray<T> rhs = SimpleArray<T>::create(boundary.dim().size(), stream);

        boundaryCorrection(boundary, rhs, stream);

        return rhs;
    }


}

#define INSTANTIATE_POISSON_BOUNDARY(Real, SegX, SegY, SegZ) \
template BandedMat<Real> poisson::laplacian<Real, BoundaryConfig<Real, SegX, SegY, SegZ>>( \
const BoundaryConfig<Real, SegX, SegY, SegZ>&, cudaStream_t); \
template SimpleArray<Real> poisson::boundaryCorrection<Real, BoundaryConfig<Real, SegX, SegY, SegZ>>( \
const BoundaryConfig<Real, SegX, SegY, SegZ>&, cudaStream_t); \
template void poisson::boundaryCorrection<Real, BoundaryConfig<Real, SegX, SegY, SegZ>>( \
const BoundaryConfig<Real, SegX, SegY, SegZ>&, SimpleArray<Real>, cudaStream_t);

#define INSTANTIATE_POISSON_ALL(Real) \
INSTANTIATE_POISSON_BOUNDARY(Real, UniformSegment<Real>,  UniformSegment<Real>,  UniformSegment<Real>)  \
INSTANTIATE_POISSON_BOUNDARY(Real, UniformSegment<Real>,  UniformSegment<Real>,  VariableSegment<Real>) \
INSTANTIATE_POISSON_BOUNDARY(Real, UniformSegment<Real>,  VariableSegment<Real>, UniformSegment<Real>)  \
INSTANTIATE_POISSON_BOUNDARY(Real, UniformSegment<Real>,  VariableSegment<Real>, VariableSegment<Real>) \
INSTANTIATE_POISSON_BOUNDARY(Real, VariableSegment<Real>, UniformSegment<Real>,  UniformSegment<Real>)  \
INSTANTIATE_POISSON_BOUNDARY(Real, VariableSegment<Real>, UniformSegment<Real>,  VariableSegment<Real>) \
INSTANTIATE_POISSON_BOUNDARY(Real, VariableSegment<Real>, VariableSegment<Real>, UniformSegment<Real>)  \
INSTANTIATE_POISSON_BOUNDARY(Real, VariableSegment<Real>, VariableSegment<Real>, VariableSegment<Real>)

INSTANTIATE_POISSON_ALL(float)
INSTANTIATE_POISSON_ALL(double)

#define INSTANTIATE_LAPLACIAN(T)                                                                       \
template SimpleArray<T> poisson::boundaryCorrection<T, BoundaryConfigT>(const BoundaryConfigT&, cudaStream_t);        \
template void poisson::boundaryCorrection<T, BoundaryConfigT>(const BoundaryConfigT&, SimpleArray<T>, cudaStream_t);  \
template BandedMat<T> poisson::laplacian<T, BoundaryConfigT>(const BoundaryConfigT&, cudaStream_t);                   \
template BandedMat<T> poisson::laplacian<T, BoundaryConfigT>(const BoundaryConfigT&, Mat<T>&, Vec<int32_t>&, cudaStream_t);
