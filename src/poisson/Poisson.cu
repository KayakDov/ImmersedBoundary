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

// 1. Define the macro to instantiate the Poisson functions for the Device Config
#define INSTANTIATE_POISSON_FUNCTIONS(Real, SegX, SegY, SegZ) \
/* laplacian returning BandedMat */ \
template BandedMat<Real> poisson::laplacian<Real, BoundaryConfig<Real, SegX, SegY, SegZ>>( \
const BoundaryConfig<Real, SegX, SegY, SegZ>&, cudaStream_t); \
/* boundaryCorrection modifying array */ \
template void poisson::boundaryCorrection<Real, BoundaryConfig<Real, SegX, SegY, SegZ>>( \
const BoundaryConfig<Real, SegX, SegY, SegZ>&, SimpleArray<Real>, cudaStream_t); \
/* boundaryCorrection returning array */ \
template SimpleArray<Real> poisson::boundaryCorrection<Real, BoundaryConfig<Real, SegX, SegY, SegZ>>( \
const BoundaryConfig<Real, SegX, SegY, SegZ>&, cudaStream_t);

// 2. Trigger the macro
APPLY_TO_ALL_SEGMENT_COMBOS(double, INSTANTIATE_POISSON_FUNCTIONS)
APPLY_TO_ALL_SEGMENT_COMBOS(float,  INSTANTIATE_POISSON_FUNCTIONS)