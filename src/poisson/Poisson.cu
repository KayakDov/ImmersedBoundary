//
// Created by usr on 12/24/25.
//

#include "poisson/Poisson.cuh"

#include <vector>

#include "deviceArrays/headers/Support/Streamable.h"


#include <memory>

#include "LaplacianKernels.cuh"


namespace poisson {

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

    template<typename T, typename BoundaryConfigT>
    BandedMat<T> laplacian(const BoundaryConfigT& boundary, cudaStream_t stream) {
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

        buildRhsBoundaryCorrectionKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
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


}

#define INSTANTIATE_LAPLACIAN(T)                                                                       \
template SimpleArray<T> poisson::boundaryCorrection<T>(const BoundaryConfig<T>&, cudaStream_t);        \
template void poisson::boundaryCorrection<T>(const BoundaryConfig<T>&, SimpleArray<T>, cudaStream_t);  \
template BandedMat<T> poisson::laplacian<T>(const BoundaryConfig<T>&, cudaStream_t);                   \
template BandedMat<T> poisson::laplacian<T>(const BoundaryConfig<T>&, Mat<T>&, Vec<int32_t>&, cudaStream_t);

INSTANTIATE_LAPLACIAN(float)
INSTANTIATE_LAPLACIAN(double)