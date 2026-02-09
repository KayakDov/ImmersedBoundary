//
// Created by usr on 2/9/26.
//

#include "ADIThomas.cuh"

//TODO:implement these methods for 2d.
template<typename Real>//TODO: combine this into single kernel with Thomas solver
__global__ void setRowRKernel(DeviceData3d<Real> r, DeviceData3d<Real> x, DeviceData3d<Real> b) {
    if (GridInd3d ind; ind < r) {
        r[ind] = b[ind]
            + (ind.col < x.cols - 1) * x(ind, 0, 1, 0)
            + (ind.col > 0) * x(ind, 0, -1, 0)
            + (ind.layer < x.layers - 1) * x(ind, 0, 0, 1)
            + (ind.layer > 0) * x(ind, 0, 0, -1);
    }
}
template<typename Real>
__global__ void setColRKernel(DeviceData3d<Real> r, DeviceData3d<Real> x, DeviceData3d<Real> b) {
    if (GridInd3d ind; ind < r) {
        r[ind] = b[ind]
            + (ind.row < x.rows - 1) * x(ind, 1, 0, 0)
            + (ind.row > 0) * x(ind, -1, 0, 0)
            + (ind.layer < x.layers - 1) * x(ind, 0, 0, 1)
            + (ind.layer > 0) * x(ind, 0, 0, -1);
    }

}
template<typename Real>
__global__ void setDepthRKernel(DeviceData3d<Real> r, DeviceData3d<Real> x, DeviceData3d<Real> b) {
    if (GridInd3d ind; ind < r) {
        r[ind] = b[ind]
            + (ind.col < x.cols - 1) * x(ind, 0, 1, 0)
            + (ind.col > 0) * x(ind, 0, -1, 0)
            + (ind.row < x.rows - 1) * x(ind, 1, 0, 0)
            + (ind.row > 0) * x(ind, -1, 0, 0);
    }

}

template<typename Real>//TODO: implement for 2d
__global__ void residualNorm3dBlockKernel(DeviceData3d<Real> x, DeviceData3d<Real> b, Real* blockSums) {
    extern __shared__ Real s[];
    Real& sum = s[threadIdx.x];
    sum = 0;

    if (GridInd3d ind; ind < x) {

        Real Lx =
              6 * x[ind]
            - (ind.col   < x.cols   - 1) * x(ind, 0,  1,  0)
            - (ind.col   > 0)            * x(ind, 0, -1,  0)
            - (ind.row   < x.rows   - 1) * x(ind, 1,  0,  0)
            - (ind.row   > 0)            * x(ind,-1,  0,  0)
            - (ind.layer < x.layers - 1) * x(ind, 0,  0,  1)
            - (ind.layer > 0)            * x(ind, 0,  0, -1);

        Real r = b[ind] - Lx;
        sum = r * r;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sum += s[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        blockSums[blockIdx.x] = sum;
}

template<typename Real>
void ADIThomas<Real>::solve(SimpleArray<Real>& x, const SimpleArray<Real>& b, Handle &hand) {
    auto xTensor = x.tensor(dim.rows, dim.cols);

    auto bTensor = b.tensor(dim.rows, dim.cols);
    auto bMat = b.matrix(dim.rows);
    auto rTensor = r.tensor(dim.rows, dim.cols);

    KernelPrep kp = rTensor.kernelPrep();


    for (size_t i = 0; i < maxIterations; ++i) {
        setRowRKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(r, xTensor, bTensor);
        thomas.solveLaplacian(x.matrix(dim.rows * dim.layers), b, dim.numDims() == 3, hand);
        setColRKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(r, xTensor, bTensor);
        thomas.solveLaplacianTranspose(x.matrix(dim.rows), b, dim.numDims() == 3, hand);
        if (dim.numDims() == 3) {
            setDepthRKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(r, xTensor, bTensor);
            thomas.solveLaplacianDepths(xTensor, bTensor, hand);
        }

        residualNorm3dBlockKernel<<<>>>(x.tokernel3d(), b.tokernel3d(), residual.data(), hand);
        if (r.get() < tolerance)
    }
}