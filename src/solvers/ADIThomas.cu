//
// Created by usr on 2/9/26.
//

#include "ADIThomas.cuh"

//TODO: add in delta x, y, and z
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

template<typename Real>
__global__ void residual3dKernel(
    DeviceData3d<Real> x,
    DeviceData3d<Real> b,
    DeviceData3d<Real> residual

) {
    if (GridInd3d ind; ind < x) {

        Real Lx =
              6 * x[ind]
            - (ind.col   < x.cols   - 1) * x(ind, 0,  1,  0)
            - (ind.col   > 0)            * x(ind, 0, -1,  0)
            - (ind.row   < x.rows   - 1) * x(ind, 1,  0,  0)
            - (ind.row   > 0)            * x(ind,-1,  0,  0)
            - (ind.layer < x.layers - 1) * x(ind, 0,  0,  1)
            - (ind.layer > 0)            * x(ind, 0,  0, -1);

        residual[ind] = b[ind] - Lx;
    }
}

template<typename Real>
ADIThomas<Real>::ADIThomas(const GridDim &dim, size_t max_iterations, const Real &tolerance) :
    dim(dim),
    maxIterations(max_iterations),
    tolerance(tolerance),
    bNorm(Singleton<Real>::create()),
    rNorm(Singleton<Real>::create()),
    thomasSratch(Mat<Real>::create(dim.maxDim(), 2 * std::max(std::max(dim.rows * dim.cols, dim.rows * dim.layers),dim.layers*dim.cols))),
    thomasCols(thomasSratch.subMat(0,0,dim.rows, 2 * dim.cols * dim.layers)),
    thomasRows(thomasSratch.subMat(0,0,dim.cols, 2 * dim.rows * dim.layers)),
    thomasDepths(thomasSratch.subMat(0,0,dim.layers, 2 * dim.rows * dim.cols))
{
}

template<typename Real>
void ADIThomas<Real>::solve(SimpleArray<Real>& x, const SimpleArray<Real>& b, Handle &hand) {

    auto xTensor = x.tensor(dim.rows, dim.cols);
    auto bTensor = b.tensor(dim.rows, dim.cols);
    auto rTensor = r.tensor(dim.rows, dim.cols);


    KernelPrep kp = rTensor.kernelPrep();

    b.norm(bNorm, hand);

    double bNormHost = bNorm.get(hand);

    if (bNormHost < tolerance) {
        x.fill(0, hand);
        return;
    }

    for (size_t i = 0; i < maxIterations; ++i) {
        setColRKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(rTensor, xTensor, bTensor);
        thomasCols.solveLaplacian(x.matrix(dim.rows * dim.layers), r.matrix(dim.rows * dim.layers), dim.numDims() == 3, hand);

        setRowRKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(rTensor, xTensor, bTensor);
        thomasRows.solveLaplacianTranspose(x.matrix(dim.rows), r.matrix(dim.rows), dim.numDims() == 3, hand);

        if (dim.numDims() == 3) {
            setDepthRKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(rTensor, xTensor, bTensor);
            thomasDepths.solveLaplacianDepths(xTensor, rTensor, hand);
        }

        residual3dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
            xTensor.toKernel3d(),
            bTensor.toKernel3d(),
            rTensor.toKernle3d()
        );

        r.norm(rNorm, hand);


        if (rNorm.get(hand)/bNormHost < tolerance) break;
    }
}
