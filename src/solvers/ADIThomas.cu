//
// Created by usr on 2/9/26.
//

#include "ADIThomas.cuh"

#include <vector>

#include "ToeplitzLaplacian.cuh"
#include "Support/Streamable.h"

//TODO: add in delta x, y, and z
//TODO:implement these methods for 2d.
//TODO: combine this into single kernel with Thomas solver
template<typename Real>
__global__ void setRowRKernel(DeviceData3d<Real> r, DeviceData3d<Real> x, DeviceData3d<Real> b) {
    if (GridInd3d ind; ind < r) {
        r[ind] = b[ind] //+ 4 * x[ind]
            - (ind.col < x.cols - 1) * x(ind, 0, 1, 0)
            - (ind.col > 0) * x(ind, 0, -1, 0)
            - (ind.layer < x.layers - 1) * x(ind, 0, 0, 1)
            - (ind.layer > 0) * x(ind, 0, 0, -1);
    }
}
template<typename Real>
__global__ void setColRKernel(DeviceData3d<Real> r, DeviceData3d<Real> x, DeviceData3d<Real> b) {
    if (GridInd3d ind; ind < r) {
        r[ind] = b[ind] //+ 4 * x[ind]
            - (ind.row < x.rows - 1) * x(ind, 1, 0, 0)
            - (ind.row > 0) * x(ind, -1, 0, 0)
            - (ind.layer < x.layers - 1) * x(ind, 0, 0, 1)
            - (ind.layer > 0) * x(ind, 0, 0, -1);
    }
}
template<typename Real>
__global__ void setDepthRKernel(DeviceData3d<Real> r, DeviceData3d<Real> x, DeviceData3d<Real> b) {
    if (GridInd3d ind; ind < r) {
        r[ind] = b[ind] //+ 4 * x[ind]
            - (ind.col < x.cols - 1) * x(ind, 0, 1, 0)
            - (ind.col > 0) * x(ind, 0, -1, 0)
            - (ind.row < x.rows - 1) * x(ind, 1, 0, 0)
            - (ind.row > 0) * x(ind, -1, 0, 0);
    }
}

template<typename Real>
__global__ void residual3dKernel(
    DeviceData3d<Real> x,
    DeviceData3d<Real> b,
    DeviceData3d<Real> residual

) {
    if (GridInd3d ind; ind < x)
        residual[ind] = b[ind] +  6 * x[ind]
            - (ind.col   < x.cols   - 1) * x(ind, 0,  1,  0)
            - (ind.col   > 0)            * x(ind, 0, -1,  0)
            - (ind.row   < x.rows   - 1) * x(ind, 1,  0,  0)
            - (ind.row   > 0)            * x(ind,-1,  0,  0)
            - (ind.layer < x.layers - 1) * x(ind, 0,  0,  1)
            - (ind.layer > 0)            * x(ind, 0,  0, -1);
}

template<typename Real>
ADIThomas<Real>::ADIThomas(const GridDim &dim, size_t max_iterations, const Real &tolerance, SimpleArray<Real> dimSize, Handle& hand) :
    dim(dim),
    maxIterations(max_iterations),
    tolerance(tolerance),
    bNorm(Singleton<Real>::create(hand)),
    rNorm(Singleton<Real>::create(hand)),
    thomasSratch(Mat<Real>::create(dim.maxDim(), 2 * std::max(std::max(dim.rows * dim.cols, dim.rows * dim.layers),dim.layers*dim.cols))),
    thomasCols(thomasSratch.subMat(0,0,dim.rows, 2 * dim.cols * dim.layers)),
    thomasRows(thomasSratch.subMat(0,0,dim.cols, 2 * dim.rows * dim.layers)),
    thomasDepths(thomasSratch.subMat(0,0,dim.layers, 2 * dim.rows * dim.cols)),
    r(dimSize)
{
}

template<typename Real>
void ADIThomas<Real>::solve(SimpleArray<Real>& x, const SimpleArray<Real>& b, Handle &hand) {

    auto xTensor = x.tensor(dim.rows, dim.cols);
    auto bTensor = b.tensor(dim.rows, dim.cols);
    auto rTensor = r.tensor(dim.rows, dim.cols);


    const KernelPrep kp = rTensor.kernelPrep();

    b.norm(bNorm, hand);

    double bNormHost = bNorm.get(hand);

    if (bNormHost < tolerance) {
        x.fill(0, hand);
        return;
    }
    size_t i = 0;
    for (; i < maxIterations; ++i) {

        // std::cout << "x at start = \n" << GpuOut<Real>(xTensor, hand) << std::endl;

        setColRKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(rTensor.toKernel3d(), xTensor.toKernel3d(), bTensor.toKernel3d());
        // std::cout << "r = \n" << GpuOut<Real>(rTensor, hand) << std::endl;

        auto rMat = r.matrix(dim.rows);
        auto xMat = x.matrix(dim.rows);
        thomasCols.solveLaplacian(xMat, rMat, dim.numDims() == 3, hand);

        // std::cout << "x1 = \n" << GpuOut<Real>(xTensor, hand) << std::endl;

        setRowRKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(rTensor.toKernel3d(), xTensor.toKernel3d(), bTensor.toKernel3d());
        // std::cout << "r1 = \n" << GpuOut<Real>(rTensor, hand) << std::endl;

        auto rMatT = r.matrix(dim.rows * dim.layers);
        auto xMatT = x.matrix(dim.rows * dim.layers);
        thomasRows.solveLaplacianTranspose(xMatT, rMatT, dim.numDims() == 3, hand);
        // std::cout << "x2 = \n" << GpuOut<Real>(xTensor, hand) << std::endl;

        if (dim.numDims() == 3) {
            setDepthRKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(rTensor.toKernel3d(), xTensor.toKernel3d(), bTensor.toKernel3d());
            // std::cout << "r2 = \n" << GpuOut<Real>(rTensor, hand) << std::endl;
            // std::cout << "r tensor = \n" << GpuOut<Real>(rTensor, hand) << std::endl;
            // std::cout << "x tensor = \n" << GpuOut<Real>(xTensor, hand) << std::endl;
            thomasDepths.solveLaplacianDepths(xTensor, rTensor, hand);
            // std::cout << "x3 = \n" << GpuOut<Real>(xTensor, hand) << std::endl;
        }

        residual3dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
            xTensor.toKernel3d(),
            bTensor.toKernel3d(),
            rTensor.toKernel3d()
        );

        r.norm(rNorm, hand);

        // std::cout << GpuOut<Real>(r, hand) << std::endl;

        if (rNorm.get(hand)/*/bNormHost*/ < tolerance) break;
    }
    if (i == maxIterations) std::cout << "ADIThomas reached it's maximum number of iterations: " << maxIterations << std::endl;
}

template<typename Real>
void ADIThomas<Real>::test() {
    GridDim dim(3, 2, 2);

    Handle hand;
    auto x = SimpleArray<Real>::create(dim.size(), hand);
    auto b = SimpleArray<Real>::create(dim.size(), hand);
    auto scratch = SimpleArray<Real>::create(dim.size(), hand);
    std::vector<Real> bHost = {1,2,3,4,5,6,7,8,9,10,11,12};
    b.set(bHost.data(), hand);

    ADIThomas adiThomas(dim, 10000, 1e-5, scratch, hand);
    adiThomas.solve(x, b, hand);

    ToeplitzLaplacian<Real>::printL(dim, hand);

    std::cout << "b = " << GpuOut<Real>(b, hand) << std::endl;

    std::cout << "x = " << GpuOut<Real>(x, hand) << std::endl;
    std::cout << "expected: -1.14496, -1.72689, -1.64496, -1.76261, -2.43277, -2.26261, -2.38025, -3.13866, -2.88025, -2.99790, -3.84454, -3.49790" << std::endl;

}


template class ADIThomas<double>;
template class ADIThomas<float>;
