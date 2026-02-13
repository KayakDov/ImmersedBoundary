//
// Created by usr on 2/9/26.
//

#include "ADIThomas.cuh"

#include <vector>

#include "ToeplitzLaplacian.cuh"
#include "Support/Streamable.h"

template<typename Real>
__device__ Real deltaXSide(const DeviceData3d<Real> grid, const GridInd3d& ind) {
    return (ind.col < grid.cols - 1)  * grid(ind, 0, 1, 0) +
        (ind.col > 0) * grid(ind, 0, -1, 0);
}
template<typename Real>
__device__ Real deltaYSide(const DeviceData3d<Real> grid, const GridInd3d& ind) {
    return (ind.row < grid.rows - 1)  * grid(ind, 1, 0, 0) +
        (ind.row > 0)  * grid(ind, -1, 0, 0);
}
template<typename Real>
__device__ Real deltaZSide(const DeviceData3d<Real> grid, const GridInd3d& ind) {
    return (ind.layer < grid.layers - 1)  * grid(ind, 0, 0, 1) +
        (ind.layer > 0) * grid(ind, 0, 0, -1);
}



template<typename Real>
__device__ Real deltaX(const DeviceData3d<Real> grid, const GridInd3d& ind) {
    return -2*grid[ind] + deltaXSide(grid, ind);
}
template<typename Real>
__device__ Real deltaY(const DeviceData3d<Real> grid, const GridInd3d& ind) {
    return -2*grid[ind] + deltaYSide(grid, ind);
}
template<typename Real>
__device__ Real deltaZ(const DeviceData3d<Real> grid, const GridInd3d& ind) {
    return -2*grid[ind] + deltaZSide(grid, ind);
}

//TODO: add in delta x, y, and z
//TODO:implement these methods for 2d.
//TODO: combine this into single kernel with Thomas solver
template<typename Real>
__global__ void setRHSA(DeviceData3d<Real> r, DeviceData3d<Real> x, DeviceData3d<Real> b) {
    if (GridInd3d ind; ind < r)
        r[ind] = 2*b[ind] - (-8 * x[ind] + deltaXSide(x, ind) + 2 * deltaYSide(x, ind) + 2 * deltaZSide(x, ind));
}
template<typename Real>
__global__ void setRHSB(DeviceData3d<Real> r, DeviceData3d<Real> x, DeviceData3d<Real> xStar) {
    if (GridInd3d ind; ind < r) r[ind] = deltaY(x, ind) - 2 * xStar[ind];
}
template<typename Real>
__global__ void setRHSC(DeviceData3d<Real> r, DeviceData3d<Real> x,DeviceData3d<Real> xStarStar) {
    if (GridInd3d ind; ind < r) r[ind] = deltaZ(x, ind) - 2 * xStarStar[ind];
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
ADIThomas<Real>::ADIThomas(const GridDim &dim, size_t max_iterations, const Real &tolerance, Mat<Real> dimSizeX2, Handle& hand) :
    dim(dim),
    maxIterations(max_iterations),
    tolerance(tolerance),
    bNorm(Singleton<Real>::create(hand)),
    rNorm(Singleton<Real>::create(hand)),
    thomasSratchMaxDim(Mat<Real>::create(dim.maxDim(), 2 * std::max(std::max(dim.rows * dim.cols, dim.rows * dim.layers),dim.layers*dim.cols))),
    thomasCols(thomasSratchMaxDim.subMat(0,0,dim.rows, 2 * dim.cols * dim.layers)),
    thomasRows(thomasSratchMaxDim.subMat(0,0,dim.cols, 2 * dim.rows * dim.layers)),
    thomasDepths(thomasSratchMaxDim.subMat(0,0,dim.layers, 2 * dim.rows * dim.cols)),
    rhs(dimSizeX2.col(0)),
    xThirdStep(dimSizeX2.col(1))
{
}

template<typename Real>
void ADIThomas<Real>::solve(SimpleArray<Real>& x, const SimpleArray<Real>& b, Handle &hand) {

    auto xTensor = x.tensor(dim.rows, dim.cols);
    auto bTensor = b.tensor(dim.rows, dim.cols);
    auto rTensor = rhs.tensor(dim.rows, dim.cols);
    auto xThirdStepTensor = xThirdStep.tensor(dim.rows, dim.cols);


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

        setRHSA<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(rTensor.toKernel3d(), xTensor.toKernel3d(), bTensor.toKernel3d());
        // std::cout << "After RHSA = \n" << GpuOut<Real>(rTensor, hand) << std::endl;

        auto rMat = rhs.matrix(dim.rows);
        auto xMat = xThirdStep.matrix(dim.rows);
        thomasCols.solveLaplacian(xMat, rMat, dim.numDims() == 3, hand);

        // std::cout << "x 1/3 step = \n" << GpuOut<Real>(xThirdStepTensor, hand) << std::endl;

        setRHSB<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(rTensor.toKernel3d(), xTensor.toKernel3d(), xThirdStepTensor.toKernel3d());
        // std::cout << "After RHSB = \n" << GpuOut<Real>(rTensor, hand) << std::endl;

        auto rMatT = rhs.matrix(dim.rows * dim.layers);
        auto xMatT = xThirdStep.matrix(dim.rows * dim.layers);
        thomasRows.solveLaplacianTranspose(xMatT, rMatT, dim.numDims() == 3, hand);
        // std::cout << "x 2/3 step = \n" << GpuOut<Real>(xThirdStepTensor, hand) << std::endl;


        setRHSC<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(rTensor.toKernel3d(), xTensor.toKernel3d(), xThirdStepTensor.toKernel3d());
        // std::cout << "After RHSC = \n" << GpuOut<Real>(rTensor, hand) << std::endl;
        thomasDepths.solveLaplacianDepths(xTensor, rTensor, hand);
        // std::cout << "x at end = \n" << GpuOut<Real>(xTensor, hand) << std::endl;


        residual3dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
            xTensor.toKernel3d(),
            bTensor.toKernel3d(),
            rTensor.toKernel3d()
        );

        rhs.norm(rNorm, hand);

        // std::cout << GpuOut<Real>(r, hand) << std::endl;

        if (rNorm.get(hand)/*/bNormHost*/ < tolerance) break;
        std::cout << "x = " << GpuOut<Real>(x, hand) << std::endl;
    }
    if (i == maxIterations) std::cout << "ADIThomas reached it's maximum number of iterations: " << maxIterations << std::endl;
}

template<typename Real>
void ADIThomas<Real>::test() {
    std::cout << std::setprecision(15);

    GridDim dim(3, 2, 2);

    Handle hand;
    auto x = SimpleArray<Real>::create(dim.size(), hand);
    x.fill(100, hand);
    auto b = SimpleArray<Real>::create(dim.size(), hand);
    auto scratch = Mat<Real>::create(dim.size(), 2);
    std::vector<Real> bHost = {1,2,3,4,5,6,7,8,9,10,11,12};
    b.set(bHost.data(), hand);

    ADIThomas adiThomas(dim, 20, 1e-5, scratch, hand);
    adiThomas.solve(x, b, hand);

    ToeplitzLaplacian<Real>::printL(dim, hand);

    std::cout << "b = " << GpuOut<Real>(b, hand) << std::endl;

    std::cout << "x = " << GpuOut<Real>(x, hand) << std::endl;

    std::vector<Real> actualSolution = {-1.1449579831932773, -1.7268907563025210, -1.6449579831932773, -1.7626050420168067, -2.4327731092436975, -2.2626050420168067, -2.3802521008403361, -3.1386554621848739, -2.8802521008403361, -2.9978991596638655, -3.8445378151260504, -3.4978991596638655};
    std::vector<Real> result(12, 0);
    x.get(result.data(), hand);

    double r = 0;
    for (size_t i = 0; i < result.size(); ++i) r += (result[i] - actualSolution[i]) * (result[i] - actualSolution[i]);
    std::cout << "r = " << std::sqrt(r) << std::endl;


    std::cout << "expected: " << std::endl;
    for (size_t i = 0; i < result.size(); ++i) std::cout << actualSolution[i] << " ";
    std::cout << std::endl;
}


template class ADIThomas<double>;
template class ADIThomas<float>;
