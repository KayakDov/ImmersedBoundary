
#include <chrono>
#include "poisson/CubeBoundary.h"
#include "solvers/EigenDecompSolver.h"
// #include "FortranBindings.hpp"
#include "ToeplitzLaplacian.cuh"
#include "wrapFortranBindings.h"
using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

template<typename T>
void benchMarkEigenDecompSolver(size_t height, size_t width, size_t depth, std::array<Handle, 3>& hand3) {
    const auto boundary = CubeBoundary<double>::ZeroTo1(height, width, depth, hand3[0]);

    auto memAlocFX = Mat<double>::create(boundary.internalSize(), 2);


    size_t maxDim = std::max(width, std::max(height, depth));

    Mat<T> eigenStorage = Mat<T>::create(maxDim, width + height + depth + 3);
    SquareMat<T> eX = eigenStorage.sqSubMat(0, 0, width),
            eY = eigenStorage.sqSubMat(0, width, height),
            eZ = eigenStorage.sqSubMat(0, width + height, depth);

    Mat<T> vals = eigenStorage.subMat(0, width + height + depth, maxDim, 3);

    auto x = memAlocFX.col(0);
    auto f = memAlocFX.col(1);

    f.fill(0, hand3[0]);

    cudaDeviceSynchronize();

    TimePoint start = std::chrono::steady_clock::now();
    PoissonRHS<double> poisson(boundary, f, hand3[0]);
    EigenDecompSolver3d<double> fdm(x, f, eX, eY, eZ, vals, hand3);
    fdm.solve(x, f, hand3[0]);
    cudaDeviceSynchronize();
    TimePoint end = std::chrono::steady_clock::now();


    double iterationTime = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();

    std::cout << iterationTime << ", ";

    // std::cout << "x = \n" << GpuOut<double>(x.tensor(height, depth), hand3[0]) << std::endl;
}

template<typename T>
void benchMarkEigenDecompSolver(size_t dim, std::array<Handle, 3>& hand3) {
    benchMarkEigenDecompSolver<T>(dim, dim, dim, hand3);
}

template<typename T>
void testEigenDecompNoPoisson() {

    Handle hand;

    GridDim dim(2, 3, 1);

    size_t numDiags = dim.layers == 1? 5 : 7;
    auto preA = Mat<T>::create(dim.size(), numDiags);

    auto preInds = Vec<int32_t>::create(numDiags);

    BandedMat<T> A = ToeplitzLaplacian<T>(dim).setA(hand, preA, preInds);

    std::cout << "A matrices math= \n" << GpuOut(A, hand) << std::endl;

    std::cout << "inds = \n" << GpuOut(A._indices, hand) << std::endl;

    auto b = Vec<T>::create(dim.size(), hand);
    auto x = Vec<T>::create(dim.size(), hand);

    std::vector<T> hostB = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    b.set(hostB.data(), hand);

    std::cout << "b = " << GpuOut(b, hand) << std::endl;

    // BiCGSTAB<T>::solve(A, b);

    auto rowsMat = SquareMat<T>::create(dim.rows, hand);
    auto colsMat = SquareMat<T>::create(dim.cols, hand);
    auto layersMat = SquareMat<T>::create(dim.layers, hand);
    auto valsMat = Mat<T>::create(dim.size(), 3);

    cudaDeviceSynchronize();
    std::array<Handle, 2/*3*/> hand3{};

    EigenDecompSolver2d<T> eds(rowsMat, colsMat/*, layersMat*/, valsMat, hand3);

    eds.solve(x, b, hand3[0]);

    std::cout << "x = \n" << GpuOut(x, hand) << std::endl;
    std::cout << "3d results should be\n" << -241.0/238 << ", " << -145.0/119 << ", " << -339.0/238 << ", " << -194.0/119 << std::endl;
    std::cout << "2d results should be\n" << -83.0/69 << ", " << -101.0/69 << ", " << -54.0/23 << std::endl;
    //x_1=(-241)/238
    // x_2=(-145)/119
    // x_3=(-339)/238
    // x_4=(-194)/119
    // x_5=(-579)/238
    // x_6=(-635)/238
    // x_7=(-691)/238
    // x_8=(-747)/238
    // x_9=(-717)/238
    // x_10=(-383)/119
    // x_11=(-815)/238
    // x_12=(-432)/119
}

/**
 * @brief Main entry point to demonstrate the FastDiagonalizationMethod
 *        for a 2x2x2 grid.
 */
// int main() {
    // std::array<Handle, 3> hand3;
    //
    // constexpr size_t maxDimensions = 5;
    //
    // std::cout << "dim, time" << std::endl;
    //
    // for (size_t dim = 1; dim < maxDimensions; dim++) {
    //     std::cout << dim << ", ";
    //     benchMarkEigenDecompSolver<double>(dim, hand3);
    //     cudaDeviceSynchronize();
    //
    //     testPoisson(dim, hand3[0]);
    //     std::cout << std::endl;
    // }

//     testEigenDecompNoPoisson<double>();
//
//
//
//     return 0;
// }


