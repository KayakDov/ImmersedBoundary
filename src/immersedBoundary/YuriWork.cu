#include <iostream>
#include <ostream>

#include "Streamable.h"
#include <vector>


#include "FortranBindings.hpp"
#include "ToeplitzLaplacian.cuh"
#include "../solvers/EigenDecompSolver.h"
#include "solvers/BiCGSTAB.cuh"


// void benchmark(size_t maxSize) {
//     for (size_t dim = 2; dim < maxSize; dim++) {
//         size_t dim3 = dim * dim * dim;
//         std::vector<size_t> rows;
//         std::vector<size_t> cols;
//         std::vector<double> vals;
//         std::vector<double> f;
//         std::vector<double> p;
//
//         for (size_t i = 0; i < dim3; i++ ) {
//             rows.push_back(1);
//             cols.push_back(1);
//             vals.push_back(1);
//             f.push_back(1);
//             p.push_back(1);
//         }
//
//         cols.push_back(1);
//         std::unique_ptr<double[]> result(new double[dim3]);
//
//         cudaDeviceSynchronize();
//         auto start = std::chrono::high_resolution_clock::now();
//         solveImmersedBody(dim, dim, dim, dim3, dim3, cols.data(), rows.data(), vals.data(), f.data(), p.data(), result.get());
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double, std::milli> duration = end - start;
//         std::cout << "Time: " << duration.count()  << std::endl;
//
//     }
// }

template<typename Real>
void printL(const GridDim &dim, Handle &hand) {
    size_t numInds = 7;
    auto spaceForA = Mat<Real>::create(dim.size(), numInds);
    auto inds = SimpleArray<int32_t>::create(numInds, hand);
    auto A = ToeplitzLaplacian<Real>(dim).setL(hand, spaceForA, inds, Real3d(1, 1, 1));
    auto aDense = SquareMat<Real>::create(dim.size());
    A.getDense(aDense, &hand);
    std::cout << "L = \n" << GpuOut<Real>(aDense, hand) << std::endl;
}

template<typename Real, typename Int>
void testPrimes() {
    GridDim dim(3, 2, 1);
    Real3d delta(1, 1, 1);

    Handle hand;

    constexpr size_t size = 6;
    constexpr size_t fSize = 2;

    size_t uStarSize = dim.numDims() * dim.size()
        + dim.cols * dim.layers
        + dim.rows * dim.layers
        + dim.cols * dim.rows * (dim.layers > 1);

    std::vector<Real> uStar(uStarSize, 0);
    uStar[3] = 1;
    uStar[9] = 1;

    std::vector<Int> colOffsetsR(3);
    colOffsetsR[0] = 0;
    colOffsetsR[1] = uStarSize;
    colOffsetsR[2] = uStarSize;

    std::vector<Int> rowIndsR(uStarSize);
    for (size_t i = 0; i < rowIndsR.size(); i++) rowIndsR[i] = i;

    std::vector<Real> valsR(uStarSize, 1);

    std::vector<Real> UGamma(2, 3);

    double deltaT = 3.0/2.0;

    // printL<Real>(dim, hand);

    std::vector<Int> rowOffsetsB(3);
    rowOffsetsB[0] = 0;
    rowOffsetsB[1] = 1;
    rowOffsetsB[2] = 2;

    std::vector<Real> valuesB = {1, 1};
    std::vector<Int> colIndsB(2);
    colIndsB[0] = 0;
    colIndsB[1] = 1;

    std::vector<Real> f = {1,2};
    std::vector<Real> p(size, 0);
    p[0] = 2;
    p[size - 1] = -2;

    Real resultP[size];
    Real resultF[fSize];

    auto start = std::chrono::high_resolution_clock::now();

    ImmersedEq<Real, Int> imEq(dim, f.size(), valsR.size(), p.data(), f.data(), delta, deltaT, 1e-6, 1000);
    // initImmersedEq<Real, Int>(dim.rows, dim.cols, dim.layers, f.size(), f.size(), p.data(), f.data(), delta.x, delta.y, delta.z, 1e-6, 3000);

    // imEq.solve(resultP, valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data(), true);
    imEq.solve(resultP, resultF, valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data(), valsR.size(), colOffsetsR.data(), rowIndsR.data(), valsR.data(), UGamma.data(), uStar.data(), true);
    // solveImmersedEq<Real, Int>(result, values.size(), rowPointers.data(), colOffsets, values.data(), true);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Total Solver Time: " << duration.count() << " ms" << std::endl;

    std::cout << "resultP = ";
    for(auto & i : resultP) std::cout << i << " ";

    std::cout << "\nexpected 7.61797, 10.1498, 2.955056, 3.08614, 3.72659, 1.67041" << std::endl;

    std::cout << "resultF = ";
    for(auto & i : resultF) std::cout << i << " ";

    std::cout << "\nexpected 17.23595, 26.29962" << std::endl;

    // auto denseB = Mat<Real>::create(f.size(), p.size());
    // imEq.baseData.B->getDense(denseB, hand);
    // std::cout << "\nB = \n" << GpuOut<Real>(denseB, hand) << std::endl;

    // auto inverseL = imEq.eds->inverseL(hand);
    // std::cout << "L^-1 = \n" << GpuOut<Real>(inverseL, hand) << std::endl;
    // std::cout << "LHS of equation is\n" << GpuOut<Real>(imEq.LHSMat(), hand) << std::endl;
    // std::cout << "RHS of equation is\n" << GpuOut<Real>(imEq.RHS(false), hand) << std::endl;

}

template<typename Real, typename Int>
void smallTestWithoutFiles() {
    GridDim dim(3, 2, 1);
    Real3d delta(1, 1, 1);

    Handle hand;

    constexpr size_t size = 6;


    // printL<Real>(dim, hand);

    std::vector<Int> rowOffsets = {0, 1, 2};
    std::vector<Real> values = {1, 1};
    std::vector<Int> colInds = {0, 1};

    std::vector<Real> f = {1,2};
    std::vector<Real> p(size, 0);
    p[0] = 2;
    p[size - 1] = -2;

    Real result[size];
    auto start = std::chrono::high_resolution_clock::now();
    ImmersedEq<Real, Int> imEq(dim, f.size(), values.size(), p.data(), f.data(), delta, 1, 1e-6, 1000);
    // initImmersedEq<Real, Int>(dim.rows, dim.cols, dim.layers, f.size(), f.size(), p.data(), f.data(), delta.x, delta.y, delta.z, 1e-6, 3000);

    imEq.solve(result, values.size(), rowOffsets.data(), colInds.data(), values.data(), true);
    // solveImmersedEq<Real, Int>(result, values.size(), rowPointers.data(), colOffsets, values.data(), true);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Total Solver Time: " << duration.count() << " ms" << std::endl;

    std::cout << "result = ";
    for(auto & i : result) std::cout << i << " ";

    std:: cout << "\nexpected: -7.483126, -8.359545, -2.292128, -2.606740, -2.943816, -0.808988" << std::endl;



    // auto denseB = Mat<Real>::create(f.size(), p.size());
    // imEq.baseData.B->getDense(denseB, hand);

    // std::cout << "\nB = \n" << GpuOut<Real>(denseB, hand) << std::endl;
    // auto inverseL = imEq.eds->inverseL(hand);
    // std::cout << "L^-1 = \n" << GpuOut<Real>(inverseL, hand) << std::endl;
    // std::cout << "LHS of equation is\n" << GpuOut<Real>(imEq.LHSMat(), hand) << std::endl;
    // std::cout << "RHS of equation is\n" << GpuOut<Real>(imEq.RHS(false), hand) << std::endl;

}

// void testOnFiles(const GridDim& dim) {
//     Handle hand2[2]{};
//
//     std::ifstream inX("../dataFromYuri/x_f64.bin", std::ios::binary);
//     std::ifstream inB("../dataFromYuri/B_csc.bin", std::ios::binary);
//
//     FileMeta fm(inX, inB);
//
//     BaseData<double> bd(fm, hand2);
//
//     cudaDeviceSynchronize();
     // auto start = std::chrono::high_resolution_clock::now();
//
//     ImmersedEq<double> imEq(bd, dim, hand2);
//
//     ImmersedEqSolver<double> solver(imEq);
//
//     auto result = SimpleArray<double>::create(fm.pSize, hand2[0]);
//     solver.solveUnpreconditionedBiCGSTAB(result);
//
//     cudaDeviceSynchronize();
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> duration = end - start;
//     std::cout << "Total Solver Time: " << duration.count() << " ms" << std::endl;
//
//     std::ofstream outResult("../dataFromYuri/result.bin", std::ios::binary);
//     outResult << GpuOut<double>(result, hand2[0], false, true);
// }


int main(int argc, char *argv[]) {
    // testOnFiles(GridDim(2000, 2000, 1));
    // smallTestWithoutFiles<double, int64_t>();

    testPrimes<double, int32_t>();
    // BCGBanded<double>::test();

    // BCGDense<double>::test();


}
