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
void smallTestWithoutFiles() {
    GridDim dim(3, 2, 1);
    Real3d delta(1, 1, 1);

    Handle hand;

    constexpr size_t size = 6;


    // printL<Real>(dim, hand);

    std::vector<Int> rowPointers = {0, 1};
    std::vector<Real> values = {1, 1};
    Int colOffsets [size + 1];
    colOffsets[0] = 0;
    colOffsets[1] = 1;
    for (size_t i = 2; i < size + 1; ++i) colOffsets[i] = static_cast<Int>(2);

    std::vector<Real> f = {1,2};
    std::vector<Real> p(size, 0);
    p[0] = 2;
    p[size - 1] = -2;

    Real result[size];

    ImmersedEq<Real, Int> imEq(dim, f.size(), values.size(), p.data(), f.data(), delta, 1e-6, 1000);
    // initImmersedEq<Real, Int>(dim.rows, dim.cols, dim.layers, f.size(), f.size(), p.data(), f.data(), delta.x, delta.y, delta.z, 1e-6, 3000);

    imEq.solve(result, values.size(), rowPointers.data(), colOffsets, values.data(), true);
    // solveImmersedEq<Real, Int>(result, values.size(), rowPointers.data(), colOffsets, values.data(), true);

    std::cout << "result = ";
    for(auto & i : result) std::cout << i << " ";

    std:: cout << "\nexpected: -7.483126, -8.359545, -2.292128, -2.606740, -2.943816, -0.808988" << std::endl;


    cudaDeviceSynchronize();
    auto denseB = Mat<Real>::create(f.size(), p.size());
    imEq.baseData.B->getDense(denseB, hand);

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
//     auto start = std::chrono::high_resolution_clock::now();
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
    smallTestWithoutFiles<double, int32_t>();
    // benchmark(3);
    // BCGBanded<double>::test();

    // BCGDense<double>::test();
}
