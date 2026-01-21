#include <iostream>
#include <ostream>

#include "Streamable.h"
#include <vector>

#include "SparseCSC.cuh"
#include "BaseData.h"
#include "ToeplitzLaplacian.cuh"
#include "../solvers/EigenDecompSolver.h"


template <typename Real, typename Ind>
void solveImmersedBody(size_t gridHeight, size_t gridWidth, size_t gridDepth, size_t fSize, size_t nnzB, Ind* colsB, Ind* rowsB, Real* valsB, Real* f, Real* p, double deltaX, double deltaY, double deltaZ, Real* result) {
    Handle hand3[3]{};

    const GridDim dim(gridHeight, gridWidth, gridDepth);

    auto resultDevice = SimpleArray<Real>::create(dim.volume(), hand3[0]);

    BaseData<Real, Ind> baseData(dim, fSize, nnzB, colsB,rowsB, valsB, f, p, deltaX, deltaY, deltaZ, hand3[0]);

    ImmersedEq<Real, Ind> imEq(baseData, dim, hand3);

    ImmersedEqSolver<Real, Ind>::solve(imEq, resultDevice);

    resultDevice.get(result, hand3[0]);
}

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

void printL(const GridDim& dim, Handle* hand4, size_t numInds) {
    auto spaceForA = Mat<double>::create(dim.volume(), numInds);
    auto inds = SimpleArray<int32_t>::create(numInds, hand4[0]);
    auto A = ToeplitzLaplacian<double>(dim).setL(hand4[0], spaceForA, inds, Real3d(1, 1, 1));
    auto aDense = SquareMat<double>::create(dim.volume());
    A.getDense(aDense, &hand4[0]);
    std::cout << "L = \n" << GpuOut<double>(aDense, hand4[0]) << std::endl;
}

void smallTestWithoutFiles() {
    Handle hand4[4]{};
    GridDim dim(3, 2, 2);
    Real3d delta(1,1,1);

    constexpr size_t size = 12;

    size_t numInds = 7, fSize = 2;

    printL(dim, hand4, numInds);

    std::vector<size_t> rowPointers = {0};
    std::vector<double> values = {1};
    std::vector<size_t> colOffsets = {0,1,1};

    std::vector<double> f = {1,2};
    std::vector<double> p = {-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2};

    double result[size];

    ImmersedEq<double, size_t> imEq(dim, hand4, f.size(), values.size(), p.data(), f.data(), delta, 1e-6, 1000);

    imEq.solve(result, 1, rowPointers.data(), colOffsets.data(), values.data());

    for(int i = 0; i < size; ++i) std::cout << result[i] << " ";


    // std::cout << "LHS of equation is\n" << GpuOut<double>(imEq.LHSMat(hand4[0]), hand4[0]) << std::endl;
    // std::cout << "RHS of equation is\n" << GpuOut<double>(imEq.RHS, hand4[0]) << std::endl;

    // std::cout << "Result is \n" << GpuOut<double>(result, hand4[0]) << std::endl;
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
    smallTestWithoutFiles();
    // benchmark(3);

}

