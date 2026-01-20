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

void benchmark(size_t maxSize) {
    for (size_t dim = 2; dim < maxSize; dim++) {
        size_t dim3 = dim * dim * dim;
        std::vector<size_t> rows;
        std::vector<size_t> cols;
        std::vector<double> vals;
        std::vector<double> f;
        std::vector<double> p;

        for (size_t i = 0; i < dim3; i++ ) {
            rows.push_back(1);
            cols.push_back(1);
            vals.push_back(1);
            f.push_back(1);
            p.push_back(1);
        }

        cols.push_back(1);
        std::unique_ptr<double[]> result(new double[dim3]);

        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        solveImmersedBody(dim, dim, dim, dim3, dim3, cols.data(), rows.data(), vals.data(), f.data(), p.data(), result.get());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Time: " << duration.count()  << std::endl;

    }
}

void smallTestWithoutFiles() {
    Handle hand3[3]{};

    GridDim dim(3, 2, 2);
    size_t numInds = 7, fSize = 2;

    auto spaceForA = Mat<double>::create(dim.volume(), numInds);
    auto inds = SimpleArray<int32_t>::create(numInds, hand3[0]);
    auto A = ToeplitzLaplacian<double>(dim).setL(hand3[0], spaceForA, inds, Real3d(1, 1, 1));
    auto aDense = SquareMat<double>::create(dim.volume());
    A.getDense(aDense, &hand3[0]);
    std::cout << "L = \n" << GpuOut<double>(aDense, hand3[0]) << std::endl;

    auto B = SparseCSC<double>::create(1, fSize, dim.volume(), hand3[0]);

    std::vector<uint32_t> rowPointersHost = {0};
    B.rowPointers.set(rowPointersHost.data(), hand3[0]);

    B.columnOffsets.subAray(0, 1).fill(0, hand3[0]);
    B.columnOffsets.subAray(1, B.columnOffsets.size() - 1).fill(1, hand3[0]);

    B.values.fill(1, hand3[0]);

    auto denseB = Mat<double>::create(fSize, dim.volume());
    B.getDense(denseB, hand3[0]);
    std::cout << "dense B = \n" << GpuOut<double>(denseB, hand3[0]) << std::endl;

    auto xf = SimpleArray<double>::create(fSize, hand3[0]);
    std::vector<double> fHost = {1, 2};
    xf.set(fHost.data(), hand3[0]);

    auto xp = SimpleArray<double>::create(dim.volume(), hand3[0]);
    std::vector<double> xpHost = {-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2};
    xp.set(xpHost.data(), hand3[0]);

    auto result = SimpleArray<double>::create(dim.volume(), hand3[0]);
    BaseData<double> bd(B, xf, xp, dim.volume());

    ImmersedEq<double> imEq(bd, dim, hand3);

    std::cout << "LHS of equation is\n" << GpuOut<double>(imEq.LHSMat(hand3[0]), hand3[0]) << std::endl;
    std::cout << "RHS of equation is\n" << GpuOut<double>(imEq.RHS, hand3[0]) << std::endl;

    ImmersedEqSolver<double> solver(imEq);

    solver.solveUnpreconditionedBiCGSTAB(result);

    std::cout << "Result is \n" << GpuOut<double>(result, hand3[0]) << std::endl;

    // std::cout << "Expected: (0.5, 0, 0, -0.5)" << std::endl;
}

void testOnFiles(const GridDim& dim) {
    Handle hand2[2]{};

    std::ifstream inX("../dataFromYuri/x_f64.bin", std::ios::binary);
    std::ifstream inB("../dataFromYuri/B_csc.bin", std::ios::binary);

    FileMeta fm(inX, inB);

    BaseData<double> bd(fm, hand2);

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    ImmersedEq<double> imEq(bd, dim, hand2);

    ImmersedEqSolver<double> solver(imEq);

    auto result = SimpleArray<double>::create(fm.pSize, hand2[0]);
    solver.solveUnpreconditionedBiCGSTAB(result);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Total Solver Time: " << duration.count() << " ms" << std::endl;

    std::ofstream outResult("../dataFromYuri/result.bin", std::ios::binary);
    outResult << GpuOut<double>(result, hand2[0], false, true);
}



int main(int argc, char *argv[]) {
    // testOnFiles(GridDim(2000, 2000, 1));
    // smallTestWithoutFiles();
    benchmark(3);

}

