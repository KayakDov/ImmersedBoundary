#include <iostream>
#include <ostream>

#include "../deviceArrays/headers/Support/Streamable.h"
#include <vector>


#include "FortranBindings.hpp"
#include "../deviceArrays/headers/sparse/SparseCOO.h"
#include "ToeplitzLaplacian.cuh"
#include "../solvers/EigenDecompSolver.h"
#include "solvers/BiCGSTAB.cuh"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>


//TODO: ensure all cuda operations that use 0 to clear the space also clear nans.
class LoadBHost {
public:
    std::vector<int32_t> bRows, bCols;
    std::vector<double> bVals;

    LoadBHost(std::string bPath = "../dataFromYuri/B.dat", size_t defSize = 40000) {
        bRows.reserve(defSize);
        bCols.reserve(defSize);
        bVals.reserve(defSize);

        std::ifstream bFile(bPath);

        if (bFile.is_open()) {
            int32_t r, c;
            double v;
            while (bFile >> r >> c >> v) {
                bRows.push_back(r - 1);
                bCols.push_back(c - 1);
                bVals.push_back(v);
            }
            bFile.close();
            std::cout << "Loaded B.dat: " << bRows.size() << " entries." << std::endl;
        } else {
            std::cerr << "Unable to open B.dat at " << bPath << std::endl;
        }
    }

    size_t nnz() {
        return bVals.size();
    }
};

class LoadRHSHost {
public:
    std::vector<double> pHost;
    std::vector<double> FHost;

    LoadRHSHost(std::string rhsPath = "../dataFromYuri/rhs.dat", size_t defSizeP = 1000000, size_t defSizeF = 1300) {
        std::ifstream rhsFile(rhsPath);
        pHost.reserve(defSizeP);
        FHost.reserve(defSizeF);

        if (rhsFile.is_open()) {
            int32_t idx;
            double val;
            while (pHost.size() < 1000000 && rhsFile >> idx >> val) pHost.push_back(val);
            while (rhsFile >> idx >> val) FHost.push_back(val);

            rhsFile.close();
            std::cout << "Loaded rhs.dat: Total " << FHost.size() + pHost.size() << " rows." << std::endl;
            std::cout << "pHost size: " << pHost.size() << ", FHost size: " << FHost.size() << std::endl;
        } else {
            std::cerr << "Unable to open rhs.dat at " << rhsPath << std::endl;
        }
    }
};

template<typename Real>
void loadYuriData() {

    Handle hand;

    size_t n = 1000000;

    LoadBHost b;

    auto cooB = SparseCOO<Real, int32_t>::create(b.bVals.size(), n, n, hand);
    cooB.set(b.bRows.data(), b.bCols.data(), b.bVals.data(), hand);

    auto offsets = SimpleArray<int32_t>::create(cooB.rows + 1, hand);
    auto nnzAllocated = SimpleArray<int32_t>::create(b.nnz(), hand);
    std::unique_ptr<SimpleArray<Real>> buffer = nullptr;

    auto csrB = cooB.getCSR(offsets, nnzAllocated, buffer, hand);

    std::vector<Real> bVals(csrB.values.size());
    csrB.values.get(bVals.data(), hand);
    std::vector<int32_t> bColInds(csrB.inds.size());
    csrB.inds.get(bColInds.data(), hand);
    std::vector<int32_t> bRowOffsets(csrB.offsets.size());
    csrB.offsets.get(bRowOffsets.data(), hand);

    LoadRHSHost rhs;

    std::cout << "f size = " << rhs.FHost.size() << std::endl;

    cudaDeviceSynchronize();

    std::vector<Real> result(n);

    GridDim dim(1000, 1000, 1);
    Real3d delta(1.0/1000, 1.0/1000, 1.0/1000);

    auto start = std::chrono::high_resolution_clock::now();
    ImmersedEq<Real, int32_t> imEq(dim, rhs.FHost.size(), b.nnz(), rhs.pHost.data(), rhs.FHost.data(), delta, 1, 1e-18, 100);
    imEq.solve(result.data(), b.nnz(), bRowOffsets.data(), bColInds.data(), bVals.data());

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Total Solver Time: " << duration.count() << " ms" << std::endl;

    std::ofstream resFile("../dataFromYuri/result.dat");
    if (resFile.is_open()) {
        for (size_t i = 0; i < result.size(); ++i) {
            resFile << i << " " << std::scientific << result[i] << "\n";
        }
        resFile.close();
        std::cout << "Saved results to result.dat" << std::endl;
    }

}



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

    imEq.solve(resultP, valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data());
    std::cout << "resultP = ";
    for(auto & i : resultP) std::cout << i << " ";
    std:: cout << "\nexpected: -7.483126, -8.359545, -2.292128, -2.606740, -2.943816, -0.808988" << std::endl;

    imEq.solve(resultP, resultF, valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data(), valsR.size(), colOffsetsR.data(), rowIndsR.data(), valsR.data(), UGamma.data(), uStar.data());
    std::cout << "resultP = ";
    for(auto & i : resultP) std::cout << i << " ";
    std::cout << "\nexpected 7.61797, 10.1498, 2.955056, 3.08614, 3.72659, 1.67041" << std::endl;// solveImmersedEq<Real, Int>(result, values.size(), rowPointers.data(), colOffsets, values.data());

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Total Solver Time: " << duration.count() << " ms" << std::endl;



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

    imEq.solve(result, values.size(), rowOffsets.data(), colInds.data(), values.data());
    // solveImmersedEq<Real, Int>(result, values.size(), rowPointers.data(), colOffsets, values.data());

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
//     outResult << GpuOut<double>(result, hand2[0], false);
// }


int main(int argc, char *argv[]) {

    // smallTestWithoutFiles<double, int64_t>();
    testPrimes<double, int32_t>();

    loadYuriData<double>();
}
