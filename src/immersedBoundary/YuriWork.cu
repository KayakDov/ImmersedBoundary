#include <iostream>
#include <ostream>

#include "../deviceArrays/headers/Support/Streamable.h"
#include <vector>


#include "FortranBindings.hpp"
#include "../deviceArrays/headers/sparse/SparseCOO.h"
#include "ToeplitzLaplacian.cuh"
#include "../solvers/EigenDecomp/EigenDecompSolver.h"
#include "solvers/BiCGSTAB.cuh"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

#include "solvers/ADIThomas.cuh"
#include "solvers/Thomas.cuh"


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

    GridDim dim(1000, 1000, 1);

    LoadBHost b;

    auto cooB = SparseCOO<Real, int32_t>::create(b.bVals.size(), dim.size(), dim.size(), hand);
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

    auto A = ToeplitzLaplacian<Real>::L(dim, hand);

    LoadRHSHost rhs;

    std::cout << "f size = " << rhs.FHost.size() << std::endl;

    cudaDeviceSynchronize();

    Real3d delta(1.0/1000, 1.0/1000, 1.0/1000);

    std::vector<Real> result(dim.size());

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

template<typename Real, typename Int>
void testPrimes() {
    GridDim dim(3, 2, 1);
    Real3d delta(1, 1, 1);

    Handle hand;

    //set u*
    size_t uStarSize = dim.numDims() * dim.size()
        + dim.cols * dim.layers
        + dim.rows * dim.layers
        + dim.cols * dim.rows * (dim.layers > 1);

    std::vector<Real> uStar(uStarSize, 0);
    uStar[3] = 1;
    uStar[9] = 1;

    //set R
    std::vector<Int> colOffsetsR(3);
    colOffsetsR[0] = 0;
    colOffsetsR[1] = uStarSize;
    colOffsetsR[2] = uStarSize;

    std::vector<Int> rowIndsR(uStarSize);
    for (size_t i = 0; i < rowIndsR.size(); i++) rowIndsR[i] = i;

    std::vector<Real> valsR(uStarSize, 1);

    //set UGamma
    std::vector<Real> UGamma(2, 3);

    //delta T
    double deltaT = 3.0/2.0;

    // printL<Real>(dim, hand);

    //Set B
    std::vector<Int> rowOffsetsB(3, 0);
    rowOffsetsB[0] = 0;
    rowOffsetsB[1] = 1;
    rowOffsetsB[2] = 2;

    std::vector<Real> valuesB = {1, 1};
    std::vector<Int> colIndsB = {0, 1};

    //setF
    std::vector<Real> f = {1,2};

    //set p
    std::vector<Real> p(dim.size(), 0);
    p[0] = 2;
    p[dim.size() - 1] = -2;

    //set result
    std::vector<Real> resultP(dim.size(), 0);
    std::vector<Real> resultF(f.size(), 0);

    auto start = std::chrono::high_resolution_clock::now();

    ImmersedEq<Real, Int> imEq(dim, f.size(), valsR.size(), p.data(), f.data(), delta, deltaT, 1e-6, 1000);
    // initImmersedEq<Real, Int>(dim.rows, dim.cols, dim.layers, f.size(), f.size(), p.data(), f.data(), delta.x, delta.y, delta.z, 1e-6, 3000);

    imEq.solve(resultP.data(), valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data());
    std::cout << "resultP = ";
    for(auto & i : resultP) std::cout << i << " ";
    std:: cout << "\nexpected for 3 x 2 x 1: -7.483126, -8.359545, -2.292128, -2.606740, -2.943816, -0.808988" << std::endl;
    // std:: cout << "\nexpected for 3 x 2 x 2: -1.58481048081, -1.6484376726, -0.29435676885, -0.34540212532, -0.35729172036, -0.05885147027, -0.34540212532, -0.35729172036, -0.05885147027, -0.13031055077, 0.29853966757" << std::endl;

    imEq.solve(resultP.data(), resultF.data(), valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data(), valsR.size(), colOffsetsR.data(), rowIndsR.data(), valsR.data(), UGamma.data(), uStar.data());

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Total Solver Time: " << duration.count() << " ms" << std::endl;


    std::cout << "resultP = ";
    for(size_t i = 0; i < resultP.size(); ++i) std::cout << resultP[i] << " ";
    std::cout << "\nexpected 3X2X1:  7.61797, 10.1498, 2.955056, 3.08614, 3.72659, 1.67041" << std::endl;// solveImmersedEq<Real, Int>(result, values.size(), rowPointers.data(), colOffsets, values.data());
    std::cout << "resultF = ";
    for(size_t i = 0; i < resultF.size(); ++i) std::cout << resultF[i] << " ";
    std::cout << "\nexpected 3X2X1: 17.23595, 26.29962" << std::endl;

    // auto denseB = Mat<Real>::create(f.size(), p.size());
    // imEq.baseData.B->getDense(denseB, hand);
    // std::cout << "\nB = \n" << GpuOut<Real>(denseB, hand) << std::endl;

    // auto inverseL = imEq.eds->inverseL(hand);
    // std::cout << "L^-1 = \n" << GpuOut<Real>(inverseL, hand) << std::endl;
    // std::cout << "LHS of equation is\n" << GpuOut<Real>(imEq.LHSMat(), hand) << std::endl;
    // std::cout << "RHS of equation is\n" << GpuOut<Real>(imEq.RHS(false), hand) << std::endl;

}

// int main(int argc, char *argv[]) {

    // testPrimes<double, int32_t>();

    // loadYuriData<double>();

    // ADIThomas<double>::test2();
// }
