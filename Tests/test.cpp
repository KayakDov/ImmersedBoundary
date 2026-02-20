#include <gtest/gtest.h>
#include "../src/solvers/EigenDecomp/EigenDecomp3d.cuh"

#include "immersedBoundary/ImerssedEquation.h"

TEST(ImmersedEq, SolvesPrimes_3x2x1) {
    using Real = double;
    using Int  = int;

    GridDim dim(3, 2, 1);
    Real3d delta(1, 1, 1);
    Handle hand;

    // ----------------------------
    // Build u*
    // ----------------------------
    size_t uStarSize =
        dim.numDims() * dim.size()
        + dim.cols * dim.layers
        + dim.rows * dim.layers
        + dim.cols * dim.rows * (dim.layers > 1);

    std::vector<Real> uStar(uStarSize, 0);
    uStar[3] = 1;
    uStar[9] = 1;

    // ----------------------------
    // Build R (identity-like)
    // ----------------------------
    std::vector<Int> colOffsetsR(3);
    colOffsetsR[0] = 0;
    colOffsetsR[1] = uStarSize;
    colOffsetsR[2] = uStarSize;

    std::vector<Int> rowIndsR(uStarSize);
    for (size_t i = 0; i < uStarSize; ++i)
        rowIndsR[i] = static_cast<Int>(i);

    std::vector<Real> valsR(uStarSize, 1);

    // ----------------------------
    // UGamma
    // ----------------------------
    std::vector<Real> UGamma(2, 3);

    double deltaT = 3.0 / 2.0;

    // ----------------------------
    // Build B (CSR)
    // ----------------------------
    std::vector<Int> rowOffsetsB = {0, 1, 2};
    std::vector<Int> colIndsB    = {0, 1};
    std::vector<Real> valuesB    = {1, 1};

    // ----------------------------
    // RHS and p
    // ----------------------------
    std::vector<Real> f = {1, 2};

    std::vector<Real> p(dim.size(), 0);
    p[0] = 2;
    p[dim.size() - 1] = -2;

    std::vector<Real> resultP(dim.size(), 0);
    std::vector<Real> resultF(f.size(), 0);

    ImmersedEq<Real, Int> imEq(
        dim,
        f.size(),
        valsR.size(),
        p.data(),
        f.data(),
        delta,
        deltaT,
        1e-6,
        1000
    );

    // ----------------------------
    // First solve
    // ----------------------------
    imEq.solve(
        resultP.data(),
        valuesB.size(),
        rowOffsetsB.data(),
        colIndsB.data(),
        valuesB.data()
    );

    cudaDeviceSynchronize();

    // Expected values (3x2x1)
    std::vector<Real> expectedP1 = {
        -7.483126, -8.359545,
        -2.292128, -2.606740,
        -2.943816, -0.808988
    };

    for (size_t i = 0; i < resultP.size(); ++i)
        EXPECT_NEAR(resultP[i], expectedP1[i], 1e-5);

    // ----------------------------
    // Full solve
    // ----------------------------
    imEq.solve(
        resultP.data(),
        resultF.data(),
        valuesB.size(),
        rowOffsetsB.data(),
        colIndsB.data(),
        valuesB.data(),
        valsR.size(),
        colOffsetsR.data(),
        rowIndsR.data(),
        valsR.data(),
        UGamma.data(),
        uStar.data()
    );

    cudaDeviceSynchronize();

    std::vector<Real> expectedP2 = {
        7.61797, 10.1498,
        2.955056, 3.08614,
        3.72659, 1.67041
    };

    std::vector<Real> expectedF = {
        17.23595, 26.29962
    };

    for (size_t i = 0; i < resultP.size(); ++i)
        EXPECT_NEAR(resultP[i], expectedP2[i], 1e-4);

    for (size_t i = 0; i < resultF.size(); ++i)
        EXPECT_NEAR(resultF[i], expectedF[i], 1e-4);
}

int main(int argc, char **argv) {
    std::cout << "--- DIAGNOSTIC: Test Binary Starting ---" << std::endl;
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}