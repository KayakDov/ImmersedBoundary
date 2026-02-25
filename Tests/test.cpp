#include <gtest/gtest.h>

#include "ToeplitzLaplacian.cuh"
#include "../src/solvers/EigenDecomp/EigenDecomp3d.cuh"

#include "immersedBoundary/ImerssedEquation.h"

//TODO: test against NaN values that should be overwritten by 0 multiplicaiton!

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
        1e-8,
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
        EXPECT_NEAR(resultP[i], expectedP1[i], 1e-4);

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

TEST(EigenDecomp, ThreeD) {

    using Real = double;
    using Int  = int;
    GridDim dim(3, 2, 2);
    Real3d delta(1, 1, 1);

    Handle hand3[3];

    auto x = SimpleArray<Real>::create(12, hand3[0]);
    std::vector<Real> xHost = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    x.set(xHost.data(), hand3[0]);

    auto L = ToeplitzLaplacian<Real>::L(dim, hand3[0], delta);

    auto b = SimpleArray<Real>::create(12, hand3[0]);
    b.fill(0, hand3[0]);

    L.bandedMult(x, b, &hand3[0]);

    x.fill(0, hand3[0]);

    x.get(xHost.data(), hand3[0]);
    for (size_t i = 0; i < xHost.size(); ++i) EXPECT_NEAR(xHost[i], 0, 1e-10);

    Event event3[3];

    EigenDecomp3d<Real> eds(dim, hand3, delta, event3);

    eds.solve(x, b, hand3[0]);

    x.get(xHost.data(), hand3[0]);
    for (size_t i = 0; i < xHost.size(); ++i) EXPECT_NEAR(xHost[i], i + 1, 1e-10);
}

TEST(BCGDenseTest, ConvergenceValidation) {
    using Real = double;

    Handle hand4[4]{};
    Event events11[11];
    size_t n = 6;
    Real tolerance = 1.0e-6;
    size_t maxIterations = 100;

    auto A = SquareMat<Real>::create(n);
    std::vector<Real> hostA = {
         0.410352, -0.186335, -0.0563147, -0.172257, -0.0993789, -0.0389234,
        -0.186335,  0.354037, -0.186335, -0.0993789, -0.21118,  -0.0993789,
         0.0,       0.0,       1.0,        0.0,       0.0,       0.0,
         0.0,       0.0,       0.0,        1.0,       0.0,       0.0,
         0.0,       0.0,       0.0,        0.0,       1.0,       0.0,
         0.0,       0.0,       0.0,        0.0,       0.0,       1.0
    };
    A.set(hostA.data(), hand4[0]);

    auto b = SimpleArray<Real>::create(n, hand4[0]);
    std::vector<Real> hostB = {-1.51304, -1.56522, -0.313043, -0.486957, -0.434783, 0.313043};
    b.set(hostB.data(), hand4[0]);

    auto result = SimpleArray<Real>::create(n, hand4[0]);
    auto bHeightX7 = Mat<Real>::create(n, 7);
    auto aX9 = SimpleArray<Real>::create(9, hand4[0]);

    BCGDense<Real>::solve(hand4, A, result, b, events11, &bHeightX7, &aX9, tolerance, maxIterations);

    std::vector<Real> actual(n);
    result.get(actual.data(), hand4[0]);

    std::vector<Real> expected = {-7.48312639568, -8.35954534961, -2.29212890075, -2.60674032488, -2.94381665669, -0.80898814329};

    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(actual[i], expected[i], 1e-5) << "Mismatch at solution vector index " << i;
}

int main(int argc, char **argv) {
    std::cout << "--- DIAGNOSTIC: Test Binary Starting ---" << std::endl;
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}