#include <gtest/gtest.h>

#include "solvers/EigenDecomp/EigenDecomp2d.h"
#include "solvers/EigenDecomp/EigenDecompThomas.cuh"
#include "poisson/Laplacian.cuh"
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"

#include "immersedBoundary/ImerssedEquation.h"
//
// TEST(ImmersedEq, SolvesPrimes_3x2x1) {
//
//     using Real = double;
//     using Int  = int;
//
//     GridDim dim(3, 2, 1);
//     Real3d delta(1, 1, 1);
//     Handle hand;
//
//     size_t uStarSize =
//         dim.numDims() * dim.size()
//         + dim.cols * dim.layers
//         + dim.rows * dim.layers
//         + dim.cols * dim.rows * (dim.layers > 1);
//
//     std::vector<Real> uStar(uStarSize, 0);
//     uStar[3] = 1;
//     uStar[9] = 1;
//
//     std::vector<Int> colOffsetsR(3);
//     colOffsetsR[0] = 0;
//     colOffsetsR[1] = uStarSize;
//     colOffsetsR[2] = uStarSize;
//
//     std::vector<Int> rowIndsR(uStarSize);
//     for (size_t i = 0; i < uStarSize; ++i) rowIndsR[i] = static_cast<Int>(i);
//
//     std::vector<Real> valsR(uStarSize, 1);
//
//     std::vector<Real> UGamma(2, 3);
//
//     double deltaT = 3.0 / 2.0;
//
//     std::vector<Int> rowOffsetsB = {0, 1, 2};
//     std::vector<Int> colIndsB    = {0, 1};
//     std::vector<Real> valuesB    = {1, 1};
//
//     std::vector<Real> f = {1, 2};
//
//     std::vector<Real> p(dim.size(), 0);
//     p[0] = 2;
//     p[dim.size() - 1] = -2;
//
//     std::vector<Real> resultP(dim.size(), 0);
//     std::vector<Real> resultF(f.size(), 0);
//
//     // ToeplitzLaplacian<Real>::printL(dim, hand, delta);
//
//     ImmersedEq<Real, Int> imEq(dim, f.size(), valsR.size(), p.data(), f.data(), delta, deltaT, 1e-8, 1000);
//
//     imEq.solve(resultP.data(), valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data());
//
//     cudaDeviceSynchronize();
//
//     std::vector<Real> expectedP1 = {-7.483126, -8.359545, -2.292128, -2.606740, -2.943816, -0.808988};
//
//     for (size_t i = 0; i < resultP.size(); ++i) EXPECT_NEAR(resultP[i], expectedP1[i], 1e-4);
//
//     imEq.solve(
//         resultP.data(),
//         resultF.data(),
//         valuesB.size(),
//         rowOffsetsB.data(),
//         colIndsB.data(),
//         valuesB.data(),
//         valsR.size(),
//         colOffsetsR.data(),
//         rowIndsR.data(),
//         valsR.data(),
//         UGamma.data(),
//         uStar.data()
//     );
//
//     cudaDeviceSynchronize();
//
//     std::vector<Real> expectedP2 = {7.61797, 10.1498, 2.955056, 3.08614, 3.72659, 1.67041};
//
//     std::vector<Real> expectedF = {17.23595, 26.29962};
//
//     for (size_t i = 0; i < resultP.size(); ++i) EXPECT_NEAR(resultP[i], expectedP2[i], 1e-4);
//
//     for (size_t i = 0; i < resultF.size(); ++i) EXPECT_NEAR(resultF[i], expectedF[i], 1e-4);
// }
// TEST(ImmersedEq, SolvesImmeresed_Generic) {
//
//     using Real = double;
//     using Int  = int;
//
//     GridDim dim(25, 30, 20);
//     Real3d delta(1, 0.5, 2);
//     Handle hand;
//
//     std::vector<Int> rowOffsetsB = {0, 1, 2};
//     std::vector<Int> colIndsB    = {0, 1};
//     std::vector<Real> valuesB    = {1, 1};
//     auto B = SparseCSR<Real, Int>::create(valuesB.size(), rowOffsetsB.size() - 1, dim.size(), hand);
//     B.set(rowOffsetsB.data(), colIndsB.data(), valuesB.data(), hand);
//     auto BDense = Mat<Real>::create(B.rows, B.cols);
//     B.getDense(BDense, hand);
//
//     std::vector<Real> xHost(dim.size(), 0);
//     for (size_t i = 0; i < xHost.size(); ++i) xHost[i] = i + 1.0;
//     auto x = SimpleArray<Real>::create(dim.size(), hand);
//     x.set(xHost.data(), hand);
//
//     Laplacian<Real> L(dim, delta, BoundaryConfig<Real>(ConditionType::DirichletStaggered, 0, delta, dim));
//     L.setOperation(hand);
//
//     auto LDense = SquareMat<Real>::create(dim.size());
//     L.banded().getDense(LDense, &hand);
//
//     auto LPlus2BTBx = SimpleArray<Real>::create(dim.size(), hand);
//     LPlus2BTBx.fill(0, hand);
//
//     BDense.mult(BDense, &LDense, &hand,&GPUConst<Real>::get(2), &GPUConst<Real>::get(1),  true, false);
//     LDense.mult(x, LPlus2BTBx, &hand, &GPUConst<Real>::get(1), &GPUConst<Real>::get(0), false);
//
//     std::vector<Real> fHost(rowOffsetsB.size() - 1, 0);
//     fHost[0] = 1;
//     fHost[1] = 2;
//     auto f = SimpleArray<Real>::create(fHost.size(), hand);
//     f.set(fHost.data(), hand);
//
//     auto TwoBTF = SimpleArray<Real>::create(dim.size(), hand);
//     BDense.mult(f, TwoBTF, &hand, &GPUConst<Real>::get(2), &GPUConst<Real>::get(0), true);
//
//     LPlus2BTBx.add(TwoBTF, &GPUConst<Real>::get(-1), &hand);
//
//     std::vector<Real> p(dim.size(), 0);
//     LPlus2BTBx.get(p.data(), hand);
//
//     std::vector<Real> resultP(dim.size(), 0);
//     std::vector<Real> resultF(fHost.size(), 0);
//
//     ImmersedEq<Real, Int> imEq(dim, fHost.size(), valuesB.size(), p.data(), fHost.data(), delta, 1, 1e-8, 1000);
//
//     imEq.solve(resultP.data(), valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data());
//
//     cudaDeviceSynchronize();
//
//     for (size_t i = 0; i < resultP.size(); ++i) EXPECT_NEAR(resultP[i], i + 1, 1e-4);
//
// }
//
// TEST(EigenDecomp, ThreeD) {
//
//     using Real = double;
//     using Int  = int;
//     GridDim dim(3, 2, 2);
//     Real3d delta(1, 1, 1);
//
//     Handle hand3[3];
//
//     //ToeplitzLaplacian<Real>::printL(dim, hand3[0], delta);
//
//     auto x = SimpleArray<Real>::create(dim.size(), hand3[0]);
//     std::vector<Real> xHost = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
//     x.set(xHost.data(), hand3[0]);
//
//     Laplacian<Real> L(dim, delta,  BoundaryConfig<Real>(ConditionType::DirichletStaggered, 0, delta, dim));
//
//     auto b = SimpleArray<Real>::create(12, hand3[0]);
//     b.fill(0, hand3[0]);
//
//     L.banded().bandedMult(x, b, &hand3[0]);
//
//     x.fill(0, hand3[0]);
//
//     x.get(xHost.data(), hand3[0]);
//
//     cudaDeviceSynchronize();
//     for (size_t i = 0; i < xHost.size(); ++i) EXPECT_NEAR(xHost[i], 0, 1e-10);
//
//     Event event3[3];
//
//     EigenDecomp3d<Real> eds(dim, hand3, delta, event3);
//
//     eds.solve(x, b, hand3[0]);
//
//     x.get(xHost.data(), hand3[0]);
//
//     cudaDeviceSynchronize();
//     for (size_t i = 0; i < xHost.size(); ++i) EXPECT_NEAR(xHost[i], i + 1, 1e-10);
//
//     L.banded().bandedMult(x, b, &hand3[0]);
//     x.fill(0, hand3[0]);
//     x.get(xHost.data(), hand3[0]);
//
//     cudaDeviceSynchronize();
//     for (size_t i = 0; i < xHost.size(); ++i) EXPECT_NEAR(xHost[i], 0, 1e-10);
//
//     EigenDecompThomas<Real> edt(dim, hand3, delta, event3);
//
//     edt.solve(x, b, hand3[0]);
//
//     x.get(xHost.data(), hand3[0]);
//
//     cudaDeviceSynchronize();
//     for (size_t i = 0; i < xHost.size(); ++i)
//         EXPECT_NEAR(xHost[i], i + 1, 1e-10);
// }
//
//
// TEST(EigenDecomp, TwoD) {
//
//     using Real = double;
//     using Int  = int;
//     GridDim dim(3, 2, 1);
//     Real2d delta(1, 1);
//
//     Handle hand2[2];
//
//     //ToeplitzLaplacian<Real>::printL(dim, hand3[0], delta);
//
//     auto x = SimpleArray<Real>::create(dim.size(), hand2[0]);
//     std::vector<Real> xHost = {1, 2, 3, 4, 5, 6};
//     x.set(xHost.data(), hand2[0]);
//
//     Laplacian<Real> L(dim, delta, BoundaryConfig<Real>(ConditionType::DirichletStaggered, 0, delta, dim));
//
//     auto b = SimpleArray<Real>::create(dim.size(), hand2[0]);
//     b.fill(0, hand2[0]);
//
//     L.banded().bandedMult(x, b, &hand2[0]);
//
//     x.fill(0, hand2[0]);
//
//     x.get(xHost.data(), hand2[0]);
//     for (size_t i = 0; i < xHost.size(); ++i) EXPECT_NEAR(xHost[i], 0, 1e-10);
//
//     Event event;
//
//     EigenDecomp2d<Real> eds(dim, hand2, delta, event);
//
//     eds.solve(x, b, hand2[0]);
//
//     x.get(xHost.data(), hand2[0]);
//     for (size_t i = 0; i < xHost.size(); ++i) EXPECT_NEAR(xHost[i], i + 1, 1e-10);
// }
//
//
// TEST(BCGDenseTest, ConvergenceValidation) {
//     using Real = double;
//
//     Handle hand4[4]{};
//     Event events12[12];
//     size_t n = 6;
//     Real tolerance = 1e-6;
//     size_t maxIterations = 100;
//
//     auto A = SquareMat<Real>::create(n);
//     std::vector<Real> hostA = {
//          1, 2, 3, 4, 5, 6,
//          6, 5, 4, 3, 2, 1,
//          2, 4, 2, 6, 2, 7,
//          0, 1, 1, 0, 2, 3,
//          4, 5, 6, 7, 8, -3,
//          -2, -1, 5, -2, -4, 6
//     };
//     A.set(hostA.data(), hand4[0]);
//
//     auto result = SimpleArray<Real>::create(n, hand4[0]);
//     std::vector<Real> resultHost = {1, -1, 2, -2, 3, -3};
//     result.set(resultHost.data(), hand4[0]);
//
//     auto b = SimpleArray<Real>::create(n, hand4[0]);
//     b.fill(0, hand4[0]);
//     A.mult(result, b, hand4, &GPUConst<Real>::get(1), &GPUConst<Real>::get(0), false);
//     result.fill(0, hand4[0]);
//
//     auto bHeightX7 = Mat<Real>::create(n, 7);
//     auto aX9 = SimpleArray<Real>::create(9, hand4[0]);
//
//     BCGDense<Real>::solve(hand4, A, result, b, events12, &bHeightX7, &aX9, tolerance, maxIterations);
//
//     std::vector<Real> actual(n, 0);
//     result.get(actual.data(), hand4[0]);
//
//     cudaDeviceSynchronize();
//     for (size_t i = 0; i < n; ++i)
//         EXPECT_NEAR(actual[i], resultHost[i], 1e-5) << "Mismatch at solution vector index " << i;
// }

TEST(LaplacianMath, laplacian) {
    using Real = double;
    GridDim dim(2, 3, 2);
    Handle hand3[3];
    Event event2[2];
    Real3d delta(1, 1, 1);
    ConditionType type = ConditionType::DirichletNodeCentered;

    BoundaryConfig<Real> boundary(type, 0, 1, dim);
    Laplacian<Real> laplacian(dim, delta, boundary);
    SquareMat<Real> dense = laplacian.dense(hand3[0]);
    std::cout << "The Laplacian for " << dim << " is \n"<< GpuOut<Real>(dense, hand3[0]) << std::endl;

    Laplacian1d<Real> laplacian1d(boundary, hand3[0]);

    std::cout << "L 1d matrices:\nx:\n" << GpuOut<Real>(laplacian1d.dense(0, hand3[0]), hand3[0])
                                    << "y\n" << GpuOut<Real>(laplacian1d.dense(1, hand3[0]), hand3[0])
                                    << "z\n" << GpuOut<Real>(laplacian1d.dense(2, hand3[0]), hand3[0])
                                    << std::endl;
    
    LaplacianEigen<Real> laplacianEigen = LaplacianEigen<Real>::make(boundary, hand3, event2);
    std::cout << "Eigenvectors:\n" << GpuX3Out<SquareMat<Real>, Real>(laplacianEigen.vecs, hand3[0]) << std::endl;
    std::cout << "Eigenvalues:\n" << GpuX3Out<Vec<Real>, Real>(laplacianEigen.vals, hand3[0]) << std::endl;


}

int main(int argc, char **argv) {
    std::cout << "--- DIAGNOSTIC: Test Binary Starting ---" << std::endl;
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}