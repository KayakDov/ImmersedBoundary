#include <gtest/gtest.h>

#include "solvers/EigenDecomp/EigenDecomp2d.h"
#include "solvers/EigenDecomp/EigenDecompThomas.cuh"
#include "poisson/Laplacian.cuh"
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"
#include <cmath>
#include "immersedBoundary/ImerssedEquation.h"
#include "kronecker/KroneckerTriplet.h"

// TEST(ImmersedEq, SolvesPrimes_3x2x1) {
//
//     using Real = double;
//     using Int  = int32_t;
//
//     GridDim dim(3, 2, 1);
//     Real3d delta(1, 1, 1);
//     Handle hand;
//
//     BoundaryConfig<Real> boundary(
//         {true, true, true},
//         {true, true, true},
//         {0, 0, 0},
//         {0, 0, 0},
//         delta, dim, true
//     );
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
//     ImmersedEq<Real, Int> imEq(boundary, f.size(), valsR.size(), p.data(), f.data(), delta, deltaT, 1e-8, 1000);
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
//     BoundaryConfig<Real> boundary(
//         {true, true, true},
//         {true, true, true},
//         {0, 0, 0},
//         {0, 0, 0},
//         delta, dim, true
//     );
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
//     BandedMat<Real> L = poisson::laplacian(boundary, hand);
//     auto LDense = SquareMat<Real>::create(dim.size());
//     L.getDense(LDense, &hand);
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
//     ImmersedEq<Real, Int> imEq(boundary, fHost.size(), valuesB.size(), p.data(), fHost.data(), delta, 1, 1e-8, 1000);
//
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
//     BoundaryConfig<Real> boundary(
//         {true, true, true},
//         {true, true, true},
//         {0, 0, 0},
//         {0, 0, 0},
//         delta, dim, true
//     );
//
//     auto x = SimpleArray<Real>::create(dim.size(), hand3[0]);
//     std::vector<Real> xHost = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
//     x.set(xHost.data(), hand3[0]);
//
//     BandedMat<Real> L = poisson::laplacian(boundary, hand3[0]);
//
//     auto b = SimpleArray<Real>::create(12, hand3[0]);
//     b.fill(0, hand3[0]);
//
//     L.bandedMult(x, b, &hand3[0]);
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
//     EigenDecomp3d<Real> eds(boundary, hand3, event3);
//
//     eds.solve(x, b, hand3[0]);
//
//     x.get(xHost.data(), hand3[0]);
//
//     cudaDeviceSynchronize();
//     for (size_t i = 0; i < xHost.size(); ++i) EXPECT_NEAR(xHost[i], i + 1, 1e-10);
//
//     L.bandedMult(x, b, &hand3[0]);
//     x.fill(0, hand3[0]);
//     x.get(xHost.data(), hand3[0]);
//
//     cudaDeviceSynchronize();
//     for (size_t i = 0; i < xHost.size(); ++i) EXPECT_NEAR(xHost[i], 0, 1e-10);
//
//     EigenDecompThomas<Real> edt(boundary, delta.x, hand3, event3);
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
//     BoundaryConfig<Real> boundary(
//         {true, true, true},
//         {true, true, true},
//         {0, 0, 0},
//         {0, 0, 0},
//         delta, dim, true
//     );
//
//     Handle hand2[2];
//
//     //ToeplitzLaplacian<Real>::printL(dim, hand3[0], delta);
//
//     auto x = SimpleArray<Real>::create(dim.size(), hand2[0]);
//     std::vector<Real> xHost = {1, 2, 3, 4, 5, 6};
//     x.set(xHost.data(), hand2[0]);
//
//     auto L = poisson::laplacian(boundary, hand2[0]);
//
//     auto b = SimpleArray<Real>::create(dim.size(), hand2[0]);
//     b.fill(0, hand2[0]);
//
//     L.bandedMult(x, b, &hand2[0]);
//
//     x.fill(0, hand2[0]);
//
//     x.get(xHost.data(), hand2[0]);
//     for (size_t i = 0; i < xHost.size(); ++i) EXPECT_NEAR(xHost[i], 0, 1e-10);
//
//     Event event;
//
//     EigenDecomp2d<Real> eds(boundary, hand2,event);
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


template<typename T>
void expectMatrixNear(const Mat<T>& A, const Mat<T>& B, Handle& hand, T tol = 1e-6) {
    ASSERT_EQ(A._rows, B._rows);
    ASSERT_EQ(A._cols, B._cols);

    std::vector<T> aCpu(A.size(), 0), bCpu(B.size(), 0);
    A.get(aCpu.data(), hand);
    B.get(bCpu.data(), hand);

    for (size_t i = 0; i < aCpu.size(); i++)
        ASSERT_NEAR(aCpu[i], bCpu[i], tol) << "Mismatch at (" << i % A._rows << "," << i / A._rows << ")";
}

// ---------- test ----------

TEST(KroneckerTripletTest, ProductMatchesMultOnIdentity) {
    using T = double;

    Handle hand;

    // Nontrivial sizes
    GridDim dim(3, 2, 2);

    auto X = SquareMat<T>::create(dim.cols);
    std::vector<T> xCpu ={2, 1, 1, 2};
    X.set(xCpu.data(), hand);

    auto Y = SquareMat<T>::create(dim.rows);
    std::vector<T> yCpu ={3, 1, 0, 1, 3, 1, 0, 1, 3};
    Y.set(yCpu.data(), hand);

    auto Z = SquareMat<T>::create(dim.layers);
    std::vector<T> zCpu ={5, 1, 1, 5};
    Z.set(zCpu.data(), hand);

    XYZ<Mat<T>> mats{X, Y, Z};
    KroneckerTriplet<T> kt(mats);

    Mat<T> implicitResult = kt.product(hand);

    auto I = SquareMat<T>::create(dim.size()).setToIdentity(hand);

    Mat<T> explicitResult = Mat<T>::create(dim.size(), dim.size());
    kt.mult(I, explicitResult, false, hand);

    expectMatrixNear(explicitResult, implicitResult, hand);
}

/**
 * Examines eigen vectors and values, confiriming all vectors have norm 1, they are all orthonormal to one another,
 * and that L V = V Lambda.
 * @tparam T
 * @param L The 1d laplacian.
 * @param V The eigenvectors for the 1d laplacian.  Each column is a vector.
 * @param lambda The eigen values.
 * @param hand The context.
 * @param errorMsg Anthing that should be appended to an error message.
 * @param tol The tolerance.
 */
template<typename T>
static void checkEigens(const SquareMat<T>& L, const SquareMat<T>& V, const Vec<T>& lambda, Handle& hand, std::string errorMsg, T tol = 1e-6){


    auto normGpu= Singleton<T>::create(hand);

    for (size_t i = 0; i < lambda.size(); ++i) {
        Vec<T> vi = V.col(i);

        vi.norm(normGpu, hand);
        T err = normGpu.get(hand) - 1;
        EXPECT_LT(err, tol)
            << errorMsg << "\nEigen Vector is not orthogonal, col " << i << " has a norm not equal to 1 "
            << " residual = " << err;

        Vec Lvi = SimpleArray<T>::create(L._rows, hand);
        L.mult(vi, Lvi, &hand, &GPUConst<T>::get(1), &GPUConst<T>::get(0), false);


        Vec<T> lam_vi = SimpleArray<T>::create(L._rows, hand);
        lam_vi.set(vi, hand);
        lam_vi.mult(lambda[i], &hand);

        Lvi.add(lam_vi, &GPUConst<T>::get(-1), &hand);

        Lvi.norm(normGpu, hand);

        err = normGpu.get(hand);

        EXPECT_LT(err, tol)
            << errorMsg << "\nEigenpair failed at index " << i
            << " residual = " << err;
    }

    for (size_t i = 0; i < V._cols; ++i)
        for (size_t j = i + 1; j < V._cols; ++j) {
            T err = normGpu.get(hand);
            V.col(i).mult(V.col(j), normGpu, &hand);
            EXPECT_LT(err, tol)
            << errorMsg << "\nEigenpair failed at index " << i
            << " residual = " << err;
        }


}

/**
 * Verifies the numerical identity of the EigenDecomposition solver.
 * Performs a round-trip operation: rhs = L * x_orig, then x_final = L^-1 * rhs.
 *
 * @tparam Real Floating point type (float or double).
 * @param dim The dimensions of the grid.
 * @param boundary Configuration for boundary conditions.
 * @param laplacian The banded Laplacian matrix to multiply by.
 * @param hands Array of handles for stream management.
 * @param events Array of events for synchronization.
 * @param tolerance Maximum allowable difference for numerical validation.
 */
template<typename Real>
void verifyEigenSolverIdentity(
    const GridDim& dim,
    BoundaryConfig<Real>& boundary,
    Handle* hands,
    Event* events,
    Real tolerance) {

    auto laplacian = poisson::laplacian(boundary, hands[0]);

    auto x = SimpleArray<Real>::create(dim.size(), hands[0]);
    std::vector<Real> xCpu(dim.size());

    // Initialize with a non-trivial cubic signal
    for (size_t i = 0; i < dim.size(); ++i) {
        xCpu[i] = static_cast<Real>(i * i * i);
    }
    x.set(xCpu.data(), hands[0]);

    auto rhs = SimpleArray<Real>::create(dim.size(), hands[0]);
    rhs.fill(0, hands[0]);

    // Forward operation: rhs = L * x
    laplacian.bandedMult(x, rhs, hands, GPUConst<Real>::get(1), GPUConst<Real>::get(0), false);

    // Clear x to ensure the solver is responsible for the final values
    x.fill(0, hands[0]);

    if (dim.numDims() == 2) {
        EigenDecomp2d<Real> ed(boundary, hands, events[0]);
        ed.solve(x, rhs, hands[0]);
    } else {
        EigenDecomp3d<Real> ed(boundary, hands, events);
        ed.solve(x, rhs, hands[0]);
    }

    // Retrieve results to host
    x.get(xCpu.data(), hands[0]);
    cudaDeviceSynchronize();

    // Validate identity: x_final ≈ x_orig
    for (size_t i = 0; i < dim.size(); ++i) {
        Real expected = static_cast<Real>(i * i * i);
        EXPECT_NEAR(xCpu[i], expected, tolerance)
            << "Solver divergence at index " << i << " in " << dim.numDims() << "D";
    }
}

TEST(LaplacianMath, laplacian) {

    using Real = double;

    Handle hand3[3];
    Event event2[2];
    double tolerance = 1e-12;

    size_t j, k, l, m, n, o; j = k = l = n = 0; o = m = 1;

    // for (size_t j = 0; j < 2; ++j) {
    //     for (size_t k = 0; k < 2; ++k) {
    //         for (size_t l = 0; l < 2; ++l) {
    //             for (size_t m = 1; m < 2; ++m) {
    //                 for (size_t n = 0; n < 2; ++n) {
    //                     for (size_t o = 1; o < 2; ++o) {
                            GridDim dim(2 + m, 2 + n, 1 + o);
                            std::stringstream ss;
                            ss << "startIsNeuman = " << static_cast<bool>(j)
                               << " endIsNeumann = " << static_cast<bool>(k)
                               << " isStagered = " << static_cast<bool>(l)
                               << " dim = " << dim;

                            std::string locMsg = ss.str();

                            std::cout << locMsg << std::endl;

                            BoundaryConfig<Real> boundary(j, k, l,  dim);

                            verifyEigenSolverIdentity(dim, boundary,  hand3, event2, tolerance);


                             // poisson::Laplacian1d<Real> laplacian1d(boundary, hand3[0]);
                             //
                             // // std::cout << "L 1d matrices:\nx:\n" << GpuOut<Real>(laplacian1d.dense(0, hand3[0]), hand3[0])
                             // //                                 << "y\n" << GpuOut<Real>(laplacian1d.dense(1, hand3[0]), hand3[0])
                             // //                                 << "z\n" << GpuOut<Real>(laplacian1d.dense(2, hand3[0]), hand3[0])
                             // //                                 << std::endl;
                             //
                             // poisson::Eigen<Real> laplacianEigen = poisson::Eigen<Real>::make(boundary, hand3, event2);
                             // // std::cout << "Eigenvectors:\n" << GpuX3Out<SquareMat<Real>, Real>(laplacianEigen.vecs, hand3[0]) << std::endl;
                             // // std::cout << "Eigenvalues:\n" << GpuX3Out<Vec<Real>, Real>(laplacianEigen.vals, hand3[0]) << std::endl;
                             //
                             //
                             // for (size_t i = 0; i < dim.numDims(); ++i)
                             //     checkEigens(laplacian1d.dense(i, hand3[i]), laplacianEigen.vecs[i], laplacianEigen.vals[i],  hand3[i], locMsg);
                         // }
     //                 }
     //             }
     //         }
     //     }
     // }
}

int main(int argc, char **argv) {
    std::cout << "--- DIAGNOSTIC: Test Binary Starting ---" << std::endl;
    testing::InitGoogleTest(&argc, argv);

    auto mat = Mat<double>::create(2, 2);

    return RUN_ALL_TESTS();
}