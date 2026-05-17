#include <gtest/gtest.h>

#include "solvers/EigenDecomp/EigenDecomp2d.h"
#include "solvers/EigenDecomp/EigenDecompThomas.cuh"
#include "poisson/Laplacian.cuh"
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"
#include <cmath>
#include <random>

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

TEST(ImmersedEq, SolvesImmeresed_Generic) {

    using Real = double;
    using Int  = int;

    GridDim dim(25, 30, 20);
    Real3d delta(1, 0.5, 2);
    Handle hand;
    BoundaryConfig<Real> boundary(
        {false, false, false},
        {false, false, false},
        {0, 0, 0},
        {0, 0, 0},
        delta, dim, true
    );

    std::vector<Int> rowOffsetsB = {0, 1, 2};
    std::vector<Int> colIndsB    = {0, 1};
    std::vector<Real> valuesB    = {1, 1};
    auto B = SparseCSR<Real, Int>::create(valuesB.size(), rowOffsetsB.size() - 1, dim.size(), hand);
    B.set(rowOffsetsB.data(), colIndsB.data(), valuesB.data(), hand);
    auto BDense = Mat<Real>::create(B.rows, B.cols);
    B.getDense(BDense, hand);

    std::vector<Real> xHost(dim.size(), 0);
    for (size_t i = 0; i < xHost.size(); ++i) xHost[i] = i + 1.0;
    auto x = SimpleArray<Real>::create(dim.size(), hand);
    x.set(xHost.data(), hand);

    BandedMat<Real> L = poisson::laplacian(boundary, hand);
    auto LDense = SquareMat<Real>::create(dim.size());
    L.getDense(LDense, &hand);

    auto LPlus2BTBx = SimpleArray<Real>::create(dim.size(), hand);
    LPlus2BTBx.fill(0, hand);

    BDense.mult(BDense, &LDense, &hand,&GPUScalar<Real>::get(2), &GPUScalar<Real>::get(1),  true, false);
    LDense.mult(x, LPlus2BTBx, &hand, &GPUScalar<Real>::get(1), &GPUScalar<Real>::get(0), false);

    std::vector<Real> fHost(rowOffsetsB.size() - 1, 0);
    fHost[0] = 1;
    fHost[1] = 2;
    auto f = SimpleArray<Real>::create(fHost.size(), hand);
    f.set(fHost.data(), hand);

    auto TwoBTF = SimpleArray<Real>::create(dim.size(), hand);
    BDense.mult(f, TwoBTF, &hand, &GPUScalar<Real>::get(2), &GPUScalar<Real>::get(0), true);

    LPlus2BTBx.add(TwoBTF, &GPUScalar<Real>::get(-1), &hand);

    std::vector<Real> p(dim.size(), 0);
    LPlus2BTBx.get(p.data(), hand);

    std::vector<Real> resultP(dim.size(), 0);
    std::vector<Real> resultF(fHost.size(), 0);

    ImmersedEq<Real, Int> imEq(boundary, fHost.size(), valuesB.size(), p.data(), fHost.data(), delta, 1, 1e-12, 1000);


    imEq.solve(resultP.data(), valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data());

    cudaDeviceSynchronize();

    for (size_t i = 0; i < resultP.size(); ++i) EXPECT_NEAR(resultP[i], i + 1, 1e-4);

}


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
    std::vector<T> xCpu ={2, 1, 7, 2};
    X.set(xCpu.data(), hand);

    auto Y = SquareMat<T>::create(dim.rows);
    std::vector<T> yCpu ={3, 1, 5, 1, 3, 1, 0, 1, 3};
    Y.set(yCpu.data(), hand);

    auto Z = SquareMat<T>::create(dim.layers);
    std::vector<T> zCpu ={5, 2, 1, 5};
    Z.set(zCpu.data(), hand);


    KroneckerTriplet<T> kt(X, Y, Z);

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
        L.mult(vi, Lvi, &hand, &GPUScalar<T>::get(1), &GPUScalar<T>::get(0), false);


        Vec<T> lam_vi = SimpleArray<T>::create(L._rows, hand);
        lam_vi.set(vi, hand);
        lam_vi.mult(lambda[i], &hand);

        Lvi.add(lam_vi, &GPUScalar<T>::get(-1), &hand);

        Lvi.norm(normGpu, hand);

        err = normGpu.get(hand);

        EXPECT_LT(std::abs(err), tol)
            << errorMsg << "\nEigenpair failed at index " << i
            << " residual = " << err;
    }

    for (size_t i = 0; i < V._cols; ++i)
        for (size_t j = i + 1; j < V._cols; ++j) {
            V.col(i).mult(V.col(j), normGpu, &hand);
            T err = normGpu.get(hand);
            EXPECT_LT(std::abs(err), tol)
            << errorMsg << "\nEigenpair failed at index " << i
            << " residual = " << err;
        }


    auto LambdaVT = SquareMat<T>::create(V._cols);
    auto VLambdaVT = SquareMat<T>::create(V._cols);
    auto Lambda = SquareMat<T>::create(V._cols);
    Lambda.fill(0, hand);
    Lambda.diag(0).set(lambda, hand);
    Lambda.mult(V, &LambdaVT, &hand, false, true);
    V.mult(LambdaVT, &VLambdaVT, &hand, false, false);

    auto diff =  SquareMat<T>::create(V._cols);
    diff.set(L, hand);
    diff.add(VLambdaVT, diff, GPUScalar<T>::get(1), GPUScalar<T>::get(-1), false, false, hand);

    std::vector<T> diffHost(diff.size(), 0);
    diff.get(diffHost.data(), hand);
    for (size_t i = 0; i < diffHost.size(); ++i) {
        T val = diffHost[i];
        EXPECT_LT(std::abs(val), tol)
            << "failed at index " << i
            << " (row " << i % V._cols << ", col " << i / V._cols << ")"
            << " with diff: " << val
            << " (tol: " << tol << ")";
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

    // std::cout << "verifyEigenSolverIdentity banded laplacian = \n" << GpuOut<Real>(laplacian, hands[0]) << std::endl;
    // std::cout << "verifyEigenSolverIdentity dense laplacian = \n" << GpuOut<Real>(laplacian.getDense(hands[0]), hands[0]) << std::endl;

    auto x = SimpleArray<Real>::create(dim.size(), hands[0]);

    std::vector<Real> xCpuOrig(dim.size());

    // for (size_t i = 0; i < dim.size(); ++i) xCpuOrig[i] = 3;
    // GridInd3d ind(0, 0, 0);
    // for (; ind.layer <  dim.layers; ++ind.layer)
    //     for (; ind.row < dim.rows; ++ind.row)
    //         for (; ind.col < dim.cols; ++ind.col)
    //             xCpuOrig[dim[ind]] = std::cos(M_PI * ind.col / dim.cols) * std::cos(M_PI * ind.row / dim.rows) * std::cos(M_PI * ind.layer / dim.layers);
    std::mt19937 rng(0);
    std::uniform_real_distribution<Real> dist(-1, 1);
    for (size_t i = 0; i < dim.size(); ++i)
        xCpuOrig[i] = dist(rng);

    x.set(xCpuOrig.data(), hands[0]);


    auto rhs = SimpleArray<Real>::create(dim.size(), hands[0]);
    rhs.fill(0, hands[0]);

    laplacian.bandedMult(x, rhs, hands, GPUScalar<Real>::get(1), GPUScalar<Real>::get(0), false);

    // std::cout << "x = " << GpuOut<Real>(x, hands[0]) << std::endl;
    // std::cout << "rhs = " << GpuOut<Real>(rhs, hands[0]) << std::endl;

    x.fill(0, hands[0]);

    if (dim.numDims() == 2) {
        EigenDecomp2d<Real> ed(boundary, hands, events[0]);
        ed.solve(x, rhs, hands[0]);
    } else {
        EigenDecomp3d<Real> ed(boundary, hands, events);
        ed.solve(x, rhs, hands[0]);
    }

    std::vector<Real> xCpuResult(dim.size());
    x.get(xCpuResult.data(), hands[0]);

    cudaDeviceSynchronize();

    if (boundary.allNeumann()) {

        Real offset = xCpuOrig[0] - xCpuResult[0];

        for (size_t i = 0; i < dim.size(); ++i){
            // Shift the solver's zero-mean result back to the original mean
            Real shiftedResult = xCpuResult[i] + offset;

            EXPECT_NEAR(xCpuOrig[i], shiftedResult, 1e-7 * std::abs(xCpuOrig[i]) + 1e-10)
                << "Solver divergence at index " << i << " in " << dim.numDims()
                << "D (Singular Mode Shift: " << offset << ")";
        }
    } else {
        // Standard Dirichlet/Mixed validation
        for (size_t i = 0; i < dim.size(); ++i) {
            EXPECT_NEAR(xCpuOrig[i], xCpuResult[i], tolerance)
                << "Solver divergence at index " << i << " in " << dim.numDims() << "D";
        }
    }

    if (dim.numDims()== 3) {
        EigenDecompThomas<Real> ed(boundary, 1, hands, events);
        x.fill(0, hands[0]);
        ed.solve(x, rhs, hands[0]);

        if (boundary.allNeumann()) {
            Real offset = xCpuOrig[0] - xCpuResult[0];
            for (size_t i = 0; i < dim.size(); ++i){
                Real shiftedResult = xCpuResult[i] + offset;
                EXPECT_NEAR(xCpuOrig[i], shiftedResult, 1e-7 * std::abs(xCpuOrig[i]) + 1e-10)
                    << "Solver divergence at index " << i << " in " << dim.numDims()
                    << "D (Singular Mode Shift: " << offset << ")";
            }
        } else {
            for (size_t i = 0; i < dim.size(); ++i) {
                EXPECT_NEAR(xCpuOrig[i], xCpuResult[i], tolerance)
                    << "Solver divergence at index " << i << " in " << dim.numDims() << "D";
            }
        }
    }
}


template <typename Real, typename Int>
void verifyImmersedEqWithBoundary(const BoundaryConfig<Real>& boundary, Handle& hand, Real tolerance, const std::string& locMsg, bool isCout = false){
    GridDim dim = boundary.dim();
    size_t n = dim.size();

    // ============================================================
    // 1. Setup sparse boundary matrix B
    // ============================================================

    std::vector<Int> rowOffsetsB = {0, 1, 2};
    std::vector<Int> colIndsB    = {0, std::min((Int)1, (Int)(n - 1))};
    std::vector<Real> valuesB    = {1.0, 1.0};

    size_t numB = rowOffsetsB.size() - 1;

    auto B = SparseCSR<Real, Int>::create(valuesB.size(), numB, n, hand);

    B.set(rowOffsetsB.data(), colIndsB.data(), valuesB.data(), hand);

    auto denseB = Mat<Real>::create(numB, n);
    B.getDense(denseB, hand);

    if (isCout) std::cout << "B dense = \n" << GpuOut<Real>(denseB, hand) << std::endl;


    // ============================================================
    // 2. Ground truth solution x0
    // ============================================================

    std::vector<Real> x0Host(n, 0);

    for (size_t i = 0; i < n; ++i) x0Host[i] = static_cast<Real>(i + 1.0);

    auto x0 = SimpleArray<Real>::create(n, hand);
    x0.set(x0Host.data(), hand);

    if (isCout) std::cout << "x0 = " << GpuOut<Real>(x0, hand) << std::endl;


    // ============================================================
    // 3. Boundary force vector f
    // ============================================================

    std::vector<Real> fHost(numB, 0);

    for (size_t i = 0; i < numB; ++i) fHost[i] = static_cast<Real>((i + 1) * (i + 1));

    auto f = SimpleArray<Real>::create(numB, hand);
    f.set(fHost.data(), hand);

    if (isCout) std::cout << "f = " << GpuOut<Real>(f, hand) << std::endl;


    // ============================================================
    // 4. Construct explicit manufactured system
    //
    // p0 = Lx + 2 B^T B x - 2 B^T f - bc
    // ============================================================

    auto p0 = SimpleArray<Real>::create(n, hand);
    p0.fill(0, hand);

    if (isCout) std::cout << "p0 init = " << GpuOut<Real>(p0, hand) << std::endl;


    // ------------------------------------------------------------
    // Laplacian
    // ------------------------------------------------------------

    auto lhs = SquareMat<Real>::create(n);

    BandedMat<Real> L = poisson::laplacian(boundary, hand);

    L.bandedMult(x0, p0, &hand, GPUScalar<Real>::get(1), GPUScalar<Real>::get(0), false);

    L.getDense(lhs, &hand);

    if (isCout) std::cout << "L = \n" << GpuOut<Real>(lhs, hand) << std::endl;

    if (isCout) std::cout << "Lx = " << GpuOut<Real>(p0, hand) << std::endl;


    // ------------------------------------------------------------
    // Add 2 B^T B
    // ------------------------------------------------------------

    denseB.mult(denseB, &lhs, &hand, &GPUScalar<Real>::get(2), &GPUScalar<Real>::get(1), true, false);

    lhs.add(denseB, lhs, GPUScalar<Real>::get(1), GPUScalar<Real>::get(0), false, false, hand);


    // ------------------------------------------------------------
    // tempB = B x0
    // ------------------------------------------------------------

    auto tempB = SimpleArray<Real>::create(numB, hand);

    size_t wsSizeB = B.multWorkspaceSize(x0, tempB, GPUScalar<Real>::get(1), GPUScalar<Real>::get(0), false, hand);

    auto workspaceB = SimpleArray<Real>::create(wsSizeB, hand);

    B.mult(x0, tempB, GPUScalar<Real>::get(1), GPUScalar<Real>::get(0), false, workspaceB, hand);

    if (isCout) std::cout << "Bx = " << GpuOut<Real>(tempB, hand) << std::endl;


    // ------------------------------------------------------------
    // p0 += 2 B^T B x0
    // ------------------------------------------------------------

    size_t wsSizeBt = B.multWorkspaceSize( tempB, p0, GPUScalar<Real>::get(2), GPUScalar<Real>::get(1), true, hand);

    auto workspaceBt = SimpleArray<Real>::create(wsSizeBt, hand);

    B.mult(tempB, p0, GPUScalar<Real>::get(2), GPUScalar<Real>::get(1), true, workspaceBt, hand);

    if (isCout) std::cout << "2 B^T B x + Lx = " << GpuOut<Real>(p0, hand) << std::endl;


    // ------------------------------------------------------------
    // p0 -= 2 B^T f
    // ------------------------------------------------------------

    auto rhs = SimpleArray<Real>::create(n, hand);

    size_t wsSizeBtf = B.multWorkspaceSize(f, p0, GPUScalar<Real>::get(2), GPUScalar<Real>::get(0), true, hand);

    auto workspaceBtf = SimpleArray<Real>::create(wsSizeBtf, hand);

    B.mult(f, p0, GPUScalar<Real>::get(-2), GPUScalar<Real>::get(1), true, workspaceBtf, hand);

    B.mult(f, rhs, GPUScalar<Real>::get(2), GPUScalar<Real>::get(0), true, workspaceBtf, hand);

    if (isCout) std::cout << " 2 B^T f = " << GpuOut<Real>(rhs, hand) << std::endl;
    if (isCout) std::cout << "2 B^T B x + Lx - 2 B^T f = " << GpuOut<Real>(p0, hand) << std::endl;


    // ------------------------------------------------------------
    // Boundary correction
    // ------------------------------------------------------------

    auto bc = poisson::boundaryCorrection(boundary, hand);

    if (isCout) std::cout << "bc = " << GpuOut<Real>(bc, hand) << std::endl;

    p0.add(bc, &GPUScalar<Real>::get(-1), &hand);

    if (isCout) std::cout << "2 B^T B x + Lx - 2 B^T f - bc = " << GpuOut<Real>(p0, hand) << std::endl;


    // ------------------------------------------------------------
    // Construct explicit RHS
    // rhs = p0 + 2 B^T f + bc
    // ------------------------------------------------------------

    rhs.add(p0, &GPUScalar<Real>::get(1), &hand);
    rhs.add(bc, &GPUScalar<Real>::get(1), &hand);

    if (isCout) std::cout << "rhs test = " << GpuOut<Real>(rhs, hand) << std::endl;

    if (isCout) std::cout << "lhs test =\n" << GpuOut<Real>(lhs, hand) << std::endl;


    // ============================================================
    // 5. Detect singularity using LU
    // ============================================================

    auto lhsCopy = SquareMat<Real>::create(lhs._cols);
    lhsCopy.set(lhs, hand);

    double det =  lhsCopy.determinant(hand);

    if (isCout) std::cout << "LU packed matrix = \n" << GpuOut<Real>(lhsCopy, hand) << std::endl;


    bool isSingular = (std::abs(det) < 1e-5);

    if (isCout) std::cout << "LHS matrix is " << (isSingular ? "SINGULAR" : "INVERTIBLE") << " (det info = " << det << ")" << std::endl;



    // ============================================================
    // 6. Solve using immersed equation
    // ============================================================

    std::vector<Real> p0Host(n, 0);
    p0.get(p0Host.data(), hand);

    ImmersedEq<Real, Int> imEq(boundary, numB, valuesB.size(), p0Host.data(), fHost.data(), Real3d(1, 1, 1), 1, 1e-12, 1000);

    std::vector<Real> resultX(n, 0);

    imEq.solve(resultX.data(), valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data());


    // ============================================================
    // 7. Recover transformed immersed operators
    // ============================================================

    // ============================================================
    // 7. Recover transformed immersed operators
    // ============================================================

    // 1. Generate the LHS Matrix (Runs on hand5[0])
    auto invLLHS = imEq.LHSMat();

    // CRITICAL: Wait for hand5[0] to finish before copying on `hand`
    cudaDeviceSynchronize();

    // 2. DEEP COPY the LHS matrix so it survives the RHS generation
    auto invLLHSCopy = SquareMat<Real>::create(n);
    invLLHSCopy.set(invLLHS, hand);
    cudaDeviceSynchronize();

    // 3. Generate the RHS Vector (Runs on hand5[0])
    // This will safely overwrite the shared alias buffer.
    auto invLRHS = imEq.getRHS(p0, f, bc, B);

    // CRITICAL: Wait for hand5[0] to finish before L.bandedMult reads it
    cudaDeviceSynchronize();

    auto lhsImEq = SquareMat<Real>::create(n);
    auto rhsImEq = SimpleArray<Real>::create(n, hand);

    // 4. Multiply using the safe, isolated copies
    L.getDense(hand).mult(invLLHSCopy, &lhsImEq, &hand, &GPUScalar<Real>::get(1), &GPUScalar<Real>::get(0), false, false);
    L.bandedMult(invLRHS, rhsImEq, &hand, GPUScalar<Real>::get(1), GPUScalar<Real>::get(0), false);

    if (isCout) std::cout << "rhs ImEq = " << GpuOut<Real>(rhsImEq, hand) << std::endl;
    if (isCout) std::cout << "lhs ImEq = \n" << GpuOut<Real>(lhsImEq, hand) << std::endl;

    // ============================================================
    // 8. Structural equality checks
    // ============================================================

    std::vector<Real> lhsHost(n * n, 0);
    std::vector<Real> lhsImEqHost(n * n, 0);

    lhs.get(lhsHost.data(), hand);
    lhsImEq.get(lhsImEqHost.data(), hand);

    for (size_t i = 0; i < n * n; ++i)
        EXPECT_NEAR(lhsHost[i], lhsImEqHost[i], tolerance) << locMsg << " - LHS Matrix mismatch at flat index " << i;


    std::vector<Real> rhsHost(n, 0);
    std::vector<Real> rhsImEqHost(n, 0);

    rhs.get(rhsHost.data(), hand);
    rhsImEq.get(rhsImEqHost.data(), hand);

    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(rhsHost[i], rhsImEqHost[i], tolerance) << locMsg << " - RHS Vector mismatch at flat index " << i;


    // ============================================================
    // 9. Validate recovered solution
    // ============================================================

    if (isSingular) {

        // --------------------------------------------------------
        // Singular systems:
        //
        // We only require that the returned solution satisfies:
        //
        //     A x = b
        //
        // because the solution may not be unique.
        // --------------------------------------------------------

        auto resultDevice = SimpleArray<Real>::create(n, hand);

        resultDevice.set(resultX.data(), hand);

        auto residual = SimpleArray<Real>::create(n, hand);

        residual.fill(0, hand);

        // residual = A * resultX

        lhs.mult(resultDevice, residual, &hand, &GPUScalar<Real>::get(1), &GPUScalar<Real>::get(0), false);

        // residual -= rhs

        residual.add(rhs, &GPUScalar<Real>::get(-1), &hand);

        std::vector<Real> residualHost(n, 0);

        residual.get(residualHost.data(), hand);

        for (size_t i = 0; i < n; ++i)
            EXPECT_NEAR(residualHost[i], Real(0), tolerance) << locMsg << " - Singular-system residual mismatch at flat index " << i << " residual = " << residualHost[i];
    } else {

        // --------------------------------------------------------
        // Nonsingular systems:
        // require exact recovery of x0.
        // --------------------------------------------------------

        for (size_t i = 0; i < n; ++i)
            EXPECT_NEAR(resultX[i], x0Host[i], tolerance) << locMsg << " - Solution vector mismatch at flat index " << i;
    }
}

template <typename Real>
void boundaryBattery(XYZ<bool> startIsN, XYZ<bool> endIsN, XYZ<Real> startVal, XYZ<Real> endVal, GridDim dim, bool isStag, Handle* hand3, Event* event2, double tolerance) {

    std::stringstream ss;
    ss << "startIsNeuman = " << startIsN
       << " endIsNeumann = " << endIsN
       << " isStagered = " << isStag
       << " startVal = " << startVal
       << " endVal = " << endVal
       << " dim = " << dim;

    std::string locMsg = ss.str();

    // std::cout << locMsg << std::endl;

    BoundaryConfig<Real> boundary(
        startIsN,
        endIsN,
        startVal,
        endVal,
        Real3d(1, 1, 1),
        dim,
        isStag
    );

    poisson::Laplacian1d<Real> laplacian1d(boundary, hand3[0]);

    poisson::Eigen<Real> laplacianEigen = poisson::Eigen<Real>::make(boundary, hand3, event2);

    for (size_t i = 0; i < dim.numDims(); ++i)
        checkEigens(laplacian1d.dense(i, hand3[i]), laplacianEigen.vecs[i], laplacianEigen.vals[i],  hand3[i], locMsg);

    verifyEigenSolverIdentity(dim, boundary,  hand3, event2, tolerance);

    verifyImmersedEqWithBoundary<Real, int32_t>(boundary, hand3[0], tolerance, locMsg);
}

TEST(LaplacianMath, laplacian) {
    Handle hand3[3];
    Event event2[2];
    using Real = double;

    double tolerance = 1e-10;

    size_t maxDim = 3;
    size_t startRowsCols = 2;

    // boundaryBattery<Real>(
    //     {0,0,0},
    //     {0, 0, 0},
    //     {0, 0, 0},
    //     {0, 0, 0},
    //     {2, 2, 1},
    //     1,
    //     hand3, event2, tolerance
    // );



    for (size_t x0IsN = 0; x0IsN < 2; ++x0IsN)
        for (size_t x1IsN = 0; x1IsN < 2; ++x1IsN)
            for (size_t y0IsN = 0; y0IsN < 2; ++y0IsN)
                for (size_t y1IsN = 0; y1IsN < 2; ++y1IsN)
                    for (size_t z0IsN = 0; z0IsN < 2; ++z0IsN)
                        for (size_t z1IsN = 0; z1IsN < 2; ++z1IsN)
                            for (size_t isStag = 0; isStag < 2; ++isStag)
                                for (size_t x0Val = 0; x0Val < 2; ++x0Val)
                                    for (size_t x1Val = 0; x1Val < 2; ++x1Val)
                                        for (size_t y0Val = 0; y0Val < 2; ++y0Val)
                                            for (size_t y1Val = 0; y1Val < 2; ++y1Val)
                                                for (size_t z0Val = 0; z0Val < 2; ++z0Val)
                                                    for (size_t z1Val = 0; z1Val < 2; ++z1Val)
                                                        for (size_t rows = startRowsCols; rows < maxDim; ++rows)
                                                            for (size_t cols = startRowsCols; cols < maxDim; ++cols)
                                                                for (size_t layers = 1; layers < maxDim; ++layers)
                                                                    boundaryBattery<Real>(
                                                                        XYZ<bool>(x0IsN, y0IsN, z0IsN),
                                                                        XYZ<bool>(x1IsN, y1IsN, z1IsN),
                                                                        XYZ<Real>(static_cast<Real>(x0Val), static_cast<Real>(y0Val), static_cast<Real>(z0Val)),
                                                                        XYZ<Real>(static_cast<Real>(x1Val), static_cast<Real>(y1Val), static_cast<Real>(z1Val)),
                                                                        GridDim(rows, cols, layers),
                                                                        isStag,
                                                                        hand3, event2, tolerance
                                                                    );
}


int main(int argc, char **argv) {
    std::cout << "--- DIAGNOSTIC: Test Binary Starting ---" << std::endl;
    testing::InitGoogleTest(&argc, argv);

    auto mat = Mat<double>::create(2, 2);

    return RUN_ALL_TESTS();
}