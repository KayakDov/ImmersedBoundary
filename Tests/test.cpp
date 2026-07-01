#include <gtest/gtest.h>

#include "solvers/EigenDecomp/EigenDecomp2d.h"
#include "solvers/EigenDecomp/EigenDecompThomas.cuh"
#include "poisson/Poisson.cuh"
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"
#include <cmath>
#include <random>


#include "FortranBindings.hpp"
#include "deviceArrays/headers/DeviceMemory.h"
#include "immersedBoundary/ImerssedEquation.h"
#include "kronecker/KroneckerTriplet.h"
#include "poisson/Laplacian1d.cuh"

template <typename Real, typename Int, typename BoundaryConfigT>
SparseCSR<Real, Int> basics(const BoundaryConfigT& boundary, size_t n, std::vector<Real>& x0Host, SquareMat<Real>& lhsOperator, std::vector<Int>& rowOffsetsB, std::vector<Real>& valuesB, std::vector<Int>& colIndsB, Handle& hand) {

    std::mt19937 rng(42); // Deterministic seed for reproducible testing
    std::uniform_real_distribution<Real> dist(-5.0, 5.0);
    for (size_t i = 0; i < n; ++i) x0Host[i] = dist(rng);

    auto B = SparseCSR<Real, Int>::create(valuesB.size(), rowOffsetsB.size() - 1, n, hand);
    B.set(rowOffsetsB.data(), colIndsB.data(), valuesB.data(), hand);

    poisson::laplacian<Real>(boundary, hand).getDense(lhsOperator, hand);
    auto denseB = Mat<Real>::create(rowOffsetsB.size() - 1, n);
    B.getDense(denseB, hand);
    denseB.mult(denseB, &lhsOperator, &hand, &GPUScalar<Real>::get(2), &GPUScalar<Real>::get(1), true, false);

    return B;
}


TEST(FortranWrapper, SmokeTestAlex)
{
    using Real = double;

    constexpr size_t N = 257;

    //------------------------------------------------------------------
    // Alex's HP arrays
    //------------------------------------------------------------------

    XYZ<std::vector<Real>> d(std::vector<Real>(N + 1), std::vector<Real>(N + 1), std::vector<Real>(N + 1));
    const Real h = 1.0 / static_cast<Real>(N);

    d.x.front() = h * 0.5;
    d.y.front() = h * 0.5;
    d.z.front() = h * 0.5;

    for(size_t i=1;i<N;i++)
    {
        d.x[i]=h;
        d.y[i]=h;
        d.z[i]=h;
    }

    d.x.back() = h * 0.5;
    d.y.back() = h * 0.5;
    d.z.back() = h * 0.5;

    //------------------------------------------------------------------
    // RHS
    //------------------------------------------------------------------

    const size_t size = N * N * N;

    std::vector<Real> rhs(size,1.0);
    std::vector<Real> xStandard(size,-9999.0);
    std::vector<Real> xThomas(size,-9999.0);

    //------------------------------------------------------------------
    // Exactly the same API Alex calls
    //------------------------------------------------------------------

    size_t handleStandard = eigen::initEigenDecomp_d(
            N, N, N,
            d.x.data(), d.y.data(), d.z.data(),
            false, false, false,
            false, false, false,
            false, false, false,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            false, false
        );

    size_t handleThomas = eigen::initEigenDecomp_d(
            N, N, N,
            d.x.data(), d.y.data(), d.z.data(),
            false, false, false,
            false, false, false,
            false, false, false,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            false, true
        );

    eigen::solveEigenDecomp_d(handleStandard, xStandard.data(), rhs.data());
    eigen::solveEigenDecomp_d(handleThomas, xThomas.data(), rhs.data());


    //------------------------------------------------------------------
    // Diagnostics
    //------------------------------------------------------------------

    for(size_t i=0;i<size;i++) ASSERT_NEAR(xThomas[i], xStandard[i], 1e-7) <<  " i = " << i;

    eigen::finalizeEigenDecomp_d();
}

TEST(ImmersedEq, SolvesPrimes_3x2x1) {

    using Real = double;
    using Int  = int32_t;

    GridDim dim(3, 2, 1);
    Real3d delta(1, 1, 1);
    Handle hand;

    auto boundary = makeUniformBoundaryConfig<Real>(
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        delta, dim, false
    );

    size_t uStarSize =
        dim.numDims() * dim.size()
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
    for (size_t i = 0; i < uStarSize; ++i) rowIndsR[i] = static_cast<Int>(i);

    std::vector<Real> valsR(uStarSize, 1);

    std::vector<Real> UGamma(2, 3);

    double deltaT = 3.0 / 2.0;

    std::vector<Int> rowOffsetsB = {0, 1, 2};
    std::vector<Int> colIndsB    = {0, 1};
    std::vector<Real> valuesB    = {1, 1};

    std::vector<Real> f = {1, 2};

    std::vector<Real> p(dim.size(), 0);
    p[0] = 2;
    p[dim.size() - 1] = -2;

    std::vector<Real> resultP(dim.size(), 0);
    std::vector<Real> resultF(f.size(), 0);

    // ToeplitzLaplacian<Real>::printL(dim, hand, delta);

    ImmersedEq<Real, Int> imEq(boundary, f.size(), valsR.size(), p.data(), f.data(), deltaT, 1e-8, 1000);

    imEq.solve(resultP.data(), valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data());

    cudaDeviceSynchronize();

    std::vector<Real> expectedP1 = {-7.483126, -8.359545, -2.292128, -2.606740, -2.943816, -0.808988};

    for (size_t i = 0; i < resultP.size(); ++i) ASSERT_NEAR(resultP[i], expectedP1[i], 1e-4);

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

    std::vector<Real> expectedP2 = {7.61797, 10.1498, 2.955056, 3.08614, 3.72659, 1.67041};

    std::vector<Real> expectedF = {17.23595, 26.29962};

    for (size_t i = 0; i < resultP.size(); ++i) ASSERT_NEAR(resultP[i], expectedP2[i], 1e-4);

    for (size_t i = 0; i < resultF.size(); ++i) ASSERT_NEAR(resultF[i], expectedF[i], 1e-4);
}

TEST(ImmersedEq, SolvesImmeresed_Generic) {

    using Real = double;
    using Int  = int;

    GridDim dim(25, 30, 20);
    Real3d delta(1, 0.5, 2);
    Handle hand;
    auto boundary = makeUniformBoundaryConfig<Real>(
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

    BandedMat<Real> L = poisson::laplacian<double>(boundary, hand);
    auto LDense = SquareMat<Real>::create(dim.size());
    L.getDense(LDense, hand);

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

    ImmersedEq<Real, Int> imEq(boundary, fHost.size(), valuesB.size(), p.data(), fHost.data(), 1, 1e-12, 5);


    imEq.solve(resultP.data(), valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data());

    cudaDeviceSynchronize();

    for (size_t i = 0; i < resultP.size(); ++i) ASSERT_NEAR(resultP[i], i + 1, 1e-4);

}


template<typename T>
void expectMatrixNear(const Mat<T>& A, const Mat<T>& B, Handle& hand, T tol = 1e-6) {
    ASSERT_EQ(A._rows, B._rows);
    ASSERT_EQ(A._cols, B._cols);

    std::vector<T> aCpu(A.size(), 0), bCpu(B.size(), 0);
    A.get(aCpu.data(), hand);
    B.get(bCpu.data(), hand);

    cudaStreamSynchronize(hand);

    for (size_t i = 0; i < aCpu.size(); i++)
        ASSERT_NEAR(aCpu[i], bCpu[i], tol) << "Mismatch at (" << i % A._rows << "," << i / A._rows << ")";
}

// ---------- test ----------

TEST(KroneckerTripletTest, ProductMatchesMultOnIdentity) {
    using T = double;

    Handle hand3[3];
    Event event2[2];
    Handle& hand = hand3[0];

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


    KroneckerTriplet<T> kt(X, Y, Z, {false, false, false});

    auto I = SquareMat<T>::create(dim.size()).setToIdentity(hand);

    Mat<T> implicitResult = kt.product(hand);

    Mat<T> explicitResult = Mat<T>::create(dim.size(), dim.size());

    kt.mult(I, explicitResult, hand);

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
/**
 * Examines eigen vectors and values, confirming L V = V Lambda for all systems.
 * For uniform spacing, additionally confirms orthonormality (V^T V = I).
 */
template<typename T>
static void checkEigens(const SquareMat<T>& L, const SquareMat<T>& V, const Vec<T>& lambda, Handle& hand, const std::string& errorMsg, bool uniformDelta, T tol = 1e-8) {

    // std::cout << "checkEigens L = \n" << GpuOut<T>(L, hand) << std::endl << "V = \n" << GpuOut<T>(V, hand) << std::endl << "lambda = " << GpuOut<T>(lambda, hand) << std::endl;

    // ---------------------------------------------------------
    // UNIVERSAL CHECK: L * V = V * Lambda
    // ---------------------------------------------------------
    auto LV = SquareMat<T>::create(V._rows);
    L.mult(V, &LV, &hand, &GPUScalar<T>::get(1), &GPUScalar<T>::get(0), false, false);

    auto VLambda = SquareMat<T>::create(V._rows);
    auto Lambda = SquareMat<T>::create(V._cols);
    Lambda.fill(0, hand);
    Lambda.diag(0).set(lambda, hand);
    V.mult(Lambda, &VLambda, &hand, false, false);

    // Calculate Residual: LV - VLambda
    auto diff = SquareMat<T>::create(V._rows);
    diff.set(LV, hand);
    diff.add(VLambda, diff, GPUScalar<T>::get(1), GPUScalar<T>::get(-1), false, false, hand);

    std::vector<T> diffHost(diff.size(), 0);
    diff.get(diffHost.data(), hand);
    cudaStreamSynchronize(hand);

    for (size_t i = 0; i < diffHost.size(); ++i) {
        ASSERT_NEAR(diffHost[i], 0, tol)
            << errorMsg << "\nUniversal Eigenpair Check Failed (L * V != V * Lambda) at flat index " << i
            << " (row " << i % V._cols << ", col " << i / V._cols << ")"
            << " residual = " << diffHost[i];
    }

    // ---------------------------------------------------------
    // UNIFORM-ONLY CHECK: Orthonormality (V^T * V = I)
    // ---------------------------------------------------------
    if (uniformDelta) {
        auto VTV = SquareMat<T>::create(V._cols);
        // V^T * V
        V.mult(V, &VTV, &hand, &GPUScalar<T>::get(1), &GPUScalar<T>::get(0), true, false);

        auto I = SquareMat<T>::create(V._cols).setToIdentity(hand);

        auto orthoDiff = SquareMat<T>::create(V._cols);
        orthoDiff.set(VTV, hand);
        orthoDiff.add(I, orthoDiff, GPUScalar<T>::get(1), GPUScalar<T>::get(-1), false, false, hand);

        std::vector<T> orthoHost(orthoDiff.size(), 0);
        orthoDiff.get(orthoHost.data(), hand);
        cudaStreamSynchronize(hand);

        for (size_t i = 0; i < orthoHost.size(); ++i) {
            ASSERT_NEAR(orthoHost[i], 0, tol)
                << errorMsg << "\nUniform Basis is not orthonormal (V^T * V != I) at flat index " << i
                << " (row " << i % V._cols << ", col " << i / V._cols << ")"
                << " residual = " << orthoHost[i];
        }
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
 * @param  Maximum allowable difference for numerical validation.
 */
template<typename Real, typename BoundaryConfigT>
void verifyEigenSolverIdentity(const GridDim& dim, const BoundaryConfigT& boundary, Handle* hands, Event* events, Real tolerance, std::string& msg) {

    auto laplacian = poisson::laplacian<Real>(boundary, hands[0]);

    // std::cout << "verifyEigenSolverIdentity banded laplacian = \n" << GpuOut<Real>(laplacian, hands[0]) << std::endl;
    // std::cout << "verifyEigenSolverIdentity dense laplacian = \n" << GpuOut<Real>(laplacian.getDense(hands[0]), hands[0]) << std::endl;

    auto x = SimpleArray<Real>::create(dim.size(), hands[0]);

    std::vector<Real> xCpuOrig(dim.size());

    std::mt19937 rng(0);
    std::uniform_real_distribution<Real> dist(-1, 1);

    for (size_t i = 0; i < dim.size(); ++i) xCpuOrig[i] = dist(rng);
    x.set(xCpuOrig.data(), hands[0]);

    auto rhs = SimpleArray<Real>::create(dim.size(), hands[0]);
    laplacian.bandedMult(x, rhs, hands, GPUScalar<Real>::get(1), GPUScalar<Real>::get(0), false);

    // std::cout << "init x = " << GpuOut<Real>(x, hands[0]) << std::endl;
    // std::cout << "rhs = " << GpuOut<Real>(rhs, hands[0]) << std::endl;

    x.fill(0, hands[0]);

    if (dim.numDims() == 2) {
        EigenDecomp2d<Real> ed(boundary, hands, events[0]);
        ed.solve(x, rhs, hands[0]);
    } else {
        EigenDecomp3d<Real> ed(boundary, hands, events);
        ed.solve(x, rhs, hands[0]);
    }
    // std::cout << "verifyEigenSolverIdentity x_solution = \n" << GpuOut<Real>(x, hands[0]) << std::endl;
    cudaDeviceSynchronize();

    laplacian.bandedMult(x, rhs, hands, GPUScalar<Real>::get(1), GPUScalar<Real>::get(-1), false);

    auto normDevice = x.get(0);
    Real normHost[1];
    rhs.norm(normDevice, hands[0]);
    normDevice.get(normHost, hands[0]);

    // std::cout << "verifyEigenSolverIdentity norm = \n" << normHost[0] << std::endl;

    cudaDeviceSynchronize();
    ASSERT_NEAR(normHost[0], 0, tolerance) << " || L_i x - rhs|| > " << tolerance << std::endl << msg;

    if (dim.numDims()== 3) {
        EigenDecompThomas ed(boundary, hands, events);

        x.set(xCpuOrig.data(), hands[0]);
        laplacian.bandedMult(x, rhs, hands, GPUScalar<Real>::get(1), GPUScalar<Real>::get(0), false);
        x.fill(0, hands[0]);

        ed.solve(x, rhs, hands[0]);
        // std::cout << "verifyEigenSolverIdentity x_Thomas = \n" << GpuOut<Real>(x, hands[0]) << std::endl;

        laplacian.bandedMult(x, rhs, hands, GPUScalar<Real>::get(1), GPUScalar<Real>::get(-1), false);
        auto normDevice = x.get(0);
        Real normHost[1];
        rhs.norm(normDevice, hands[0]);
        normDevice.get(normHost, hands[0]);

        // std::cout << "verifyEigenSolverIdentity norm Thomas = \n" << normHost[0] << std::endl;
        cudaDeviceSynchronize();
        ASSERT_NEAR(normHost[0], 0, tolerance) << " Thomas || L_i x - rhs|| > " << tolerance;
    }
}



template <typename Real, typename Int, typename BoundaryConfigT>
void verifyImmersedEqWithBoundary(const BoundaryConfigT& boundary, Handle& hand, Real tolerance, const std::string& locMsg, Mat<Real> bufferNXNPlus5) {
    std::stringstream errorMsg;
    errorMsg << locMsg << '\n';
    const size_t n = boundary.dim().size();

    // 1. Setup sparse boundary matrix B
    std::vector<Real> x0Host(n);
    auto lhsOperator = bufferNXNPlus5.sqSubMat(0, 3, n);

    std::vector<Int> rowOffsetsB = {0, 2, 4};
    std::vector<Int> colIndsB = {0, 1, 0, 1};
    std::vector<Real> valuesB = {1.0, -1.0, -1.0, 1.0};
    //const BoundaryConfig<Real>& boundary, size_t n, std::vector<Real>& x0Host, Mat<Real>& lhsOperator, std::vector<Real>& rowOffsetsB, std::vector<Real>& valuesB, std::vector<Real>& colIndsB, Handle& hand
    auto B = basics<Real, Int>(boundary, n, x0Host, lhsOperator, rowOffsetsB, valuesB, colIndsB, hand);
    size_t numB = B.offsets.size() - 1;

    std::vector<Real> fHost(numB);
    for (size_t i = 0; i < numB; ++i) fHost[i] = static_cast<Real>((i + 1) * (i + 1));

    auto x0 = bufferNXNPlus5.col(0), f = SimpleArray<Real>::create(numB, hand);
    x0.set(x0Host.data(), hand);
    f.set(fHost.data(), hand);

    // 3. Construct explicit manufactured system: p0 = L x0 + 2 B^T B x0 - 2 B^T f - bc
    auto p0 = bufferNXNPlus5.col(1);
    poisson::laplacian<Real>(boundary, hand).bandedMult(x0, p0, &hand, GPUScalar<Real>::get(1), GPUScalar<Real>::get(0), false);

    auto rhs = bufferNXNPlus5.col(2);

    auto tempB = SimpleArray<Real>::create(numB, hand);
    auto sparseWorkSpace = SimpleArray<Real>::create(B.multWorkspaceSize(tempB, p0, GPUScalar<Real>::get(2), GPUScalar<Real>::get(1), true, hand), hand);

    lhsOperator.mult(x0, p0, &hand, &GPUScalar<Real>::get(1), &GPUScalar<Real>::get(0), false);

    B.mult(f, p0, GPUScalar<Real>::get(-2), GPUScalar<Real>::get(1), true, sparseWorkSpace, hand);           // p0 -= 2 B^T f

    B.mult(f, rhs, GPUScalar<Real>::get(2), GPUScalar<Real>::get(0), true, sparseWorkSpace, hand);           // p0 -= 2 B^T f

    SimpleArray<Real> bc = poisson::boundaryCorrection<Real>(boundary, hand);
    p0.add(bc, &GPUScalar<Real>::get(-1), &hand); // p0 -= bc

    rhs.add(bc, &GPUScalar<Real>::get(1), &hand);
    rhs.add(p0, &GPUScalar<Real>::get(1), &hand);


    errorMsg << "x0 = " << GpuOut<Real>(x0, hand) << "\nf = " << GpuOut<Real>(f, hand) << "\np0 = " << GpuOut<Real>(p0, hand) << '\n';


    // 5. Solve using immersed equation
    std::vector<Real> p0Host(n, 0), resultX(n, 0);
    p0.get(p0Host.data(), hand);
    cudaDeviceSynchronize();

    ImmersedEq<Real, Int> imEq(boundary, numB, B.values.size(), p0Host.data(), fHost.data(), 1, 1e-13, 1000);
    imEq.solve(resultX.data(), B.values.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data());
    cudaDeviceSynchronize();

    // 6. Validate recovered solution (ASSERT_NEAR will trigger immediate test exit on failure)

    auto resultDevice = bufferNXNPlus5.col(4 + n);
    resultDevice.set(resultX.data(), hand);

    SimpleArray<Real>& lhsMultX = p0;
    lhsOperator.mult(resultDevice, lhsMultX, &hand, &GPUScalar<Real>::get(1), &GPUScalar<Real>::get(0), false);

    std::vector<Real> lhsHost(n, 0), rhsHost(n, 0);
    lhsMultX.get(lhsHost.data(), hand);
    rhs.get(rhsHost.data(), hand);

    for (size_t i = 0; i < n; ++i)
        ASSERT_NEAR( lhsHost[i], rhsHost[i], tolerance) << locMsg << " - Residual mismatch at index " << i << "\nlhs = " << lhsHost[i] << " rhs = " << rhsHost[i] << '\n' << errorMsg.str();
}

template <typename Real>
void boundaryBattery(
    XYZ<bool> startIsN, XYZ<bool> endIsN, XYZ<Real> startVal, XYZ<Real> endVal,
    GridDim dim, const XYZ<std::vector<Real>>& deltas, bool isStag,
    Handle* hand3, Event* event2, double tolerance, Mat<Real> bufferNXNPlus5) {

    std::stringstream ss;
    ss << "startIsNeuman = " << startIsN
       << " endIsNeumann = " << endIsN
       << " isStagered = " << isStag
       << " startVal = " << startVal
       << " endVal = " << endVal
       << " dim = " << dim
       << "var spacing = (" << (deltas.x.size() > 1) << ", " << (deltas.y.size() > 1) << ", " << (deltas.z.size() < 1) << ")";

    std::string locMsg = ss.str();

    // Use the factory to deduce segment types at runtime and execute the tests
    buildBoundaryConfigAndLaunch<Real>(dim, deltas, startIsN, endIsN, startVal, endVal, isStag, 0, [&](const auto& boundary) {

        // Deduce uniformity at compile time
        using Config = std::decay_t<decltype(boundary)>;
        constexpr bool isXUniform = std::is_same_v<decltype(boundary.x), UniformSegment<Real>>;
        constexpr bool isYUniform = std::is_same_v<decltype(boundary.y), UniformSegment<Real>>;
        constexpr bool isZUniform = std::is_same_v<decltype(boundary.z), UniformSegment<Real>>;

        // Unconditionally instantiate the operators
        Laplacian1d<Real> laplacian1d(boundary, hand3[0]);
        Eigen<Real> laplacianEigen = Eigen<Real>::make(boundary, hand3, event2);

        // 1. Check X
        checkEigens(
            laplacian1d.dense(0, hand3[0]),
            laplacianEigen.vecs.x,
            laplacianEigen.vals.x,
            hand3[0],
            locMsg,
            isXUniform // Test orthonormality only if true
        );

        // 2. Check Y
        checkEigens(
            laplacian1d.dense(1, hand3[1]),
            laplacianEigen.vecs.y,
            laplacianEigen.vals.y,
            hand3[1],
            locMsg,
            isYUniform
        );

        // 3. Check Z (if 3D)
        if (dim.numDims() == 3) {
            checkEigens(
                laplacian1d.dense(2, hand3[2]),
                laplacianEigen.vecs.z,
                laplacianEigen.vals.z,
                hand3[2],
                locMsg,
                isZUniform
            );
        }

        //TODO:Uncomment below!
        // 4. Always run shared verification tests
        verifyEigenSolverIdentity(dim, boundary, hand3, event2, tolerance, locMsg);
        verifyImmersedEqWithBoundary<Real, int32_t>(boundary, hand3[0], tolerance, locMsg, bufferNXNPlus5);
    });
}

template <typename Real>
XYZ<std::vector<Real>> generateDeltas(const GridDim& dim, bool isVariable, size_t seedBase) {
    // 1. Uniform Spacing Path
    if (!isVariable) return XYZ<std::vector<Real>>({1.0}, {1.0}, {1.0});

    // 2. Variable Spacing Path
    std::mt19937 rng(seedBase); // Deterministic seed
    std::uniform_real_distribution<Real> dist(0.5, 2.0);

    std::vector<Real> deltaX(dim.cols + 1);
    for (auto& val : deltaX) val = dist(rng);

    std::vector<Real> deltaY(dim.rows + 1);
    for (auto& val : deltaY) val = dist(rng);

    std::vector<Real> deltaZ;
    if (dim.layers > 1) {
        deltaZ.resize(dim.layers + 1);
        for (auto& val : deltaZ) val = dist(rng);
    } else {
        deltaZ = { 1.0 }; // Degenerate Z-axis for 2D setups
    }

    // 3. Construct and return the XYZ struct
    return XYZ<std::vector<Real>>(deltaX, deltaY, deltaZ);
}

TEST(LaplacianMath, laplacian) {
    Handle hand3[3];
    Event event2[2];
    using Real = double;

    double tolerance = 1e-11;

    size_t maxDim = 21;
    size_t startRowsCols = 2;
    size_t dimStepSize = 9;

    size_t n = maxDim * maxDim * maxDim;
    auto buffer = Mat<Real>::create(n, n + 5);

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
                                                         for (size_t rows = startRowsCols; rows < maxDim; rows+= dimStepSize)
                                                             for (size_t cols = startRowsCols; cols < maxDim; cols += dimStepSize)
                                                                 for (size_t layers = 1; layers < maxDim; layers += dimStepSize) {
                                                                     GridDim dim(rows, cols, layers);
                                                                     XYZ<bool> startIsN(x0IsN, y0IsN, z0IsN);
                                                                     XYZ<bool> endIsN(x1IsN, y1IsN, z1IsN);
                                                                     XYZ<Real> startVal(static_cast<Real>(x0Val), static_cast<Real>(y0Val), static_cast<Real>(z0Val));
                                                                     XYZ<Real> endVal(static_cast<Real>(x1Val), static_cast<Real>(y1Val), static_cast<Real>(z1Val));
                                                                     bool isStagered = isStag;

                                                                     // GridDim dim(2, 2, 1);
                                                                     // XYZ<bool> startIsN(0, 0, 0);
                                                                     // XYZ<bool> endIsN(0, 0, 0);
                                                                     // XYZ<Real> startVal(0, 0, 0);
                                                                     // XYZ<Real> endVal(0, 0, 0);
                                                                     // bool isStagered = 0;

                                                                     // Determine spacing type and generate deterministic seed
                                                                     bool testVariableSpacing = ((dim.rows + dim.cols + dim.layers + startIsN.x + isStagered) % 2 == 0);
                                                                     size_t seedBase = dim.rows * dim.cols * dim.layers + startIsN.x;

                                                                     // Safely generate deltas via helper
                                                                     XYZ<std::vector<Real>> deltas = generateDeltas<Real>(dim, testVariableSpacing, seedBase);

                                                                     boundaryBattery<Real>(
                                                                         startIsN,
                                                                         endIsN,
                                                                         startVal,
                                                                         endVal,
                                                                         dim,
                                                                         deltas,
                                                                         isStagered,
                                                                         hand3, event2, tolerance,
                                                                         buffer.subMat(0, 0, dim.size(), dim.size() + 5)
                                                                     );
                                                                 }
}


int main(int argc, char **argv) {

    std::cout << "--- DIAGNOSTIC: Test Binary Starting ---" << std::endl;
    testing::InitGoogleTest(&argc, argv);

    auto mat = Mat<double>::create(2, 2);

    return RUN_ALL_TESTS();
}
