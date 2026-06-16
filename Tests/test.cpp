#include <gtest/gtest.h>

#include "solvers/EigenDecomp/EigenDecomp2d.h"
#include "solvers/EigenDecomp/EigenDecompThomas.cuh"
#include "poisson/Poisson.cuh"
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"
#include <cmath>
#include <random>


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

    // std::cout << "ceckEigens L = \n" << GpuOut<T>(L, hand) << std::endl;
    // std::cout << "ceckEigens V = \n" << GpuOut<T>(V, hand) << std::endl;
    // std::cout << "ceckEigens lambda = " << GpuOut<T>(lambda, hand) << std::endl;

    for (size_t i = 0; i < lambda.size(); ++i) {
        Vec<T> vi = V.col(i);

        vi.norm(normGpu, hand);
        T err = normGpu.get(hand) - 1;
        ASSERT_LT(err, tol)
            << errorMsg << "\nEigen Vector is not orthogonal, col " << i << " has a norm not equal to 1 "
            << " residual = " << err << "\n" << errorMsg << "\ncol = " << GpuOut<T>(vi, hand) << std::endl <<
                "with eigen val = " << GpuOut<T>(lambda.get(i), hand);

        Vec Lvi = SimpleArray<T>::create(L._rows, hand);
        L.mult(vi, Lvi, &hand, &GPUScalar<T>::get(1), &GPUScalar<T>::get(0), false);


        Vec<T> lam_vi = SimpleArray<T>::create(L._rows, hand);
        lam_vi.set(vi, hand);
        lam_vi.mult(lambda[i], &hand);

        Lvi.add(lam_vi, &GPUScalar<T>::get(-1), &hand);

        Lvi.norm(normGpu, hand);

        err = normGpu.get(hand);

        ASSERT_LT(std::abs(err), tol)
            << errorMsg << "\nEigenpair failed at index " << i
            << " residual = " << err;
    }

    for (size_t i = 0; i < V._cols; ++i)
        for (size_t j = i + 1; j < V._cols; ++j) {
            V.col(i).mult(V.col(j), normGpu, &hand);
            T err = normGpu.get(hand);
            ASSERT_LT(std::abs(err), tol)
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
        ASSERT_LT(std::abs(val), tol)
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
 * @param  Maximum allowable difference for numerical validation.
 */
template<typename Real, typename BoundaryConfigT>
void verifyEigenSolverIdentity(const GridDim& dim, const BoundaryConfigT& boundary, Handle* hands, Event* events, Real tolerance) {

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
    // std::cout << "verifyEigenSolverIdentity x_solution = \n" << GpuOut<Real>(x, hands[0]) << std::endl;
    cudaDeviceSynchronize();

    laplacian.bandedMult(x, rhs, hands, GPUScalar<Real>::get(1), GPUScalar<Real>::get(-1), false);

    auto normDevice = x.get(0);
    Real normHost[1];
    rhs.norm(normDevice, hands[0]);
    normDevice.get(normHost, hands[0]);

    // std::cout << "verifyEigenSolverIdentity norm = \n" << normHost[0] << std::endl;

    cudaDeviceSynchronize();
    ASSERT_NEAR(normHost[0], 0, tolerance) << " || L_i x - rhs|| > " << tolerance;

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
       << " dim = " << dim;

    std::string locMsg = ss.str();

    // Use the factory to deduce segment types at runtime and execute the tests
    buildBoundaryConfigAndLaunch<Real>(
        dim, deltas, startIsN, endIsN, startVal, endVal, isStag, 0 /* default stream */,
        [&](const auto& boundary) {

            Laplacian1d<Real> laplacian1d(boundary, hand3[0]);
            Eigen<Real> laplacianEigen = Eigen<Real>::make(boundary, hand3, event2);

            for (size_t i = 0; i < dim.numDims(); ++i)
                checkEigens(laplacian1d.dense(i, hand3[i]), laplacianEigen.vecs[i], laplacianEigen.vals[i], hand3[i], locMsg);

            //TODO: uncomment these.
            // verifyEigenSolverIdentity(dim, boundary, hand3, event2, tolerance);
            // verifyImmersedEqWithBoundary<Real, int32_t>(boundary, hand3[0], tolerance, locMsg, bufferNXNPlus5);
        }
    );
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

    size_t maxDim = 4;
    size_t startRowsCols = 2;
    size_t dimStepSize = 1;

    size_t n = maxDim * maxDim * maxDim;
    auto buffer = Mat<Real>::create(n, n + 5);

     // for (size_t x0IsN = 0; x0IsN < 2; ++x0IsN)
     //     for (size_t x1IsN = 0; x1IsN < 2; ++x1IsN)
     //         for (size_t y0IsN = 0; y0IsN < 2; ++y0IsN)
     //             for (size_t y1IsN = 0; y1IsN < 2; ++y1IsN)
     //                 for (size_t z0IsN = 0; z0IsN < 2; ++z0IsN)
     //                     for (size_t z1IsN = 0; z1IsN < 2; ++z1IsN)
     //                         for (size_t isStag = 0; isStag < 2; ++isStag)
     //                             for (size_t x0Val = 0; x0Val < 2; ++x0Val)
     //                                 for (size_t x1Val = 0; x1Val < 2; ++x1Val)
     //                                     for (size_t y0Val = 0; y0Val < 2; ++y0Val)
     //                                         for (size_t y1Val = 0; y1Val < 2; ++y1Val)
     //                                             for (size_t z0Val = 0; z0Val < 2; ++z0Val)
     //                                                 for (size_t z1Val = 0; z1Val < 2; ++z1Val)
     //                                                     for (size_t rows = startRowsCols; rows < maxDim; rows+= dimStepSize)
     //                                                         for (size_t cols = startRowsCols; cols < maxDim; cols += dimStepSize)
     //                                                             for (size_t layers = 1; layers < maxDim; layers += dimStepSize) {
                                                                     // GridDim dim(rows, cols, layers);
                                                                     // XYZ<bool> startIsN(x0IsN, y0IsN, z0IsN);
                                                                     // XYZ<bool> endIsN(x1IsN, y1IsN, z1IsN);
                                                                     // XYZ<Real> startVal(static_cast<Real>(x0Val), static_cast<Real>(y0Val), static_cast<Real>(z0Val));
                                                                     // XYZ<Real> endVal(static_cast<Real>(x1Val), static_cast<Real>(y1Val), static_cast<Real>(z1Val));
                                                                     // bool isStagered = isStag;

                                                                     GridDim dim(2, 2, 2);
                                                                     XYZ<bool> startIsN(0, 0, 0);
                                                                     XYZ<bool> endIsN(0, 0, 0);
                                                                     XYZ<Real> startVal(0, 0, 0);
                                                                     XYZ<Real> endVal(0, 0, 0);
                                                                     bool isStagered = 0;

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
                                                                 // }
}


int main(int argc, char **argv) {

    std::cout << "--- DIAGNOSTIC: Test Binary Starting ---" << std::endl;
    testing::InitGoogleTest(&argc, argv);

    auto mat = Mat<double>::create(2, 2);

    return RUN_ALL_TESTS();
}






// TEST(Benchmark, SolverRuntimes) {
//     using Real = double;
//
//     Handle hands[3];
//     Event events[2];
//     Real3d delta(1, 1, 1);
//
//     std::cout << DeviceMemory() << std::endl;
//
//     std::cout << std::setw(20) << "Grid dim N"
//               << std::setw(20) << "N x N x N"
//               << std::setw(25) << "EigenDecomp3d (ms)"
//               << std::setw(25) << "EigenDecompThomas (ms)"
//               << std::setw(25) << "||x_thomas - truth||"
//               << std::endl;
//     std::cout << std::string(115, '-') << std::endl;
//
//     const size_t reallocEvery = 100;
//
//     size_t bufferN        = 0;
//     size_t bufferTotal    = 0;
//
//     std::optional<SimpleArray<Real>> rhsBuf, xEigenBuf, xThomasBuf, sizeOfBBuf;
//     std::optional<Mat<Real>>         sizeOfBX3Buf;
//     std::optional<Singleton<Real>>   normResult;
//
//     auto reallocBuffers = [&](size_t total) {
//         // 1. Explicitly destroy the old objects to free GPU memory
//         rhsBuf.reset();
//         xEigenBuf.reset();
//         xThomasBuf.reset();
//         sizeOfBBuf.reset();
//         sizeOfBX3Buf.reset();
//         normResult.reset();
//
//         // 2. Allocate the new objects (peak memory is now strictly 'total')
//         rhsBuf     .emplace(SimpleArray<Real>::create(total, hands[0]));
//         xEigenBuf  .emplace(SimpleArray<Real>::create(total, hands[0]));
//         xThomasBuf .emplace(SimpleArray<Real>::create(total, hands[0]));
//         sizeOfBBuf .emplace(SimpleArray<Real>::create(total, hands[0]));
//         sizeOfBX3Buf.emplace(Mat<Real>::create(total, 3));
//         normResult .emplace(Singleton<Real>::create(hands[0]));
//         bufferTotal = total;
//     };
//
//     for (size_t n = 2; ; n += (n < 200 ? 2 : 1)) {
//         size_t total = n * n * n;
//
//         // Reallocate every reallocEvery iterations or if buffer is too small
//         if (!rhsBuf.has_value() || total > bufferTotal) {
//
//             size_t nAhead = n + reallocEvery * 2;
//             size_t aheadTotal = std::min(nAhead * nAhead * nAhead, total * 2);
//             try {
//                 reallocBuffers(aheadTotal);
//             } catch (...) {
//                 try {
//                     reallocBuffers(total);
//                 } catch (...) {
//                     std::cout << "OOM allocating buffers at N=" << n << std::endl;
//                     break;
//                 }
//             }
//         }
//
//         GridDim dim(n, n, n);
//         BoundaryConfig<Real> boundary(//TODO: test error with all Neimand boundary conditions.
//             {false, false, false},
//             {false, false, false},
//             {0, 0, 0},
//             {0, 0, 0},
//             delta, dim, false
//         );
//
//         std::vector<Real> rhsHost(total);
//
//         std::mt19937 rng(42);
//         std::uniform_real_distribution<Real> dist(-1, 1);
//
//         Real sum = Real(0);
//
//         for (auto& v : rhsHost) {
//             v = dist(rng);
//             sum += v;
//         }
//
//         Real mean = sum / Real(rhsHost.size());
//
//         for (auto& v : rhsHost) v -= mean;
//
//
//         // Slice sub-arrays from pre-allocated buffers — no new GPU allocation
//         auto rhsSub      = rhsBuf    ->subArray(0, total);
//         auto xEigenSub   = xEigenBuf ->subArray(0, total);
//         auto xThomasSub  = xThomasBuf->subArray(0, total);
//         auto sizeOfBSub  = sizeOfBBuf->subArray(0, total);
//         auto sizeOfBX3Sub = sizeOfBX3Buf->subMat(0, 0, total, 3);
//
//         rhsSub.set(rhsHost.data(), hands[0]);
//
//         // --- EigenDecomp3d ---
//         double eigenMs = -1;
//         try {
//             xEigenSub.fill(0, hands[0]);
//
//             cudaDeviceSynchronize();
//             auto t0 = std::chrono::high_resolution_clock::now();
//             EigenDecomp3d<Real> ed(boundary, hands, events, sizeOfBSub);
//             ed.solve(xEigenSub, rhsSub, hands[0]);
//             cudaDeviceSynchronize();
//             auto t1 = std::chrono::high_resolution_clock::now();
//
//             eigenMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
//         } catch (...) {
//             std::cout << std::setw(20) << (std::to_string(n))
//                       << std::setw(20) << total
//                       << "  OOM in EigenDecomp3d" << std::endl;
//             break;
//         }
//
//         // --- EigenDecompThomas ---
//         double thomasMs = -1;
//         try {
//
//             xThomasSub.fill(0, hands[0]);
//             cudaDeviceSynchronize();
//
//             auto t0 = std::chrono::high_resolution_clock::now();
//             EigenDecompThomas<Real> ed(boundary, hands, events, sizeOfBX3Sub);
//             ed.solve(xThomasSub, rhsSub, hands[0]);
//             cudaDeviceSynchronize();
//             auto t1 = std::chrono::high_resolution_clock::now();
//
//             thomasMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
//         } catch (...) {
//             std::cout << std::setw(20) << (std::to_string(n))
//                       << std::setw(20) << total
//                       << std::setw(25) << std::fixed << std::setprecision(3) << eigenMs
//                       << "  OOM in EigenDecompThomas" << std::endl;
//             break;
//         }
//
//         // --- Diff norm: xThomas -= xEigen, then ||xThomas|| ---
//         auto L = poisson::laplacian(boundary, hands[0]);//use recycled memory here to improve speed.
//         auto rhsFromThomas = Vec<Real>::create(xEigenSub.size(), hands[0]);
//         L.bandedMult(xThomasSub, rhsFromThomas, hands);
//         rhsFromThomas.add(rhsSub, &GPUScalar<Real>::get(-1), hands);
//
//         rhsFromThomas.norm(*normResult, hands[0]);
//         cudaDeviceSynchronize();
//         double diffNorm = normResult->get(hands[0]);
//
//         std::cout << std::setw(20) << (std::to_string(n))
//                   << std::setw(20) << total
//                   << std::setw(25) << std::fixed    << std::setprecision(3) << eigenMs
//                   << std::setw(25)                                           << thomasMs
//                   << std::setw(25) << std::scientific << std::setprecision(3) << diffNorm
//                   << std::endl;
//     }
// }