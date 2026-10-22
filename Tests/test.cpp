#include <gtest/gtest.h>

#include "solvers/EigenDecomp/EigenDecomp2d.h"
#include "solvers/EigenDecomp/EigenDecompThomas.cuh"
#include "poisson/Poisson.cuh"
#include <cmath>
#include <random>
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"


#include "wrapper/FortranBindings.hpp"
#include "deviceArrays/headers/DeviceMemory.h"
#include "immersedBoundary/ImmersedEquation.h"
#include "kronecker/KroneckerTriplet.h"
#include "poisson/Laplacian1d.cuh"

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

template <typename Real, typename Int, typename BoundaryConfigT>
SparseCSR<Real, Int> basics(const BoundaryConfigT& boundary, size_t n, std::vector<Real>& x0Host, SquareMat<Real>& lhsOperator,
        std::vector<Int>& rowOffsetsB, std::vector<Real>& valuesB, std::vector<Int>& colIndsB, Handle& hand) {

    std::mt19937 rng(42); // Deterministic seed for reproducible testing
    std::uniform_real_distribution<Real> dist(-5.0, 5.0);
    for (size_t i = 0; i < n; ++i) x0Host[i] = dist(rng);

    auto B = SparseCSR<Real, Int>::create(valuesB.size(), rowOffsetsB.size() - 1, n, hand);
    B.set(rowOffsetsB.data(), colIndsB.data(), valuesB.data(), hand);

    poisson::laplacian<Real>(boundary, hand).getDense(lhsOperator, hand);
    auto denseB = Mat<Real>::create(rowOffsetsB.size() - 1, n, hand);
    B.getDense(denseB, hand);
    denseB.mult(denseB, &lhsOperator, &hand, &GPUScalar<Real>::get(2, hand), &GPUScalar<Real>::get(1, hand), true, false);

    return B;
}

TEST(CudaWrapper, transpose) {
    using Real = double;
    Handle hand;
    size_t rows = 2, cols = 3;
    std::vector<Real> host = {1, 2, 3, 4, 5, 6};
    std::vector<Real> transposedHost = {1, 3, 5, 2, 4, 6};
    auto a = Mat<Real>::create(rows, cols, hand);
    auto aT = Mat<Real>::create(cols, rows, hand);
    auto aTFromHost = Mat<Real>::create(cols, rows, hand);

    a.set(host.data(), hand);
    aTFromHost.set(transposedHost.data(), hand);

    a.transpose(aT, hand);

    expectMatrixNear(aT, aTFromHost, hand);
}



TEST(FortranWrapper, MultipleEigenSolversSideBySide){
    using Real = double;
    constexpr size_t N = 64;

    // Setup arrays for grid spacing
    XYZ<std::vector<Real>> d(std::vector<Real>(N + 1), std::vector<Real>(N + 1), std::vector<Real>(N + 1));
    const Real h = 1.0 / static_cast<Real>(N);

    d.x.front() = h * 0.5; d.y.front() = h * 0.5; d.z.front() = h * 0.5;
    for(size_t i=1; i<N; i++) {
        d.x[i] = h; d.y[i] = h; d.z[i] = h;
    }
    d.x.back() = h * 0.5; d.y.back() = h * 0.5; d.z.back() = h * 0.5;

    const size_t size = N * N * N;
    std::vector<Real> rhs(size, 1.0);

    // Result buffers
    std::vector<Real> resStandard(size, -9999.0);
    std::vector<Real> resThomas(size, -9999.0);

    // 1. Initialize three independent solvers side-by-side
    // NOTE: the three ints after the delta pointers are the per-axis
    // eigen::LaplOperatorT segment kinds; passing `false` here would silently
    // become UniformDeltaNodeCenteredLapl (0) and the N+1 delta arrays would
    // be ignored past d[0].
    size_t h1 = eigen::initEigenDecomp_d(
            N, N, N,
            d.x.data(), d.y.data(), d.z.data(),
            eigen::VariableDeltaLapl, eigen::VariableDeltaLapl, eigen::VariableDeltaLapl,
            false, false, false, false, false, false,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0, 0); // Standard

    size_t h2 = eigen::initEigenDecomp_d(
            N, N, N, d.x.data(), d.y.data(), d.z.data(),
            eigen::VariableDeltaLapl, eigen::VariableDeltaLapl, eigen::VariableDeltaLapl,
            false, false, false, false, false, false,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, 0, 0);  // Thomas (was passing thomas=false)

        // 2. Verify the wrapper assigned unique handles to the vector
    ASSERT_NE(h1, h2);

    // 3. Solve them sequentially. solve_eigen_decomp_d/synch no longer take
    // a data argument -- write the RHS directly into each solver's own
    // pinned buffer first (h1 and h2 each have their own, from their own
    // separate init call, so no aliasing between them), then launch/wait,
    // then read the solution back out of that same buffer.
    Real* pinned1 = eigen::pinnedPtr<Real>(h1);
    Real* pinned2 = eigen::pinnedPtr<Real>(h2);
    std::copy(rhs.begin(), rhs.end(), pinned1);
    std::copy(rhs.begin(), rhs.end(), pinned2);

    eigen::solveEigenDecomp_d(h1);
    eigen::solveEigenDecomp_d(h2);

    eigen::synch_d(h1);
    eigen::synch_d(h2);

    std::copy(pinned1, pinned1 + size, resStandard.begin());
    std::copy(pinned2, pinned2 + size, resThomas.begin());

    // 4. Diagnostics: If memory bled across solvers, the results would diverge
    for(size_t i=0; i<size; i++)
        ASSERT_NEAR(resStandard[i], resThomas[i], 1e-7)
            << " Standard/Thomas disagreement at i = " << i;

    eigen::finalizeEigenDecomp();
}

TEST(FortranWrapper, ThomasMatchesStandardOnLargeFluxGrid)
{
    using Real = double;

    constexpr size_t N = 257;

    //------------------------------------------------------------------
    // Spacing arrays for a uniform grid
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
    // The same interface a Fortran caller uses
    //------------------------------------------------------------------

    // The h/2, h, ..., h, h/2 arrays above are the spacing of
    // cell centers of N uniform cells tiling [0,1] wall-to-wall.  That is the
    // FluxLapl geometry, so this smoke test now exercises the flux operator
    // through both solve paths.
    size_t handleStandard = eigen::initEigenDecomp_d(
            N, N, N,
            d.x.data(), d.y.data(), d.z.data(),
            eigen::FluxLapl, eigen::FluxLapl, eigen::FluxLapl,
            false, false, false,
            false, false, false,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            false, 0, 0
        );

    size_t handleThomas = eigen::initEigenDecomp_d(
            N, N, N,
            d.x.data(), d.y.data(), d.z.data(),
            eigen::FluxLapl, eigen::FluxLapl, eigen::FluxLapl,
            false, false, false,
            false, false, false,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            true, 0, 0
        );

    // solve_eigen_decomp_d/synch no longer take a data argument -- same
    // pinned-buffer pattern as MultipleEigenSolversSideBySide above.
    Real* pinnedStandard = eigen::pinnedPtr<Real>(handleStandard);
    Real* pinnedThomas   = eigen::pinnedPtr<Real>(handleThomas);
    std::copy(rhs.begin(), rhs.end(), pinnedStandard);
    std::copy(rhs.begin(), rhs.end(), pinnedThomas);

    eigen::solveEigenDecomp_d(handleStandard);
    eigen::solveEigenDecomp_d(handleThomas);

    eigen::synch_d(handleStandard);
    eigen::synch_d(handleThomas);

    std::copy(pinnedStandard, pinnedStandard + size, xStandard.begin());
    std::copy(pinnedThomas, pinnedThomas + size, xThomas.begin());


    //------------------------------------------------------------------
    // Diagnostics
    //------------------------------------------------------------------

    for(size_t i=0;i<size;i++) ASSERT_NEAR(xThomas[i], xStandard[i], 1e-7) <<  " i = " << i;

    eigen::finalizeEigenDecomp();
}

TEST(ImmersedEq, SolvesPrimes_3x2x1) {

    using Real = double;
    using Int  = int32_t;

    GridDim dim(3, 2, 1);
    Real3d delta(1, 1, 1);
    Handle hand;

    auto boundaryHost = makeUniformBoundaryConfigHost<Real>(
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

    ImmersedEq<Real, Int> imEq(boundaryHost.forDevice(), f.size(), valsR.size(), p.data(), f.data(), deltaT, 1e-8, 1000);

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

TEST(ImmersedEq, SolvesImmersed_Generic) {

    using Real = double;
    using Int  = int;

    GridDim dim(25, 30, 20);
    Real3d delta(1, 0.5, 2);
    Handle hand;
    auto boundary = makeUniformBoundaryConfigHost<Real>(
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
    auto BDense = Mat<Real>::create(B.rows, B.cols, hand);
    B.getDense(BDense, hand);

    std::vector<Real> xHost(dim.size(), 0);
    for (size_t i = 0; i < xHost.size(); ++i) xHost[i] = i + 1.0;
    auto x = SimpleArray<Real>::create(dim.size(), hand);
    x.set(xHost.data(), hand);

    BandedMat<Real> L = poisson::laplacian<double>(boundary.forDevice(), hand);
    auto LDense = SquareMat<Real>::create(dim.size(), hand);
    L.getDense(LDense, hand);

    auto LPlus2BTBx = SimpleArray<Real>::create(dim.size(), hand);
    LPlus2BTBx.fill(0, hand);

    BDense.mult(BDense, &LDense, &hand,&GPUScalar<Real>::get(2, hand), &GPUScalar<Real>::get(1, hand),  true, false);
    LDense.mult(x, LPlus2BTBx, &hand, &GPUScalar<Real>::get(1, hand), &GPUScalar<Real>::get(0, hand), false);

    std::vector<Real> fHost(rowOffsetsB.size() - 1, 0);
    fHost[0] = 1;
    fHost[1] = 2;
    auto f = SimpleArray<Real>::create(fHost.size(), hand);
    f.set(fHost.data(), hand);

    auto TwoBTF = SimpleArray<Real>::create(dim.size(), hand);
    BDense.mult(f, TwoBTF, &hand, &GPUScalar<Real>::get(2, hand), &GPUScalar<Real>::get(0, hand), true);

    LPlus2BTBx.add(TwoBTF, &GPUScalar<Real>::get(-1, hand), &hand);

    std::vector<Real> p(dim.size(), 0);
    LPlus2BTBx.get(p.data(), hand);

    std::vector<Real> resultP(dim.size(), 0);
    std::vector<Real> resultF(fHost.size(), 0);

    ImmersedEq<Real, Int> imEq(boundary.forDevice(), fHost.size(), valuesB.size(), p.data(), fHost.data(), 1, 1e-12, 5);


    imEq.solve(resultP.data(), valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data());

    cudaDeviceSynchronize();

    for (size_t i = 0; i < resultP.size(); ++i) ASSERT_NEAR(resultP[i], i + 1, 1e-4);

}

// ---------- test ----------

TEST(KroneckerTripletTest, ProductMatchesMultOnIdentity) {
    using T = double;

    Handles hands(3);
    Handle& hand = hands[0];

    // Nontrivial sizes
    GridDim dim(3, 2, 2);

    auto X = SquareMat<T>::create(dim.cols, hand);
    std::vector<T> xCpu ={2, 1, 7, 2};
    X.set(xCpu.data(), hand);

    auto Y = SquareMat<T>::create(dim.rows, hand);
    std::vector<T> yCpu ={3, 1, 5, 1, 3, 1, 0, 1, 3};
    Y.set(yCpu.data(), hand);

    auto Z = SquareMat<T>::create(dim.layers, hand);
    std::vector<T> zCpu ={5, 2, 1, 5};
    Z.set(zCpu.data(), hand);


    KroneckerTriplet<T> kt(X, Y, Z, {false, false, false});

    auto I = SquareMat<T>::create(dim.size(), hand).setToIdentity(hand);

    Mat<T> implicitResult = kt.product(hand);

    Mat<T> explicitResult = Mat<T>::create(dim.size(), dim.size(), hand);

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
 * @param errorMsg Anything that should be appended to an error message.
 * @param tol The tolerance.
 */
/**
 * Examines eigen vectors and values, confirming L V = V Lambda for all systems.
 * For uniform spacing, additionally confirms orthonormality (V^T V = I).
 */
template<typename T>
static void checkEigens(const SquareMat<T>& L, const SquareMat<T>& V, const Vec<T>& lambda, Handle& hand, const std::string& errorMsg, bool uniformDelta, T tol = 1e-8) {

    // ---------------------------------------------------------
    // UNIVERSAL CHECK: L * V = V * Lambda
    // ---------------------------------------------------------
    auto LV = SquareMat<T>::create(V._rows, hand);
    L.mult(V, &LV, &hand, &GPUScalar<T>::get(1, hand), &GPUScalar<T>::get(0, hand), false, false);

    auto VLambda = SquareMat<T>::create(V._rows, hand);
    auto Lambda = SquareMat<T>::create(V._cols, hand);
    Lambda.fill(0, hand);
    Lambda.diag(0).set(lambda, hand);
    V.mult(Lambda, &VLambda, &hand, false, false);

    // Calculate Residual: LV - VLambda
    auto diff = SquareMat<T>::create(V._rows, hand);
    diff.set(LV, hand);
    diff.add(VLambda, diff, GPUScalar<T>::get(1, hand), GPUScalar<T>::get(-1, hand), false, false, hand);

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
        auto VTV = SquareMat<T>::create(V._cols, hand);
        // V^T * V
        V.mult(V, &VTV, &hand, &GPUScalar<T>::get(1, hand), &GPUScalar<T>::get(0, hand), true, false);

        auto I = SquareMat<T>::create(V._cols, hand).setToIdentity(hand);

        auto orthoDiff = SquareMat<T>::create(V._cols, hand);
        orthoDiff.set(VTV, hand);
        orthoDiff.add(I, orthoDiff, GPUScalar<T>::get(1, hand), GPUScalar<T>::get(-1, hand), false, false, hand);

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
 * @param hands Handles for stream management; at least 3.
 * @param  Maximum allowable difference for numerical validation.
 */
template<typename Real, typename segX, typename segY, typename segZ>
void verifyEigenSolverIdentity(const GridDim& dim, const BoundaryConfigHost<Real, segX, segY, segZ>& boundary, Handles& hands, Real tolerance, std::string& msg) {

    auto laplacian = poisson::laplacian<Real>(boundary.forDevice(), hands[0]);

    auto x = SimpleArray<Real>::create(dim.size(), hands[0]);

    std::vector<Real> xCpuOrig(dim.size());

    std::mt19937 rng(0);
    std::uniform_real_distribution<Real> dist(-1, 1);

    for (size_t i = 0; i < dim.size(); ++i) xCpuOrig[i] = dist(rng);
    x.set(xCpuOrig.data(), hands[0]);

    auto rhs = SimpleArray<Real>::create(dim.size(), hands[0]);
    laplacian.bandedMult(x, rhs, &hands[0], GPUScalar<Real>::get(1, hands[0]), GPUScalar<Real>::get(0, hands[0]), false);

    x.fill(0, hands[0]);

    if (dim.numDims() == 2) {
        EigenDecomp2d<Real> ed(boundary.forDevice(), hands);
        ed.solve(x, rhs, hands[0]);
    } else {
        EigenDecomp3d<Real> ed(boundary.forDevice(), hands);
        ed.solve(x, rhs, hands[0]);
    }
    cudaDeviceSynchronize();

    laplacian.bandedMult(x, rhs, &hands[0], GPUScalar<Real>::get(1, hands[0]), GPUScalar<Real>::get(-1, hands[0]), false);

    auto normDevice = x.get(0);
    Real normHost[1];
    rhs.norm(normDevice, hands[0]);
    normDevice.get(normHost, hands[0]);

    cudaDeviceSynchronize();
    ASSERT_NEAR(normHost[0], 0, tolerance) << " || L_i x - rhs|| > " << tolerance << std::endl << msg;

    if (dim.numDims()== 3) {
        EigenDecompThomas ed(boundary, hands);

        x.set(xCpuOrig.data(), hands[0]);
        laplacian.bandedMult(x, rhs, &hands[0], GPUScalar<Real>::get(1, hands[0]), GPUScalar<Real>::get(0, hands[0]), false);
        x.fill(0, hands[0]);

        ed.solve(x, rhs, hands[0]);

        laplacian.bandedMult(x, rhs, &hands[0], GPUScalar<Real>::get(1, hands[0]), GPUScalar<Real>::get(-1, hands[0]), false);
        auto normDevice = x.get(0);
        Real normHost[1];
        rhs.norm(normDevice, hands[0]);
        normDevice.get(normHost, hands[0]);

        cudaDeviceSynchronize();
        ASSERT_NEAR(normHost[0], 0, tolerance) << " Thomas || L_i x - rhs|| > " << tolerance;
    }
}



template <typename Real, typename Int, typename segX, typename segY, typename segZ>
void verifyImmersedEqWithBoundary(const BoundaryConfig<Real, segX, segY, segZ>& boundary, Handle& hand, Real tolerance, const std::string& locMsg, Mat<Real> bufferNXNPlus5) {
    std::stringstream errorMsg;
    errorMsg << locMsg << '\n';
    const size_t n = boundary.dim().size();

    // 1. Setup sparse boundary matrix B
    std::vector<Real> x0Host(n);
    auto lhsOperator = bufferNXNPlus5.sqSubMat(0, 3, n);

    std::vector<Int> rowOffsetsB = {0, 2, 4};
    std::vector<Int> colIndsB = {0, 1, 0, 1};
    std::vector<Real> valuesB = {1.0, -1.0, -1.0, 1.0};
    auto B = basics<Real, Int>(boundary, n, x0Host, lhsOperator, rowOffsetsB, valuesB, colIndsB, hand);
    size_t numB = B.offsets.size() - 1;

    std::vector<Real> fHost(numB);
    for (size_t i = 0; i < numB; ++i) fHost[i] = static_cast<Real>((i + 1) * (i + 1));

    auto x0 = bufferNXNPlus5.col(0), f = SimpleArray<Real>::create(numB, hand);
    x0.set(x0Host.data(), hand);
    f.set(fHost.data(), hand);

    // 3. Construct explicit manufactured system: p0 = L x0 + 2 B^T B x0 - 2 B^T f - bc
    auto p0 = bufferNXNPlus5.col(1);
    poisson::laplacian<Real>(boundary, hand).bandedMult(x0, p0, &hand, GPUScalar<Real>::get(1, hand), GPUScalar<Real>::get(0, hand), false);

    auto rhs = bufferNXNPlus5.col(2);

    auto tempB = SimpleArray<Real>::create(numB, hand);
    auto sparseWorkSpace = SimpleArray<Real>::create(B.multWorkspaceSize(tempB, p0, GPUScalar<Real>::get(2, hand), GPUScalar<Real>::get(1, hand), true, hand), hand);

    lhsOperator.mult(x0, p0, &hand, &GPUScalar<Real>::get(1, hand), &GPUScalar<Real>::get(0, hand), false);

    B.mult(f, p0, GPUScalar<Real>::get(-2, hand), GPUScalar<Real>::get(1, hand), true, sparseWorkSpace, hand);           // p0 -= 2 B^T f

    B.mult(f, rhs, GPUScalar<Real>::get(2, hand), GPUScalar<Real>::get(0, hand), true, sparseWorkSpace, hand);           // p0 -= 2 B^T f

    SimpleArray<Real> bc = poisson::boundaryCorrection<Real>(boundary, hand);
    p0.add(bc, &GPUScalar<Real>::get(-1, hand), &hand); // p0 -= bc

    rhs.add(bc, &GPUScalar<Real>::get(1, hand), &hand);
    rhs.add(p0, &GPUScalar<Real>::get(1, hand), &hand);


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
    lhsOperator.mult(resultDevice, lhsMultX, &hand, &GPUScalar<Real>::get(1, hand), &GPUScalar<Real>::get(0, hand), false);

    std::vector<Real> lhsHost(n, 0), rhsHost(n, 0);
    lhsMultX.get(lhsHost.data(), hand);
    rhs.get(rhsHost.data(), hand);

    for (size_t i = 0; i < n; ++i)
        ASSERT_NEAR( lhsHost[i], rhsHost[i], tolerance) << locMsg << " - Residual mismatch at index " << i << "\nlhs = " << lhsHost[i] << " rhs = " << rhsHost[i] << '\n' << errorMsg.str();
}

template <typename Real>
void boundaryBattery(
    XYZ<bool> startIsN, XYZ<bool> endIsN, XYZ<Real> startVal, XYZ<Real> endVal,
    GridDim dim, const XYZ<std::vector<Real>>& deltas, XYZ<eigen::LaplOperatorT> isStag,
    Handles& hands, double tolerance, Mat<Real> bufferNXNPlus5) {

    std::stringstream ss;
    ss << "startIsNeumann = " << startIsN
       << " endIsNeumann = " << endIsN
       << " isStagered = " << isStag
       << " startVal = " << startVal
       << " endVal = " << endVal
       << " dim = " << dim
       << "var spacing = (" << (deltas.x.size() > 1) << ", " << (deltas.y.size() > 1) << ", " << (deltas.z.size() < 1) << ")";

    std::string locMsg = ss.str();

    // Use the factory to deduce segment types at runtime and execute the tests
    buildBoundaryConfigAndLaunch<Real>(dim, deltas, startIsN, endIsN, startVal, endVal, isStag, hands[0], [&](const auto& boundaryHost) {

        // Deduce uniformity at compile time
        using Config = std::decay_t<decltype(boundaryHost)>;
        // boundaryHost.{x,y,z} are AxisSegmentHost<...>, not segments, so
        // comparing against UniformSegment directly was always false and the
        // orthonormality check never ran.  Compare the host's SegmentType.
        constexpr bool isXUniform = std::is_same_v<typename std::decay_t<decltype(boundaryHost.x)>::SegmentType, UniformSegment<Real>>;
        constexpr bool isYUniform = std::is_same_v<typename std::decay_t<decltype(boundaryHost.y)>::SegmentType, UniformSegment<Real>>;
        constexpr bool isZUniform = std::is_same_v<typename std::decay_t<decltype(boundaryHost.z)>::SegmentType, UniformSegment<Real>>;

        // Unconditionally instantiate the operators
        Laplacian1d<Real> laplacian1d(boundaryHost.forDevice(), hands[0]);
        Eigen<Real> laplacianEigen = Eigen<Real>::make(boundaryHost.forDevice(), hands);

        // 1. Check X
        checkEigens(
            laplacian1d.dense(0, hands[0]),
            laplacianEigen.vecs.x,
            laplacianEigen.vals.x,
            hands[0],
            locMsg,
            isXUniform // Test orthonormality only if true
        );

        // 2. Check Y
        checkEigens(
            laplacian1d.dense(1, hands[1]),
            laplacianEigen.vecs.y,
            laplacianEigen.vals.y,
            hands[1],
            locMsg,
            isYUniform
        );

        // 3. Check Z (if 3D)
        if (dim.numDims() == 3) {
            checkEigens(
                laplacian1d.dense(2, hands[2]),
                laplacianEigen.vecs.z,
                laplacianEigen.vals.z,
                hands[2],
                locMsg,
                isZUniform
            );
        }

        // 4. Always run shared verification tests
        verifyEigenSolverIdentity(dim, boundaryHost, hands, tolerance, locMsg);
        verifyImmersedEqWithBoundary<Real, int32_t>(boundaryHost.forDevice(), hands[0], tolerance, locMsg, bufferNXNPlus5);
    });
}

/**
 * Kind-consistent delta generation.  The deltas MUST match the segment kind:
 *  - uniform kinds: a single spacing (the variable-length arrays would be
 *    ignored past element 0 anyway);
 *  - VariableDeltaLapl: n+1 free random deltas;
 *  - FluxLapl: deltas derived from n random cell WIDTHS (d_0 = W_0/2,
 *    d_i = (W_{i-1}+W_i)/2, d_n = W_{n-1}/2), because FluxLapl requires --
 *    and makeFvmWidths verifies -- that the deltas come from cells tiling
 *    the segment wall-to-wall.  Free random deltas would (correctly) throw.
 */
template <typename Real>
std::vector<Real> generateAxisDeltas(size_t n, eigen::LaplOperatorT kind, std::mt19937& rng) {
    if (!eigen::hasVariableDelta(kind) || n <= 1) return {1.0};
    std::uniform_real_distribution<Real> dist(0.5, 2.0);
    std::vector<Real> d(n + 1);
    if (kind == eigen::FluxLapl) {
        std::vector<Real> w(n);
        for (auto& val : w) val = dist(rng);
        d[0] = w[0] / 2;
        for (size_t i = 1; i < n; ++i) d[i] = (w[i-1] + w[i]) / 2;
        d[n] = w[n-1] / 2;
    } else {
        for (auto& val : d) val = dist(rng);
    }
    return d;
}

template <typename Real>
XYZ<std::vector<Real>> generateDeltas(const GridDim& dim, eigen::LaplOperatorT kind, size_t seedBase) {
    std::mt19937 rng(seedBase); // Deterministic seed
    return XYZ<std::vector<Real>>(
        generateAxisDeltas<Real>(dim.cols, kind, rng),
        generateAxisDeltas<Real>(dim.rows, kind, rng),
        dim.layers > 1 ? generateAxisDeltas<Real>(dim.layers, kind, rng) : std::vector<Real>{1.0}
    );
}

template<typename Real>
void laplacianBattery(int kindLo, int kindHi, bool fullValueMatrix) {
    Handles hands(3);

    size_t maxDim = 20;
    size_t startRowsCols = 2;
    size_t dimStepSize = 6;

    size_t n = maxDim * maxDim * maxDim;
    auto buffer = Mat<Real>::create(n, n + 5, hands[0]);

    size_t count = 0;

    // Non-zero boundary values only exercise the RHS correction; one mixed
    // combination catches sign errors without the 2^6 value matrix.
    const size_t valMax = fullValueMatrix ? 2 : 1;

     for (size_t x0IsN = 0; x0IsN < 2; ++x0IsN)
         for (size_t x1IsN = 0; x1IsN < 2; ++x1IsN)
             for (size_t y0IsN = 0; y0IsN < 2; ++y0IsN)
                 for (size_t y1IsN = 0; y1IsN < 2; ++y1IsN)
                     for (size_t z0IsN = 0; z0IsN < 2; ++z0IsN)
                         for (size_t z1IsN = 0; z1IsN < 2; ++z1IsN)
                             for (int isStag = kindLo; isStag <= kindHi; ++isStag)
                                 for (size_t x0Val = 0; x0Val < valMax + 0; ++x0Val)
                                     for (size_t x1Val = 0; x1Val < valMax; ++x1Val)
                                         for (size_t y0Val = 0; y0Val < valMax; ++y0Val)
                                             for (size_t y1Val = 0; y1Val < valMax; ++y1Val)
                                                 for (size_t z0Val = 0; z0Val < valMax; ++z0Val)
                                                     for (size_t z1Val = 0; z1Val < (fullValueMatrix ? 2 : 2); ++z1Val)
                                                         for (size_t rows = startRowsCols; rows < maxDim; rows+= dimStepSize)
                                                             for (size_t cols = startRowsCols; cols < maxDim; cols += dimStepSize)
                                                                 for (size_t layers = 1; layers < maxDim; layers += dimStepSize) {
                                                                     GridDim dim(rows, cols, layers);
                                                                     XYZ<bool> startIsN(x0IsN, y0IsN, z0IsN);
                                                                     XYZ<bool> endIsN(x1IsN, y1IsN, z1IsN);
                                                                     XYZ<Real> startVal(static_cast<Real>(x0Val), static_cast<Real>(y0Val), static_cast<Real>(z0Val));
                                                                     XYZ<Real> endVal(static_cast<Real>(x1Val), static_cast<Real>(y1Val), static_cast<Real>(z1Val));
                                                                     auto laplOp = static_cast<eigen::LaplOperatorT>(isStag);
                                                                     // Degenerate z-axis (2D case) must stay uniform: a
                                                                     // variable/flux kind with a single delta has no nodes.
                                                                     auto laplOpZ = dim.layers > 1 ? laplOp : eigen::UniformDeltaNodeCenteredLapl;
                                                                     XYZ<eigen::LaplOperatorT> isStagered(laplOp, laplOp, laplOpZ);

                                                                     // Deltas must match the segment kind (uniform kinds ignore
                                                                     // all but d[0]; FluxLapl requires a wall-to-wall cell tiling).
                                                                     size_t seedBase = dim.rows * dim.cols * dim.layers + startIsN.x + isStag;

                                                                     XYZ<std::vector<Real>> deltas = generateDeltas<Real>(dim, laplOp, seedBase);

                                                                     double tolerance = 1e-11 * std::sqrt(static_cast<double>(dim.size()));


                                                                     boundaryBattery<Real>(
                                                                         startIsN,
                                                                         endIsN,
                                                                         startVal,
                                                                         endVal,
                                                                         dim,
                                                                         deltas,
                                                                         isStagered,
                                                                         hands, tolerance,
                                                                         buffer.subMat(0, 0, dim.size(), dim.size() + 5)
                                                                     );
                                                                 }
    std::cout << "  [battery] completed " << count << " configurations" << std::endl;
}

// Fast, targeted: only the new FluxLapl kind, every BC combination and dim,
// homogeneous + one nonzero value pattern.  Fails within seconds if the flux
// operator or its eigendecomposition is wrong.
TEST(LaplacianMath, fluxLaplacian) {
    laplacianBattery<double>(eigen::FluxLapl, eigen::FluxLapl, false);
}

// All four kinds with the reduced value matrix -- the routine regression run.
TEST(LaplacianMath, laplacian) {
    laplacianBattery<double>(eigen::UniformDeltaNodeCenteredLapl, eigen::FluxLapl, false);
}



template<typename Real, typename segX, typename segY, typename segZ>
void verifyShiftedEigenSolverIdentity(
    const BoundaryConfigHost<Real, segX, segY, segZ>& boundary,
    Handles& hands,
    Real shift,
    Real tolerance,
    const std::string& msg
) {
    auto laplacian = poisson::laplacian<Real>(boundary.forDevice(), hands[0]);
    auto dim = boundary.forDevice().dim();
    std::vector<Real> x0Host(dim.size());
    for (size_t i = 0; i < dim.size(); ++i) x0Host[i] = static_cast<Real>(i + 1);

    auto x = SimpleArray<Real>::create(dim.size(), hands[0]);
    x.set(x0Host.data(), hands[0]);

    auto b = SimpleArray<Real>::create(dim.size(), hands[0]);
    laplacian.bandedMult(x, b, &hands[0], GPUScalar<Real>::get(1, hands[0]), GPUScalar<Real>::get(0, hands[0]), false);
    auto shiftGpu = Singleton<Real>::create(-shift, hands[0]);
    b.add(x, &shiftGpu, &hands[0]);   // y0 += (-shift) * x0

    x.fill(0, hands[0]);

    EigenDecomp3d<Real> ed(boundary.forDevice(), hands, shift);
    ed.solve(x, b, hands[0]);
    std::vector<Real> x1Host(dim.size());
    x.get(x1Host.data(), hands[0]);

    for (size_t i = 0; i < dim.size(); ++i) ASSERT_NEAR(x0Host[i], x1Host[i], tolerance);

    EigenDecompThomas<Real, segX> edThomas(boundary, hands, shift);
    edThomas.solve(x, b, hands[0]);
    x.get(x1Host.data(), hands[0]);

    for (size_t i = 0; i < dim.size(); ++i) ASSERT_NEAR(x0Host[i], x1Host[i], tolerance);
}

// Small size: fast, first line of defense.
//
// NOTE: this test cannot catch an EigenDecompThomas bug that swaps which
// axis gets eigen-transformed and which is left in real space for the
// Thomas sweep.  X, Y and Z here all have the same length (dimLength) AND
// uniformCellDeltas(dimLength) is called three times with the same n, so X,
// Y and Z carry numerically identical data -- applying "the X transform"
// instead of "the Y transform" (or vice versa) produces the same numbers
// either way. See ThomasAxisAssignment.VxShapedAsymmetricGrid /
// VyShapedAsymmetricGrid below, which deliberately break that symmetry.
TEST(ShiftedEigen, FluxLaplacian_SmallScale) {
    Handles hands(3);
    size_t dimLength = 11;
    GridDim dim(dimLength, dimLength, dimLength);
    XYZ<eigen::LaplOperatorT> kind(eigen::FluxLapl, eigen::FluxLapl, eigen::FluxLapl);
    XYZ<bool> allNeumann(true, true, true);
    XYZ<double> zeroR3(0.0, 0.0, 0.0);

    auto uniformCellDeltas = [](size_t n) {
        std::vector<double> d(n + 1, 1.0);
        d.front() = 0.5; d.back() = 0.5;
        return d;
    };
    XYZ<std::vector<double>> deltas(uniformCellDeltas(dim.cols), uniformCellDeltas(dim.rows), uniformCellDeltas(dim.layers));

    buildBoundaryConfigAndLaunch<double>(dim, deltas, allNeumann, allNeumann, zeroR3, zeroR3, kind, hands[0],
        [&](const auto& boundaryHost) {
            std::string msg = "FluxLaplacian shifted identity, N=11";
            verifyShiftedEigenSolverIdentity<double>(boundaryHost, hands, /*shift=*/3.48568e6, /*tol=*/1e-6, msg);
        });
}

// ============================================================================
// Thomas axis-assignment regression tests.
//
// EigenDecompThomas::solve() is supposed to pre-transform the two
// "eigenvalue" axes (Y and Z -- multLEigenValInverse reads
// lapEigen.vals.y[row] and lapEigen.vals.z[layer] directly as eigenvalues)
// into eigenspace, and leave the Thomas axis (X / dim.cols, solved via the
// real boundaryX tridiagonal stencil) in real space. If it instead
// transforms X and Z and leaves Y untouched -- i.e. if multCols/multRows
// are swapped -- every field using EigenDecompThomas should fail an
// L*x_solved ~= rhs round trip, UNLESS X, Y and Z happen to carry
// numerically identical data (see the note on FluxLaplacian_SmallScale
// above for why that test can't see this).
//
// These two tests give X, Y and Z different sizes AND build the per-axis
// deltas with a kind that can vary per axis, so no two axes can accidentally
// share data. They reuse verifyEigenSolverIdentity (already defined above),
// which checks EigenDecomp3d -- the all-axes-symmetric method, not under
// suspicion -- against the same ground truth (L * x_orig) as
// EigenDecompThomas. If EigenDecomp3d passes and EigenDecompThomas fails on
// the identical boundary config, the fault is isolated to
// EigenDecompThomas itself: nothing about the Laplacian, the boundary
// conditions, or Eigen<T>::make() is in question.
// ============================================================================

// Like generateDeltas, but each axis gets its own LaplOperatorT kind instead
// of sharing one kind across all three -- needed to put VariableDeltaLapl on
// just one axis the way GpuSolverSetup.cu does for Vx/Vy/Vz.
template <typename Real>
XYZ<std::vector<Real>> generateMixedKindDeltas(const GridDim& dim, eigen::LaplOperatorT kindX, eigen::LaplOperatorT kindY, eigen::LaplOperatorT kindZ, size_t seedBase) {
    std::mt19937 rng(seedBase); // Deterministic seed, advanced sequentially across axes
    return XYZ<std::vector<Real>>(
        generateAxisDeltas<Real>(dim.cols, kindX, rng),
        generateAxisDeltas<Real>(dim.rows, kindY, rng),
        dim.layers > 1 ? generateAxisDeltas<Real>(dim.layers, kindZ, rng) : std::vector<Real>{1.0}
    );
}

// Mirrors Vx's real construction in GpuSolverSetup.cu: the non-uniform axis
// (VariableDeltaLapl) is in slot 1 / "x" / dim.cols, with FluxLapl in Y and
// Z, all-Dirichlet boundaries (matching BoundaryConditions::readVelocity).
// dim.cols/rows/layers are deliberately all different (5, 7, 9) so no two
// axes can share data.
TEST(ThomasAxisAssignment, VxShapedAsymmetricGrid) {
    Handles hands(3);
    using Real = double;

    GridDim dim(/*rows=*/7, /*cols=*/5, /*layers=*/9);
    XYZ<eigen::LaplOperatorT> kind(eigen::VariableDeltaLapl, eigen::FluxLapl, eigen::FluxLapl);
    XYZ<bool> allDirichlet(false, false, false);
    XYZ<Real> zeroR3(0.0, 0.0, 0.0);

    XYZ<std::vector<Real>> deltas = generateMixedKindDeltas<Real>(dim, kind.x, kind.y, kind.z, /*seedBase=*/12345);

    buildBoundaryConfigAndLaunch<Real>(dim, deltas, allDirichlet, allDirichlet, zeroR3, zeroR3, kind, hands[0],
        [&](const auto& boundaryHost) {
            std::string msg = "Vx-shaped asymmetric grid (dim.cols=5 Variable, dim.rows=7 Flux, dim.layers=9 Flux)";
            verifyEigenSolverIdentity<Real>(dim, boundaryHost, hands, /*tolerance=*/1e-9, msg);
        });
}

// Mirrors Vy's real construction: the non-uniform axis is in slot 2 / "y" /
// dim.rows instead of slot 1, with FluxLapl in X and Z. The bug this is
// checking for is in EigenDecompThomas::solve() itself and doesn't depend on
// which slot carries VariableDeltaLapl, so this should fail the same way as
// VxShapedAsymmetricGrid above if solve()'s axis assignment is wrong.
TEST(ThomasAxisAssignment, VyShapedAsymmetricGrid) {
    Handles hands(3);
    using Real = double;

    GridDim dim(/*rows=*/6, /*cols=*/8, /*layers=*/5);
    XYZ<eigen::LaplOperatorT> kind(eigen::FluxLapl, eigen::VariableDeltaLapl, eigen::FluxLapl);
    XYZ<bool> allDirichlet(false, false, false);
    XYZ<Real> zeroR3(0.0, 0.0, 0.0);

    XYZ<std::vector<Real>> deltas = generateMixedKindDeltas<Real>(dim, kind.x, kind.y, kind.z, /*seedBase=*/67890);

    buildBoundaryConfigAndLaunch<Real>(dim, deltas, allDirichlet, allDirichlet, zeroR3, zeroR3, kind, hands[0],
        [&](const auto& boundaryHost) {
            std::string msg = "Vy-shaped asymmetric grid (dim.cols=8 Flux, dim.rows=6 Variable, dim.layers=5 Flux)";
            verifyEigenSolverIdentity<Real>(dim, boundaryHost, hands, /*tolerance=*/1e-9, msg);
        });
}

// ============================================================================
// Hartmann=1000 investigation: Vy's own diagnostic log (CfdSimulation.cu,
// buildAndLaunchVy) shows RHSy/DGr going into vy_.solved().solve() is
// perfectly uniform -- bit-identical at (x=10,y=50,z=50), (90,50,50),
// (50,50,10) and (50,50,90) -- while Vy coming OUT of that same solve call
// breaks the x<->z exchange symmetry the square duct requires, by a factor
// of 4-7x at those same four points. That localizes the bug to inside
// vy_.solved().solve() itself: not any RHS-build kernel upstream of it (all
// exonerated by the uniform RHS), and -- per
// ThomasAxisAssignment.VyShapedAsymmetricGrid above, which already passes
// with synthetic per-axis deltas of mismatched sizes -- not the generic
// EigenDecompThomas engine in the abstract either.
//
// The one thing neither of those checks covers: Vy's REAL mesh-derived
// delta arrays, from this exact config (Nx=Ny=Nz=100, aspect_ratio=
// width_ratio=1.0, stretching_factor=10.0 -- example_config.json), as
// opposed to VyShapedAsymmetricGrid's synthetic random deltas. If this test
// fails, the bug is in the real delta data (Mesh::generate /
// computeAxisSpacing) interacting with the Thomas solve. If it passes, the
// bug only shows up when all 6 fields' solvers are co-resident and
// concurrent on the same device in the full simulation -- i.e. a
// stream/handle or buffer-aliasing issue -- since this test runs Vy's
// solver in total isolation.
//
// computeAxisSpacing is copied verbatim from Mesh.cu (CfdSolver project);
// it's self-contained (just <cmath>/<vector>), so it can live here as-is.
// ============================================================================

template<typename Real>
struct AxisSpacingForTest {
    std::vector<Real> hFull;
    std::vector<Real> hHalf;
};

template<typename Real>
AxisSpacingForTest<Real> computeAxisSpacingForTest(int n, Real scale, Real stretchFactor) {
    int n1 = n + 1;
    int n2 = n1 + 1;
    std::vector<Real> node(n1 + 1);
    std::vector<Real> nodeHalf(n2 + 1);

    Real h = Real(1) / Real(n1);
    node[0] = Real(0);
    for (int i = 1; i <= n1; ++i) node[i] = node[i - 1] + h;
    Real tanhHalfStretch = std::tanh(Real(0.5) * stretchFactor);
    for (int i = 0; i <= n1; ++i)
        node[i] = scale * (Real(0.5) + Real(0.5) * std::tanh(stretchFactor * (node[i] - Real(0.5))) / tanhHalfStretch);

    nodeHalf[0] = Real(0);
    nodeHalf[n2] = scale;
    for (int i = 1; i <= n1; ++i) nodeHalf[i] = Real(0.5) * (node[i] + node[i - 1]);

    AxisSpacingForTest<Real> result;
    result.hFull.resize(n + 1);
    for (int i = 0; i <= n; ++i) result.hFull[i] = node[i + 1] - node[i];

    result.hHalf.resize(n1 + 1);
    for (int i = 0; i <= n1; ++i) result.hHalf[i] = nodeHalf[i + 1] - nodeHalf[i];

    return result;
}

TEST(ThomasAxisAssignment, VyRealMeshUniformRhsStaysSymmetric) {
    Handles hands(3);
    using Real = double;

    // example_config.json's exact values for the Hartmann=1000 case.
    const int Nx = 100, Ny = 100, Nz = 100;
    const Real aspectRatio = 1.0, widthRatio = 1.0, stretchFactor = 10.0;

    AxisSpacingForTest<Real> xAxis = computeAxisSpacingForTest<Real>(Nx, aspectRatio, stretchFactor);
    AxisSpacingForTest<Real> yAxis = computeAxisSpacingForTest<Real>(Ny, widthRatio, stretchFactor);
    AxisSpacingForTest<Real> zAxis = computeAxisSpacingForTest<Real>(Nz, Real(1), stretchFactor);

    // Vy's real construction (GpuSolverSetup.cu::GpuSolverBuilder::build()):
    //   GridDim(ny, nx1, nz1), deltaForVy = XYZ(hostHpx(), hostHy12(), hostHpz())
    //                        = XYZ(x.hHalf, y.hFull, z.hHalf),
    //   segType = XYZ(FluxLapl, VariableDeltaLapl, FluxLapl),
    //   all-Dirichlet (no-slip velocity) on every boundary.
    GridDim dim(/*rows=*/Ny, /*cols=*/Nx + 1, /*layers=*/Nz + 1);
    XYZ<std::vector<Real>> deltas(xAxis.hHalf, yAxis.hFull, zAxis.hHalf);
    XYZ<eigen::LaplOperatorT> kind(eigen::FluxLapl, eigen::VariableDeltaLapl, eigen::FluxLapl);
    XYZ<bool> allDirichlet(false, false, false);
    XYZ<Real> zeroR3(0.0, 0.0, 0.0);

    // Vy's REAL construction passes shift.velocity, not 0 -- confirmed by
    // grepping GpuSolverSetup.cu's actual buildFieldSolver<Real> call site
    // for vy, which passes shift.velocity (not Real(0) the way pressure and
    // potential do). shift.velocity = correctionCoefficient /
    // (timeStep * gravitationalScale), per computeShiftCoefficients(), and
    // for this Hartmann=1000 config (grashof=1.85185185e7, timeStep=1e-4)
    // that's a large number (~6.5e7) -- this is a Helmholtz solve, not a
    // pure Poisson one. An earlier version of this test used the default
    // (shift=0) constructor and failed with a pattern that didn't match the
    // real diagnostic at all (it broke even plain reflection symmetry
    // within a single axis, x=10 vs x=90 alone) -- a strong sign that
    // version was testing the wrong equation, not exercising the real bug.
    // This version uses the exact real shift so there's no such ambiguity.
    const Real grashof = Real(1.85185185e7), timeStep = Real(1e-4);
    const Real correctionCoefficient = Real(1.5);
    const Real gravitationalScale = Real(1) / std::sqrt(grashof);
    const Real shiftVelocity = correctionCoefficient / (timeStep * gravitationalScale);

    buildBoundaryConfigAndLaunch<Real>(dim, deltas, allDirichlet, allDirichlet, zeroR3, zeroR3, kind, hands[0],
        [&](const auto& boundaryHost) {
            EigenDecompThomas ed(boundaryHost, hands, shiftVelocity);

            auto rhs = SimpleArray<Real>::create(dim.size(), hands[0]);
            rhs.fill(1.0, hands[0]);  // uniform -- the value itself doesn't matter, only its uniformity

            auto x = SimpleArray<Real>::create(dim.size(), hands[0]);
            x.fill(0, hands[0]);
            ed.solve(x, rhs, hands[0]);

            std::vector<Real> xHost(dim.size());
            x.get(xHost.data(), hands[0]);
            cudaDeviceSynchronize();

            // Residual check: does solve() actually satisfy (L - shift*I)*x_solved
            // ~= rhs for THIS config -- the documented contract (EigenDecomp3d.cuh:
            // "helmholtzShift set to a non zero value to solve (L - sigma I)x = b"),
            // NOT plain L*x=rhs. An earlier version of this check omitted the
            // -shift*x term entirely and reported a large "residual" that was
            // really just shift*||x|| leaking through a check of the wrong
            // equation -- caught by cross-referencing AlexCudaCompatibility.f90,
            // which uses this exact thomas=true/shift=shiftVelocity/FluxLapl-
            // VariableDeltaLapl-FluxLapl combination and gets a stable result, so
            // the shared solver code can't be at fault. This version checks the
            // equation solve() actually promises. X and Z carry bit-identical
            // delta data here (aspect_ratio=width_ratio=1.0) and identical
            // FluxLapl kind and BCs -- the only difference is which GridDim slot
            // they're in (X=dim.cols is always Thomas-solved directly; Z=dim.layers
            // is eigen-transformed with Y). If this residual is small, solve()
            // correctly inverted (L-shift*I), so any x<->z asymmetry found below
            // means L itself -- as built for this exact FluxLapl/VariableDeltaLapl/
            // FluxLapl + large-shift combination -- is not actually x<->z symmetric.
            // If the residual is large even with the shift term included, solve()
            // itself is failing to invert (L-shift*I) for this data/shift regime.
            {
                auto laplacian = poisson::laplacian<Real>(boundaryHost.forDevice(), hands[0]);
                auto residual = SimpleArray<Real>::create(dim.size(), hands[0]);
                residual.fill(Real(1.0), hands[0]);  // same uniform rhs value used above
                laplacian.bandedMult(x, residual, &hands[0], GPUScalar<Real>::get(1, hands[0]), GPUScalar<Real>::get(-1, hands[0]), false);
                // residual now holds L*x - rhs; subtract shift*x to get
                // (L - shift*I)*x - rhs, matching what solve() actually promises.
                auto negShift = Singleton<Real>::create(-shiftVelocity, hands[0]);
                residual.add(x, &negShift, &hands[0]);

                auto normDevice = x.get(0);
                Real normHost[1];
                residual.norm(normDevice, hands[0]);
                normDevice.get(normHost, hands[0]);
                cudaDeviceSynchronize();

                Real rhsNorm = std::sqrt(Real(dim.size()));  // exact norm of a uniform-1.0 vector
                Real relResidual = normHost[0] / rhsNorm;
                EXPECT_LT(relResidual, 1e-6)
                    << "||(L-shift*I)*x_solved - rhs|| / ||rhs|| = " << relResidual << " (||residual||=" << normHost[0] << ")"
                    << " -- if this is small despite the symmetry failures below, L itself is not x<->z"
                    << " symmetric as built for this config; if this is large, solve() itself is failing here.";
            }

            // Same flat layout FieldView::at(i,j,k) uses on these same solved
            // buffers in the real simulation: x (dim.cols) fastest, then y
            // (dim.rows), then z (dim.layers). 1-based, matching FieldView.
            auto at = [&](int xi, int yi, int zi) -> Real {
                size_t idx = (size_t)(xi - 1)
                    + (size_t)dim.cols * ((size_t)(yi - 1) + (size_t)dim.rows * (size_t)(zi - 1));
                return xHost[idx];
            };

            Real xVarying0 = at(10, 50, 50), xVarying1 = at(90, 50, 50);
            Real zVarying0 = at(50, 50, 10), zVarying1 = at(50, 50, 90);

            auto relDiff = [](Real a, Real b) {
                Real absA = std::abs(a), absB = std::abs(b);
                Real scale = absA > absB ? absA : absB;
                if (scale < Real(1e-300)) scale = Real(1e-300);
                return std::abs(a - b) / scale;
            };

            EXPECT_LT(relDiff(xVarying0, xVarying1), 1e-6)
                << "x<->z symmetry broken on the x-varying pair: at(10,50,50)=" << xVarying0
                << " vs at(90,50,50)=" << xVarying1;
            EXPECT_LT(relDiff(zVarying0, zVarying1), 1e-6)
                << "x<->z symmetry broken on the z-varying pair: at(50,50,10)=" << zVarying0
                << " vs at(50,50,90)=" << zVarying1;
            EXPECT_LT(relDiff(xVarying0, zVarying0), 1e-6)
                << "x<->z exchange symmetry broken: at(10,50,50)=" << xVarying0
                << " vs at(50,50,10)=" << zVarying0;

            // Exhaustive x<->z scan, replacing the 4-point spot check as the
            // decisive evidence: dim.cols == dim.layers == 101 here, so every
            // (x,y,z) has a well-defined (z,y,x) partner, not just the four
            // hand-picked points above. This is cheap (runs on the already-
            // downloaded xHost, no GPU calls) and rules out the 4-point check
            // itself being an indexing artifact -- a real indexing bug would
            // scatter inconsistently across the grid; a real construction bug
            // should show a smooth, grid-wide pattern.
            Real maxRelDiff = 0;
            int worstX = -1, worstY = -1, worstZ = -1;
            size_t violations = 0, totalChecked = 0;
            for (int yi = 1; yi <= (int)dim.rows; ++yi) {
                for (int xi = 1; xi <= (int)dim.cols; ++xi) {
                    for (int zi = xi + 1; zi <= (int)dim.layers; ++zi) {
                        Real a = at(xi, yi, zi), b = at(zi, yi, xi);
                        Real d = relDiff(a, b);
                        ++totalChecked;
                        if (d >= 1e-6) ++violations;
                        if (d > maxRelDiff) { maxRelDiff = d; worstX = xi; worstY = yi; worstZ = zi; }
                    }
                }
            }
            std::cout << "[VyRealMeshUniformRhsStaysSymmetric] exhaustive x<->z scan: "
                      << violations << "/" << totalChecked << " pairs violate 1e-6 relative tolerance; worst at ("
                      << worstX << "," << worstY << "," << worstZ << ")=" << at(worstX, worstY, worstZ)
                      << " vs (" << worstZ << "," << worstY << "," << worstX << ")=" << at(worstZ, worstY, worstX)
                      << " relDiff=" << maxRelDiff << "\n";
            EXPECT_LT(maxRelDiff, 1e-6)
                << violations << "/" << totalChecked << " (x,y,z)<->(z,y,x) pairs violate symmetry; worst at ("
                << worstX << "," << worstY << "," << worstZ << ") vs (" << worstZ << "," << worstY << "," << worstX << ")";

            // ----------------------------------------------------------------
            // Decisive A/B isolation. EigenDecompThomas treats X via a
            // real-space Thomas sweep and Y/Z via eigen-transform
            // (multCols/multDepths); EigenDecomp3d treats X, Y AND Z
            // uniformly via eigen-transform, with no Thomas sweep for any
            // axis (see the file comment above
            // TEST(ThomasAxisAssignment, VxShapedAsymmetricGrid) for the
            // standing hypothesis this directly tests, and
            // verifyShiftedEigenSolverIdentity's existing
            // `EigenDecomp3d<Real> ed(boundary.forDevice(), hands, shift);`
            // for the confirmed constructor shape). Solving the identical
            // boundaryHost / shiftVelocity / uniform-rhs problem through
            // EigenDecomp3d instead gives a controlled comparison: same
            // physical problem, only the X-axis treatment changes (eigen
            // instead of Thomas).
            //   - If EigenDecomp3d's result IS x<->z symmetric while
            //     EigenDecompThomas's is not (confirmed broken above), the
            //     fault is isolated to the real-space Thomas-sweep-for-X
            //     machinery itself -- not Laplacian1d, not Eigen<T>::make(),
            //     not the mesh/boundary construction, since all of those are
            //     shared by both solver classes.
            //   - If EigenDecomp3d is ALSO asymmetric, the fault is upstream
            //     and shared by both solvers, and EigenDecompThomas-specific
            //     code (the Thomas sweep itself) is cleared.
            {
                EigenDecomp3d<Real> ed3d(boundaryHost.forDevice(), hands, shiftVelocity);

                auto x3d = SimpleArray<Real>::create(dim.size(), hands[0]);
                x3d.fill(0, hands[0]);
                auto rhs3d = SimpleArray<Real>::create(dim.size(), hands[0]);
                rhs3d.fill(1.0, hands[0]);
                ed3d.solve(x3d, rhs3d, hands[0]);

                std::vector<Real> x3dHost(dim.size());
                x3d.get(x3dHost.data(), hands[0]);
                cudaDeviceSynchronize();

                auto at3d = [&](int xi, int yi, int zi) -> Real {
                    size_t idx = (size_t)(xi - 1)
                        + (size_t)dim.cols * ((size_t)(yi - 1) + (size_t)dim.rows * (size_t)(zi - 1));
                    return x3dHost[idx];
                };

                Real maxRelDiff3d = 0;
                int worstX3d = -1, worstY3d = -1, worstZ3d = -1;
                size_t violations3d = 0, totalChecked3d = 0;
                for (int yi = 1; yi <= (int)dim.rows; ++yi) {
                    for (int xi = 1; xi <= (int)dim.cols; ++xi) {
                        for (int zi = xi + 1; zi <= (int)dim.layers; ++zi) {
                            Real a = at3d(xi, yi, zi), b = at3d(zi, yi, xi);
                            Real d = relDiff(a, b);
                            ++totalChecked3d;
                            if (d >= 1e-6) ++violations3d;
                            if (d > maxRelDiff3d) { maxRelDiff3d = d; worstX3d = xi; worstY3d = yi; worstZ3d = zi; }
                        }
                    }
                }
                std::cout << "[VyRealMeshUniformRhsStaysSymmetric] EigenDecomp3d A/B comparison -- "
                          << "same boundaryHost/shift/rhs, X treated via eigen-transform instead of "
                          << "Thomas sweep: " << violations3d << "/" << totalChecked3d
                          << " pairs violate 1e-6 relative tolerance; worst at ("
                          << worstX3d << "," << worstY3d << "," << worstZ3d << ")=" << at3d(worstX3d, worstY3d, worstZ3d)
                          << " vs (" << worstZ3d << "," << worstY3d << "," << worstX3d << ")=" << at3d(worstZ3d, worstY3d, worstX3d)
                          << " relDiff=" << maxRelDiff3d << "\n";
                EXPECT_LT(maxRelDiff3d, 1e-6)
                    << violations3d << "/" << totalChecked3d << " (x,y,z)<->(z,y,x) pairs violate symmetry under "
                    << "EigenDecomp3d (eigen-transform for all 3 axes, no Thomas sweep); worst at ("
                    << worstX3d << "," << worstY3d << "," << worstZ3d << ") vs (" << worstZ3d << "," << worstY3d << "," << worstX3d << ")";
            }

            // ----------------------------------------------------------------
            // Direct eigenvalue-spectrum comparison. lapEigen is computed
            // once, by Eigen<T>::make(), in the EigenDecompSolver base
            // constructor, and shared by every solver subclass built on top
            // of it -- including both ed (EigenDecompThomas) and ed3d
            // (EigenDecomp3d) above, even though EigenDecompThomas's own
            // solve() never reads vals.x/vecs.x (X stays in real space via
            // the Thomas sweep there; see multLEigenValInverse in
            // EigenDecompThomas.cu, which reads only eValsY[row]/eValsZ[layer]).
            // X and Z carry bit-identical FluxLapl delta data in this config
            // (aspect_ratio=width_ratio=1.0), so their 1D operators are the
            // same matrix and their eigenvalue SETS (sorted) must be
            // identical, regardless of which "slot" each occupies. This
            // checks Eigen<T>::make() itself, directly, isolating whether the
            // asymmetry confirmed above (bit-identical in both
            // EigenDecompThomas and EigenDecomp3d, so not specific to either
            // solver's own dispatch/combination logic) already exists at the
            // eigendecomposition step, or only appears later when the
            // eigenvalues/eigenvectors get combined/applied.
            //
            // Pulled out via SquareMat::diag(0).set(...)+get(), mirroring
            // checkEigens' own Lambda construction above (SquareMat<T>
            // Lambda = create(...); Lambda.fill(0,hand);
            // Lambda.diag(0).set(lambda,hand);) rather than assuming Vec<T>
            // has its own direct host-download method.
            {
                size_t xCount = dim.cols, zCount = dim.layers;
                ASSERT_EQ(xCount, zCount) << "test assumes dim.cols == dim.layers for this config";

                auto LambdaX = SquareMat<Real>::create(xCount, hands[0]);
                LambdaX.fill(0, hands[0]);
                LambdaX.diag(0).set(ed.lapEigen.vals.x, hands[0]);
                std::vector<Real> LambdaXHost(LambdaX.size());
                LambdaX.get(LambdaXHost.data(), hands[0]);

                auto LambdaZ = SquareMat<Real>::create(zCount, hands[0]);
                LambdaZ.fill(0, hands[0]);
                LambdaZ.diag(0).set(ed.lapEigen.vals.z, hands[0]);
                std::vector<Real> LambdaZHost(LambdaZ.size());
                LambdaZ.get(LambdaZHost.data(), hands[0]);
                cudaDeviceSynchronize();

                std::vector<Real> sortedX(xCount), sortedZ(zCount);
                for (size_t i = 0; i < xCount; ++i) sortedX[i] = LambdaXHost[i * xCount + i];
                for (size_t i = 0; i < zCount; ++i) sortedZ[i] = LambdaZHost[i * zCount + i];

                // UNSORTED comparison first, in raw mode-index order, before
                // sortedX/sortedZ get mutated below -- a sorted-set match
                // (checked next) doesn't rule out the values being the same
                // SET in a different per-mode order, which matters if the
                // paired eigenvectors didn't get reordered the same way.
                Real maxUnsortedValDiff = 0;
                size_t worstUnsortedIdx = 0;
                for (size_t i = 0; i < xCount; ++i) {
                    Real d = relDiff(sortedX[i], sortedZ[i]);
                    if (d > maxUnsortedValDiff) { maxUnsortedValDiff = d; worstUnsortedIdx = i; }
                }
                std::cout << "[VyRealMeshUniformRhsStaysSymmetric] UNSORTED (raw mode-index order) eigenvalue "
                          << "comparison (lapEigen.vals.x[k] vs lapEigen.vals.z[k]): max relative diff = "
                          << maxUnsortedValDiff << " at index " << worstUnsortedIdx << "\n";

                // Manual insertion sort (avoids relying on <algorithm> being
                // transitively included); N=101 here, so O(N^2) is negligible.
                auto simpleSort = [](std::vector<Real>& v) {
                    for (size_t i = 1; i < v.size(); ++i) {
                        Real key = v[i];
                        size_t j = i;
                        while (j > 0 && v[j - 1] > key) { v[j] = v[j - 1]; --j; }
                        v[j] = key;
                    }
                };
                simpleSort(sortedX);
                simpleSort(sortedZ);

                Real maxEigRelDiff = 0;
                size_t worstEigIdx = 0;
                for (size_t i = 0; i < xCount; ++i) {
                    Real d = relDiff(sortedX[i], sortedZ[i]);
                    if (d > maxEigRelDiff) { maxEigRelDiff = d; worstEigIdx = i; }
                }
                std::cout << "[VyRealMeshUniformRhsStaysSymmetric] sorted eigenvalue-spectrum comparison "
                          << "(lapEigen.vals.x vs lapEigen.vals.z, " << xCount << " values each): "
                          << "max relative diff = " << maxEigRelDiff << " at sorted index " << worstEigIdx
                          << " (sortedX=" << sortedX[worstEigIdx] << ", sortedZ=" << sortedZ[worstEigIdx] << ")\n";
                EXPECT_LT(maxEigRelDiff, 1e-6)
                    << "X and Z axes get identical input deltas/kind/BCs but Eigen<T>::make() produces "
                    << "different eigenvalue spectra (sorted) -- max relative diff " << maxEigRelDiff
                    << " at sorted index " << worstEigIdx << " (sortedX=" << sortedX[worstEigIdx]
                    << ", sortedZ=" << sortedZ[worstEigIdx] << ")";
            }

            // ----------------------------------------------------------------
            // Direct eigenVECTOR comparison. L_x == L_z exactly (just proven
            // above) and vals.x/vals.z match both as a sorted set AND
            // (checked just above) in raw mode-index order -- so if
            // Eigen<T>::make() is a deterministic function of its input,
            // vecs.x and vecs.z should be bit-identical too. vecs.x/vecs.z
            // are SquareMat<Real>, which (unlike Vec<T> above) is already
            // confirmed to support .get(hostPtr,hand) directly (used
            // throughout checkEigens), so no SquareMat::diag() workaround is
            // needed here -- call .get() straight on the member expression,
            // with no intermediate copy of the member itself.
            {
                size_t n = ed.lapEigen.vecs.x._rows;
                ASSERT_EQ(n, ed.lapEigen.vecs.z._rows) << "test assumes vecs.x and vecs.z are the same size";
                ASSERT_EQ(ed.lapEigen.vecs.x._cols, ed.lapEigen.vecs.z._cols) << "test assumes vecs.x and vecs.z are the same size";

                std::vector<Real> VxHost(ed.lapEigen.vecs.x.size()), VzHost(ed.lapEigen.vecs.z.size());
                ed.lapEigen.vecs.x.get(VxHost.data(), hands[0]);
                ed.lapEigen.vecs.z.get(VzHost.data(), hands[0]);
                cudaDeviceSynchronize();

                Real maxVecDiff = 0;
                size_t worstVecI = 0, worstVecJ = 0;
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        Real d = relDiff(VxHost[i * n + j], VzHost[i * n + j]);
                        if (d > maxVecDiff) { maxVecDiff = d; worstVecI = i; worstVecJ = j; }
                    }
                }
                std::cout << "[VyRealMeshUniformRhsStaysSymmetric] eigenvector comparison "
                          << "(lapEigen.vecs.x vs lapEigen.vecs.z, " << n << "x" << n << "): "
                          << "max relative diff = " << maxVecDiff << " at (" << worstVecI << "," << worstVecJ << ")\n";
                EXPECT_LT(maxVecDiff, 1e-6)
                    << "L_x == L_z exactly and their eigenvalues match, but Eigen<T>::make() produces "
                    << "different eigenvectors for X and Z -- max relative diff " << maxVecDiff << " at ("
                    << worstVecI << "," << worstVecJ << ")";
            }

            // ----------------------------------------------------------------
            // Dense per-axis operator comparison, direct AND transposed.
            // vals.x/vals.z matching exactly (relDiff=0, just shown above)
            // does NOT prove L_x == L_z as matrices: a matrix and its
            // TRANSPOSE always share the same eigenvalues (same
            // characteristic polynomial) while generally having different
            // eigenvectors and producing different solutions to Lv=b. So an
            // exact eigenvalue match combined with the massively broken
            // exchange-symmetric solve confirmed above is exactly the
            // signature a transpose-type discrepancy between L_x and L_z
            // would produce -- e.g. the FluxLapl stencil's off-diagonal
            // "left"/"right" coupling built in opposite traversal order for
            // one axis vs the other. Laplacian1d::dense(axisIdx,...) is
            // already used, unmodified, by the existing
            // boundaryBattery/checkEigens tests above (axis 0=X, 1=Y, 2=Z),
            // so this reuses a proven call shape rather than introducing a
            // new one.
            {
                Laplacian1d<Real> laplacian1dXZ(boundaryHost.forDevice(), hands[0]);
                auto Lx = laplacian1dXZ.dense(0, hands[0]);
                auto Lz = laplacian1dXZ.dense(2, hands[0]);
                ASSERT_EQ(Lx._rows, Lz._rows) << "test assumes X and Z dense operators are the same size";
                ASSERT_EQ(Lx._cols, Lz._cols) << "test assumes X and Z dense operators are the same size";

                size_t n = Lx._rows;
                std::vector<Real> LxHost(Lx.size()), LzHost(Lz.size());
                Lx.get(LxHost.data(), hands[0]);
                Lz.get(LzHost.data(), hands[0]);
                cudaDeviceSynchronize();

                // Check both a direct elementwise match and a transposed
                // match; report both, so the message itself tells us which
                // hypothesis fits regardless of which one trips the assert.
                Real maxDirectDiff = 0, maxTransposeDiff = 0;
                size_t worstDirectI = 0, worstDirectJ = 0, worstTransI = 0, worstTransJ = 0;
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        Real lxij = LxHost[i * n + j];
                        Real lzij = LzHost[i * n + j];
                        Real lzji = LzHost[j * n + i];

                        Real dDirect = relDiff(lxij, lzij);
                        if (dDirect > maxDirectDiff) { maxDirectDiff = dDirect; worstDirectI = i; worstDirectJ = j; }

                        Real dTrans = relDiff(lxij, lzji);
                        if (dTrans > maxTransposeDiff) { maxTransposeDiff = dTrans; worstTransI = i; worstTransJ = j; }
                    }
                }
                std::cout << "[VyRealMeshUniformRhsStaysSymmetric] dense 1D operator comparison (X=dense(0) vs "
                          << "Z=dense(2), " << n << "x" << n << "): direct elementwise max relative diff = "
                          << maxDirectDiff << " at (" << worstDirectI << "," << worstDirectJ << "); "
                          << "transposed (Lx[i][j] vs Lz[j][i]) max relative diff = " << maxTransposeDiff
                          << " at (" << worstTransI << "," << worstTransJ << ")\n";
                EXPECT_LT(maxDirectDiff, 1e-6)
                    << "Laplacian1d builds different dense 1D operators for X and Z despite identical input "
                    << "data -- direct elementwise max relative diff " << maxDirectDiff << " at ("
                    << worstDirectI << "," << worstDirectJ << "); transposed max relative diff was "
                    << maxTransposeDiff << " (a small transposed diff would mean L_x is Z's operator, transposed)";
            }

            // ----------------------------------------------------------------
            // Universal eigenpair check (does L*V == V*Lambda?), run directly
            // on THIS real, asymmetric config via checkEigens itself (defined
            // above, already proven correct elsewhere in this file via
            // boundaryBattery) instead of a hand-rolled host-side matmul.
            //
            // Two prior attempts at this check reinvented the matmul on the
            // host, reading SquareMat::get()'s flat buffer first as row-major
            // then as column-major. BOTH produced widespread, mutually
            // inconsistent violations (8760/10201 and 1592/10201 entries
            // respectively) with neither reading clean -- two different
            // guessed layouts both failing is itself a strong sign the bug
            // was in that hand-rolled matmul, not in the solver. checkEigens
            // sidesteps the question entirely: it multiplies via
            // SquareMat::mult() internally, so whatever SquareMat's actual
            // layout is, it's handled correctly by the library's own code --
            // the same code already proven correct for boundaryBattery's
            // passing tests. The one real defect in calling it directly is
            // its default tolerance (1e-8, absolute) being meaningless at the
            // ~1e10 eigenvalue scale seen here (sortedX=-3.31078e+10, logged
            // above); passing an explicit, scale-appropriate tolerance fixes
            // that without touching checkEigens itself. Its ASSERT_NEAR loop
            // stops at the first failing entry rather than reporting the
            // worst one -- acceptable here since the question is binary
            // (does the real eigendecomposition satisfy this identity at
            // all?), not which entry is worst.
            {
                Laplacian1d<Real> laplacian1dCheck(boundaryHost.forDevice(), hands[0]);
                auto LxCheck = laplacian1dCheck.dense(0, hands[0]);
                auto LzCheck = laplacian1dCheck.dense(2, hands[0]);

                // Eigenvalues here run ~1e10 in magnitude (sortedX above), so
                // L*V and V*Lambda entries are plausibly ~1e9-1e10; 1e4 is
                // roughly 1e-6 relative at that scale -- far tighter than the
                // ~1e9-1e10-scale error the observed bug would produce if it
                // were present here too, while not spuriously tripping on
                // ordinary double-precision rounding at this magnitude.
                Real eigenpairTol = Real(1e4);
                checkEigens(LxCheck, ed.lapEigen.vecs.x, ed.lapEigen.vals.x, hands[0],
                            "Vy real X-axis universal eigenpair check (L*V=V*Lambda)", false, eigenpairTol);
                checkEigens(LzCheck, ed.lapEigen.vecs.z, ed.lapEigen.vals.z, hands[0],
                            "Vy real Z-axis universal eigenpair check (L*V=V*Lambda)", false, eigenpairTol);
                std::cout << "[VyRealMeshUniformRhsStaysSymmetric] universal eigenpair check via checkEigens "
                          << "(trusted SquareMat::mult(), not a hand-rolled host matmul) completed for both "
                          << "X and Z at absolute tolerance " << eigenpairTol << " -- any ASSERT_NEAR failure "
                          << "this triggered appears above with its own flat index, row and column.\n";
            }
        });
}

// ============================================================================
// Size x shift bisection. The N=100, shift=6.45e7 (Vy's real config) case
// above originally reported a large residual via a check that omitted the
// shift term (fixed now -- see the comment on that test). At shift=0 that
// omission is a no-op (the missing term is -0*x), so the shift=0 numbers
// below are unaffected either way; only the shift!=0 corners needed the fix.
// Cross-referenced against AlexCudaCompatibility.f90, which uses this exact
// thomas=true/shift=shiftVelocity/FluxLapl-VariableDeltaLapl-FluxLapl
// combination for Vy and is stable, so the shared solver code is not what
// we're hunting for -- this scan is about whether grid size N changes
// anything, not about re-litigating solve() itself. The synthetic
// VxShapedAsymmetricGrid/VyShapedAsymmetricGrid tests pass at small size
// (5-11 per axis) with shift=0; FluxLaplacian_SmallScale passes at small
// size (11) WITH a large shift (3.48568e6, uniform deltas on all three
// axes). This test computes the relative residual of (L-shift*I)*x_solved
// vs rhs at all four corners of N=10/100 x shift=0/real, in one run.
// ============================================================================

template<typename Real>
Real vyShapedRelativeResidual(int N, Real shiftVelocity, Handles& hands) {
    const Real stretchFactor = Real(10.0);
    AxisSpacingForTest<Real> xAxis = computeAxisSpacingForTest<Real>(N, Real(1.0), stretchFactor);
    AxisSpacingForTest<Real> yAxis = computeAxisSpacingForTest<Real>(N, Real(1.0), stretchFactor);
    AxisSpacingForTest<Real> zAxis = computeAxisSpacingForTest<Real>(N, Real(1.0), stretchFactor);

    GridDim dim(/*rows=*/N, /*cols=*/N + 1, /*layers=*/N + 1);
    XYZ<std::vector<Real>> deltas(xAxis.hHalf, yAxis.hFull, zAxis.hHalf);
    XYZ<eigen::LaplOperatorT> kind(eigen::FluxLapl, eigen::VariableDeltaLapl, eigen::FluxLapl);
    XYZ<bool> allDirichlet(false, false, false);
    XYZ<Real> zeroR3(0.0, 0.0, 0.0);

    Real relResidual = -1;
    buildBoundaryConfigAndLaunch<Real>(dim, deltas, allDirichlet, allDirichlet, zeroR3, zeroR3, kind, hands[0],
        [&](const auto& boundaryHost) {
            EigenDecompThomas ed(boundaryHost, hands, shiftVelocity);

            auto rhs = SimpleArray<Real>::create(dim.size(), hands[0]);
            rhs.fill(Real(1.0), hands[0]);

            auto x = SimpleArray<Real>::create(dim.size(), hands[0]);
            x.fill(0, hands[0]);
            ed.solve(x, rhs, hands[0]);

            auto laplacian = poisson::laplacian<Real>(boundaryHost.forDevice(), hands[0]);
            auto residual = SimpleArray<Real>::create(dim.size(), hands[0]);
            residual.fill(Real(1.0), hands[0]);
            laplacian.bandedMult(x, residual, &hands[0], GPUScalar<Real>::get(1, hands[0]), GPUScalar<Real>::get(-1, hands[0]), false);
            // residual now holds L*x - rhs; subtract shift*x to check what
            // solve() actually promises: (L - shift*I)*x - rhs.
            auto negShift = Singleton<Real>::create(-shiftVelocity, hands[0]);
            residual.add(x, &negShift, &hands[0]);

            auto normDevice = x.get(0);
            Real normHost[1];
            residual.norm(normDevice, hands[0]);
            normDevice.get(normHost, hands[0]);
            cudaDeviceSynchronize();

            Real rhsNorm = std::sqrt(Real(dim.size()));
            relResidual = normHost[0] / rhsNorm;
        });
    return relResidual;
}

TEST(ThomasAxisAssignment, VyShapedSizeShiftBisection) {
    Handles hands(3);
    using Real = double;

    const Real shiftVelocity = Real(1.5) / (Real(1e-4) * (Real(1) / std::sqrt(Real(1.85185185e7))));

    Real residSmallNoShift = vyShapedRelativeResidual<Real>(10, Real(0), hands);
    Real residSmallShift   = vyShapedRelativeResidual<Real>(10, shiftVelocity, hands);
    Real residLargeNoShift = vyShapedRelativeResidual<Real>(100, Real(0), hands);
    Real residLargeShift   = vyShapedRelativeResidual<Real>(100, shiftVelocity, hands);

    std::cout << "[VyShapedSizeShiftBisection] relative residuals:\n"
              << "  N=10,  shift=0              : " << residSmallNoShift << "\n"
              << "  N=10,  shift=" << shiftVelocity << " : " << residSmallShift << "\n"
              << "  N=100, shift=0              : " << residLargeNoShift << "\n"
              << "  N=100, shift=" << shiftVelocity << " : " << residLargeShift << "\n";

    EXPECT_LT(residSmallNoShift, 1e-6) << "N=10, shift=0 residual";
    EXPECT_LT(residSmallShift, 1e-6) << "N=10, shift=" << shiftVelocity << " residual";
    EXPECT_LT(residLargeNoShift, 1e-6) << "N=100, shift=0 residual";
    EXPECT_LT(residLargeShift, 1e-6) << "N=100, shift=" << shiftVelocity << " residual";
}

// ============================================================================
// Y/Z-role interchangeability check.
//
// Reading FortranBindings.hpp's initEigenDecomp_d (the entry point Fortran's
// AlexCudaCompatibility.f90 calls for Vy) line by line shows it does NOT
// forward dim1/dim2/dim3 straight through to initEigenDecompSolver's
// rows/cols/layers (and x/y/z delta) slots -- it reorders them:
//
//   initEigenDecompSolver<double>(
//       dim1Length, dim3Length, dim2Length,       // -> rows, cols, layers
//       dim3Delta,  dim1Delta,  dim2Delta,        // -> xDelta, yDelta, zDelta
//       dim3SegType, dim1SegType, dim2SegType,    // -> xSeg, ySeg, zSeg
//       ... dim3*Neumann/Val, dim1*Neumann/Val, dim2*Neumann/Val ...)
//
// i.e. GPU-role-X <- Fortran's dim3, GPU-role-Y <- dim1, GPU-role-Z <- dim2:
// a 3-cycle, not an identity map. AlexCudaCompatibility.f90's real Vy call
// (lines 130-149) passes dim1=HPx/FluxLapl (physical X), dim2=Hy12/
// VariableDeltaLapl (physical Y), dim3=HPz/FluxLapl (physical Z) -- its own
// natural dim1/dim2/dim3 = X/Y/Z storage order. Pushed through the 3-cycle,
// Fortran's Vy solver actually receives: GPU-X-role (Thomas-swept) <- HPz
// (physical Z), GPU-Y-role (eigen-transformed) <- HPx (physical X),
// GPU-Z-role (eigen-transformed) <- Hy12 (physical Y, the VariableDeltaLapl
// axis).
//
// GpuSolverSetup.cu's buildFieldSolver, by contrast, calls EigenDecompThomas
// directly -- no FortranBindings.hpp, no permutation -- with
// GridDim(ny, nx1, nz1) and deltaForVy(hostHpx(), hostHy12(), hostHpz()):
// GPU-X-role (Thomas) <- HPx (physical X), GPU-Y-role (eigen) <- Hy12
// (physical Y, VariableDeltaLapl), GPU-Z-role (eigen) <- HPz (physical Z).
//
// For a square duct, physical X and physical Z meshes are identical, so
// which of the two FluxLapl arrays lands on the Thomas axis is numerically
// inert either way. What's NOT obviously inert, if the solver treats its
// Y-role and Z-role asymmetrically in any way, is that Fortran's effective
// construction puts the VariableDeltaLapl axis on Z-role while C++'s puts it
// on Y-role -- the two paths would then be solving genuinely different
// linear systems, not relabelled copies of the same one. This test checks
// exactly that, standalone (synthetic hand-written delta arrays, no
// dependency on the real mesh): build one solver with (X=FluxLapl A,
// Y=VariableDeltaLapl B, Z=FluxLapl C) -- GpuSolverSetup.cu's own pattern --
// and a second with B and C swapped onto the other role, X/A untouched --
// Fortran's effective pattern -- then check whether the second solver's
// solution, read back with its Y and Z arguments swapped, reproduces the
// first solver's solution exactly at every corresponding physical point. If
// EigenDecompThomas's Y-role and Z-role are truly interchangeable, it must.
// ----------------------------------------------------------------------
TEST(ThomasAxisAssignment, VyRealMeshYZRoleSwapMatchesTranspose) {
    Handles hands(3);
    using Real = double;

    // Deliberately distinct axis lengths so a role mixup can't hide behind
    // coincidentally equal sizes. Synthetic data -- this test only checks
    // whether EigenDecompThomas's Y-role and Z-role are interchangeable, so
    // it doesn't need the real mesh.
    const int nx1 = 12, nyLen = 9, nzLen = 15;

    // Every FluxLapl/VariableDeltaLapl axis needs axisLength+1 delta values
    // (confirmed directly from FortranBindings.hpp's initEigenDecompSolver:
    // hasVariableDelta(...) is true for both kinds, and it sizes each delta
    // vector to cols+1 / rows+1 / layers+1 accordingly).
    std::vector<Real> A(nx1 + 1), B(nyLen + 1), C(nzLen + 1);
    for (int i = 0; i <= nx1; ++i) A[i] = Real(1) + Real(0.05) * i;
    for (int i = 0; i <= nyLen; ++i) B[i] = Real(0.5) + Real(0.07) * i + Real(0.01) * i * i;
    for (int i = 0; i <= nzLen; ++i) C[i] = Real(2) + Real(0.03) * i - Real(0.002) * i * i;

    // Vy's real Hartmann=1000 shift magnitude; only roughly matters here.
    const Real shiftVelocity = Real(1.5) / (Real(1e-4) * (Real(1) / std::sqrt(Real(1.85185185e7))));
    XYZ<bool> allDirichlet(false, false, false);
    XYZ<Real> zeroR3(0.0, 0.0, 0.0);

    GridDim dimOriginal(/*rows=*/nyLen, /*cols=*/nx1, /*layers=*/nzLen);
    GridDim dimSwapped(/*rows=*/nzLen, /*cols=*/nx1, /*layers=*/nyLen);

    std::vector<Real> xOriginalHost(dimOriginal.size()), xSwappedHost(dimSwapped.size());

    XYZ<std::vector<Real>> deltasOriginal(A, B, C);
    XYZ<eigen::LaplOperatorT> kindOriginal(eigen::FluxLapl, eigen::VariableDeltaLapl, eigen::FluxLapl);
    buildBoundaryConfigAndLaunch<Real>(dimOriginal, deltasOriginal, allDirichlet, allDirichlet, zeroR3, zeroR3, kindOriginal, hands[0],
        [&](const auto& boundaryHost) {
            EigenDecompThomas ed(boundaryHost, hands, shiftVelocity);

            auto rhs = SimpleArray<Real>::create(dimOriginal.size(), hands[0]);
            rhs.fill(1.0, hands[0]);
            auto x = SimpleArray<Real>::create(dimOriginal.size(), hands[0]);
            x.fill(0, hands[0]);
            ed.solve(x, rhs, hands[0]);

            x.get(xOriginalHost.data(), hands[0]);
            cudaDeviceSynchronize();
        });

    XYZ<std::vector<Real>> deltasSwapped(A, C, B);
    XYZ<eigen::LaplOperatorT> kindSwapped(eigen::FluxLapl, eigen::FluxLapl, eigen::VariableDeltaLapl);
    buildBoundaryConfigAndLaunch<Real>(dimSwapped, deltasSwapped, allDirichlet, allDirichlet, zeroR3, zeroR3, kindSwapped, hands[0],
        [&](const auto& boundaryHost) {
            EigenDecompThomas ed(boundaryHost, hands, shiftVelocity);

            auto rhs = SimpleArray<Real>::create(dimSwapped.size(), hands[0]);
            rhs.fill(1.0, hands[0]);
            auto x = SimpleArray<Real>::create(dimSwapped.size(), hands[0]);
            x.fill(0, hands[0]);
            ed.solve(x, rhs, hands[0]);

            x.get(xSwappedHost.data(), hands[0]);
            cudaDeviceSynchronize();
        });

    for (Real v : xOriginalHost) ASSERT_TRUE(std::isfinite(v)) << "original construction produced a non-finite value -- not a valid baseline.";
    for (Real v : xSwappedHost)  ASSERT_TRUE(std::isfinite(v)) << "swapped construction produced a non-finite value -- not a valid baseline.";

    auto atOriginal = [&](int xi, int yi, int zi) -> Real {
        size_t idx = (size_t)(xi - 1) + (size_t)dimOriginal.cols * ((size_t)(yi - 1) + (size_t)dimOriginal.rows * (size_t)(zi - 1));
        return xOriginalHost[idx];
    };
    // Swapped's 2nd argument is its Y-role index (native C/Z-index in
    // "original" terms), 3rd argument is its Z-role index (native B/Y-index).
    auto atSwapped = [&](int xi, int yRoleIdx, int zRoleIdx) -> Real {
        size_t idx = (size_t)(xi - 1) + (size_t)dimSwapped.cols * ((size_t)(yRoleIdx - 1) + (size_t)dimSwapped.rows * (size_t)(zRoleIdx - 1));
        return xSwappedHost[idx];
    };

    auto relDiff = [](Real a, Real b) {
        Real absA = std::abs(a), absB = std::abs(b);
        Real scale = absA > absB ? absA : absB;
        if (scale < Real(1e-300)) scale = Real(1e-300);
        return std::abs(a - b) / scale;
    };

    Real maxRelDiff = 0;
    int worstXi = 0, worstYi = 0, worstZi = 0;
    size_t violations = 0, total = 0;
    for (int zi = 1; zi <= nzLen; ++zi) {
        for (int yi = 1; yi <= nyLen; ++yi) {
            for (int xi = 1; xi <= nx1; ++xi) {
                ++total;
                Real original = atOriginal(xi, yi, zi);
                Real swapped = atSwapped(xi, /*Y-role=C/Z-native-index=*/ zi, /*Z-role=B/Y-native-index=*/ yi);
                Real d = relDiff(original, swapped);
                if (d >= 1e-6) ++violations;
                if (d > maxRelDiff) { maxRelDiff = d; worstXi = xi; worstYi = yi; worstZi = zi; }
            }
        }
    }
    std::cout << "[VyRealMeshYZRoleSwapMatchesTranspose] " << violations << "/" << total
              << " points violate 1e-6 after swapping which array (VariableDeltaLapl B vs FluxLapl C) "
              << "occupies the Y-role vs the Z-role; worst relative diff = " << maxRelDiff
              << " at physical point (xi=" << worstXi << ",yi=" << worstYi << ",zi=" << worstZi << ")\n";
    EXPECT_LT(maxRelDiff, 1e-6)
        << violations << "/" << total << " points differ by more than 1e-6 between the Y=B/Z=C "
        << "construction and the Y=C/Z=B construction, read back at the corresponding physical point. "
        << "EigenDecompThomas's Y-role and Z-role are NOT interchangeable -- which would mean Fortran's "
        << "init_eigen_decomp_d permutation (which swaps exactly this pair for Vy, see comment above) is "
        << "not a harmless relabeling: Fortran and C++ end up solving genuinely different linear systems "
        << "for what's meant to be the same physical Vy field.";
}

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
