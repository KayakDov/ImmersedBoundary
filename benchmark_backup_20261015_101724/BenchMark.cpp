#include <gtest/gtest.h>

#include "solvers/EigenDecomp/EigenDecomp2d.h"
#include "solvers/EigenDecomp/EigenDecompThomas.cuh"
#include "poisson/Poisson.cuh"
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"
#include <cmath>
#include <random>

#include "deviceArrays/headers/DeviceMemory.h"
#include "immersedBoundary/ImerssedEquation.h"
#include "poisson/Laplacian1d.cuh"
#include "poisson/BoundaryConfig.cuh"
#include "poisson/AxisSegment.cuh"
#include "poisson/BoundaryCondition.cuh"

// ─── Types ───────────────────────────────────────────────────────────────────

struct SolverTiming {
    double eigenMs      = -1;
    double thomasUnifMs = -1;
    double thomasVarMs  = -1;
    double diffNormUnif = -1;
    double diffNormVar  = -1;
};

template <typename Real>
class SolverBuffers {
public:
    size_t allocatedTotal = 0;
    size_t allocatedN     = 0;

    std::optional<SimpleArray<Real>> rhs, xEigen, xThomas, xVarThomas, sizeOfB;
    std::optional<SimpleArray<Real>> dx, dy, dz;
    std::optional<Mat<Real>>         sizeOfBX3;
    std::optional<Singleton<Real>>   normResult;

    bool isLargeEnough(size_t total) const {
        return rhs.has_value() && total <= allocatedTotal;
    }

    void deallocate() {
        rhs.reset();
        xEigen.reset();
        xThomas.reset();
        xVarThomas.reset();
        dx.reset();
        dy.reset();
        dz.reset();
        sizeOfB.reset();
        sizeOfBX3.reset();
        normResult.reset();
    }

    void allocate(size_t total, size_t maxN, Handle& hand) {
        deallocate();

        rhs        .emplace(SimpleArray<Real>::create(total,    hand));
        xEigen     .emplace(SimpleArray<Real>::create(total,    hand));
        xThomas    .emplace(SimpleArray<Real>::create(total,    hand));
        xVarThomas .emplace(SimpleArray<Real>::create(total,    hand));
        dx         .emplace(SimpleArray<Real>::create(maxN + 2, hand));
        dy         .emplace(SimpleArray<Real>::create(maxN + 2, hand));
        dz         .emplace(SimpleArray<Real>::create(maxN + 2, hand));
        sizeOfB    .emplace(SimpleArray<Real>::create(total,    hand));
        sizeOfBX3  .emplace(Mat<Real>::create(total, 3, hand));
        normResult .emplace(Singleton<Real>::create(hand));

        allocatedTotal = total;
        allocatedN     = maxN;
    }

    // Tries to allocate ahead-of-need; falls back to exact size; returns false on OOM.
    bool growToFit(size_t n, size_t total, size_t reallocEvery, Handle& hand) {
        if (isLargeEnough(total))
            return true;

        const size_t nAhead     = n + reallocEvery * 2;
        const size_t aheadTotal = std::min(nAhead * nAhead * nAhead, total * 2);

        try {
            allocate(aheadTotal, nAhead, hand);
            return true;
        } catch (...) {}

        try {
            allocate(total, n, hand);
            return true;
        } catch (...) {
            std::cout << "OOM allocating buffers at N=" << n << std::endl;
            return false;
        }
    }
};

// ─── Boundary Config Factories ───────────────────────────────────────────────

template <typename Real>
auto makeUniformNeumann(const Real3d& delta, const GridDim& dim) {
    return makeUniformBoundaryConfigHost<Real>(
        {false, false, false},
        {false, false, false},
        {0.0,   0.0,   0.0},
        {0.0,   0.0,   0.0},
        delta, dim, /*isStaggered=*/false
    );
}

template <typename Real>
auto makeVariableUnitSpacing(SolverBuffers<Real>& buf, size_t n, Handle& hand) {
    auto dxSub = buf.dx->subArray(0, n + 1); dxSub.fill(1.0, hand);
    auto dySub = buf.dy->subArray(0, n + 1); dySub.fill(1.0, hand);
    auto dzSub = buf.dz->subArray(0, n + 1); dzSub.fill(1.0, hand);

    AxisSegmentHost<VariableSegment<Real>> varX({0, true}, {0, true}, dxSub);
    AxisSegmentHost<VariableSegment<Real>> varY({0, true}, {0, true}, dySub);
    AxisSegmentHost<VariableSegment<Real>> varZ({0, true}, {0, true}, dzSub);

    return BoundaryConfigHost<Real,
                          VariableSegment<Real>,
                          VariableSegment<Real>,
                          VariableSegment<Real>>(varX, varY, varZ);
}

// ─── RHS Generation ──────────────────────────────────────────────────────────

template <typename Real>
std::vector<Real> generateZeroMeanRhs(size_t total) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<Real> dist(-1, 1);

    std::vector<Real> rhs(total);
    Real sum = Real(0);
    for (auto& v : rhs) { v = dist(rng); sum += v; }

    const Real mean = sum / Real(total);
    for (auto& v : rhs) v -= mean;

    return rhs;
}

// ─── Residual Norm ───────────────────────────────────────────────────────────

template <typename Real, typename segX, typename segY, typename segZ>
double computeResidualNorm(const BoundaryConfig<Real, segX, segY, segZ>& boundary, const SimpleArray<Real>& x, const SimpleArray<Real>& rhs, Singleton<Real>& normResult, Handle* hands) {

    auto L          = poisson::laplacian<Real>(boundary, hands[0]);
    auto lxMinusRhs = Vec<Real>::create(x.size(), hands[0]);

    L.bandedMult(x, lxMinusRhs, hands, GPUScalar<Real>::get(1, hands[0]), GPUScalar<Real>::get(0, hands[0]), false);
    lxMinusRhs.add(rhs, &GPUScalar<Real>::get(-1, hands[0]), hands);
    lxMinusRhs.norm(normResult, hands[0]);
    cudaDeviceSynchronize();

    return normResult.get(hands[0]);
}

// ─── Individual Solver Runs ───────────────────────────────────────────────────

template <typename Real, typename segX, typename  segY, typename segZ>
double runEigenDecomp3d(const BoundaryConfig<Real, segX, segY, segZ>& boundary,
                        SimpleArray<Real>& x,
                        const SimpleArray<Real>& rhs,
                        SimpleArray<Real>& sizeOfB,
                        Handle* hands, Event* events) {
    x.fill(0, hands[0]);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    EigenDecomp3d<Real> ed(boundary, hands, events, sizeOfB);
    ed.solve(x, rhs, hands[0]);
    cudaDeviceSynchronize();

    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

template <typename Real, typename SegX, typename SegY, typename SegZ>
double runEigenDecompThomas(const BoundaryConfigHost<Real, SegX, SegY, SegZ>& boundary,
                            SimpleArray<Real>& x,
                            const SimpleArray<Real>& rhs,
                            Mat<Real>& sizeOfBX3,
                            Handle* hands, Event* events) {
    x.fill(0, hands[0]);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    EigenDecompThomas<Real, SegX> ed(boundary, hands, events, sizeOfBX3);
    ed.solve(x, rhs, hands[0]);
    cudaDeviceSynchronize();

    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}
// ─── Output Formatting ───────────────────────────────────────────────────────

void printBenchmarkHeader() {
    std::cout << DeviceMemory() << std::endl;
    std::cout << std::setw(12) << "Grid dim N"
              << std::setw(15) << "N x N x N"
              << std::setw(18) << "Eigen3d (ms)"
              << std::setw(18) << "Thomas Unif (ms)"
              << std::setw(22) << "||L*x_unif - b||"
              << std::setw(18) << "Thomas Var (ms)"
              << std::setw(22) << "||L*x_var - b||"
              << std::endl;
    std::cout << std::string(125, '-') << std::endl;
}

void printBenchmarkRow(size_t n, size_t total, const SolverTiming& t) {
    std::cout << std::setw(12) << n
              << std::setw(15) << total
              << std::setw(18) << std::fixed      << std::setprecision(3) << t.eigenMs
              << std::setw(18) << std::fixed      << std::setprecision(3) << t.thomasUnifMs
              << std::setw(22) << std::scientific << std::setprecision(3) << t.diffNormUnif
              << std::setw(18) << std::fixed      << std::setprecision(3) << t.thomasVarMs
              << std::setw(22) << std::scientific << std::setprecision(3) << t.diffNormVar
              << std::endl;
}

// ─── Test ────────────────────────────────────────────────────────────────────

TEST(Benchmark, SolverRuntimes) {
    using Real = double;

    Handle hands[3];
    Event  events[2];
    Real3d delta(1, 1, 1);

    printBenchmarkHeader();

    const size_t reallocEvery = 100;
    SolverBuffers<Real> buf;

    for (size_t n = 2; ; n += (n < 200 ? 2 : 1)) {
        const size_t total = n * n * n;
        const GridDim dim(n, n, n);

        if (!buf.growToFit(n, total, reallocEvery, hands[0]))
            break;

        const auto uniformBoundary  = makeUniformNeumann<Real>(delta, dim);
        const auto variableBoundary = makeVariableUnitSpacing<Real>(buf, n, hands[0]);

        auto rhsSub        = buf.rhs       ->subArray(0, total);
        auto xEigenSub     = buf.xEigen    ->subArray(0, total);
        auto xThomasSub    = buf.xThomas   ->subArray(0, total);
        auto xVarThomasSub = buf.xVarThomas->subArray(0, total);
        auto sizeOfBSub    = buf.sizeOfB   ->subArray(0, total);
        auto sizeOfBX3Sub  = buf.sizeOfBX3 ->subMat(0, 0, total, 3);

        const auto rhsHost = generateZeroMeanRhs<Real>(total);
        rhsSub.set(rhsHost.data(), hands[0]);

        SolverTiming timing;

        try {
            timing.eigenMs = runEigenDecomp3d<Real>(uniformBoundary.forDevice(), xEigenSub, rhsSub, sizeOfBSub, hands, events);
        } catch (...) {
            std::cout << "OOM in EigenDecomp3d at N=" << n << std::endl; break;
        }

        try {
            timing.thomasUnifMs = runEigenDecompThomas<Real, UniformSegment<Real>>(uniformBoundary, xThomasSub, rhsSub, sizeOfBX3Sub, hands, events);
            timing.diffNormUnif = computeResidualNorm(uniformBoundary.forDevice(), xThomasSub, rhsSub, *buf.normResult, hands);
        } catch (...) {
            std::cout << "OOM in EigenDecompThomas (Uniform) at N=" << n << std::endl; break;
        }

        try {
            timing.thomasVarMs = runEigenDecompThomas<Real, VariableSegment<Real>>(
                variableBoundary, xVarThomasSub, rhsSub, sizeOfBX3Sub, hands, events);
            timing.diffNormVar = computeResidualNorm(
                variableBoundary.forDevice(), xVarThomasSub, rhsSub, *buf.normResult, hands);
        } catch (...) {
            std::cout << "OOM in EigenDecompThomas (Variable) at N=" << n << std::endl; break;
        }

        printBenchmarkRow(n, total, timing);
    }
}