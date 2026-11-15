#include <gtest/gtest.h>

#include "solvers/EigenDecomp/EigenDecomp2d.h"
#include "solvers/EigenDecomp/EigenDecompThomas.cuh"
#include "poisson/Poisson.cuh"
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <optional>
#include <cuda_runtime.h>

#include "deviceArrays/headers/DeviceMemory.h"
#include "immersedBoundary/ImmersedEquation.h"
#include "deviceArrays/headers/Handles.h"
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
double computeResidualNorm(const BoundaryConfig<Real, segX, segY, segZ>& boundary, const SimpleArray<Real>& x, const SimpleArray<Real>& rhs, Singleton<Real>& normResult, Handles& hands) {

    auto L          = poisson::laplacian<Real>(boundary, hands[0]);
    auto lxMinusRhs = Vec<Real>::create(x.size(), hands[0]);

    L.bandedMult(x, lxMinusRhs, &hands[0], GPUScalar<Real>::get(1, hands[0]), GPUScalar<Real>::get(0, hands[0]), false);
    lxMinusRhs.add(rhs, &GPUScalar<Real>::get(-1, hands[0]), &hands[0]);
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
                        Handles& hands) {
    x.fill(0, hands[0]);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    EigenDecomp3d<Real> ed(boundary, hands, sizeOfB);
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
                            Handles& hands) {
    x.fill(0, hands[0]);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    EigenDecompThomas<Real, SegZ> ed(boundary, hands, sizeOfBX3);
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

/**
 * Times the three solvers on cubic grids of growing size.
 *
 * The loop over n has no upper bound on purpose: the benchmark runs until the GPU runs
 * out of memory. That is the goal, since it finds the largest grid each solver handles.
 * Running out of memory ends the run cleanly (growToFit returns false, or a solver
 * throws and the catch block reports the size and stops), so this is not a failure.
 */
TEST(Benchmark, SolverRuntimes) {
    using Real = double;

    Handles hands(3);
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
            timing.eigenMs = runEigenDecomp3d<Real>(uniformBoundary.forDevice(), xEigenSub, rhsSub, sizeOfBSub, hands);
        } catch (...) {
            std::cout << "OOM in EigenDecomp3d at N=" << n << std::endl; break;
        }

        try {
            timing.thomasUnifMs = runEigenDecompThomas<Real, UniformSegment<Real>>(uniformBoundary, xThomasSub, rhsSub, sizeOfBX3Sub, hands);
            timing.diffNormUnif = computeResidualNorm(uniformBoundary.forDevice(), xThomasSub, rhsSub, *buf.normResult, hands);
        } catch (...) {
            std::cout << "OOM in EigenDecompThomas (Uniform) at N=" << n << std::endl; break;
        }

        try {
            timing.thomasVarMs = runEigenDecompThomas<Real, VariableSegment<Real>>(
                variableBoundary, xVarThomasSub, rhsSub, sizeOfBX3Sub, hands);
            timing.diffNormVar = computeResidualNorm(
                variableBoundary.forDevice(), xVarThomasSub, rhsSub, *buf.normResult, hands);
        } catch (...) {
            std::cout << "OOM in EigenDecompThomas (Variable) at N=" << n << std::endl; break;
        }

        printBenchmarkRow(n, total, timing);
    }
}

// ─── Thomas-only benchmark ───────────────────────────────────────────────────

/**
 * @brief Times of one Thomas solver at one grid size.
 */
struct ThomasTiming {
    double buildAndSolveMs = -1; ///< Construction plus one solve, measured the way SolverRuntimes measures it.
    double solveMs         = -1; ///< Average time of one solve on the already-built solver.
};

/**
 * @brief Builds a Thomas solver and times it two ways.
 *
 * buildAndSolveMs is measured exactly as runEigenDecompThomas measures it, so it can be pasted next
 * to earlier SolverRuntimes results. solveMs leaves construction out: in the CFD each solver is
 * built once and then solved every time step, so that is the number that matters there. The solve
 * inside buildAndSolveMs doubles as the warm-up for the timed solves.
 *
 * @param boundary  The boundary conditions.
 * @param x         Receives the solution; N^3 values.
 * @param rhs       The right-hand side; N^3 values, not modified.
 * @param sizeOfBX3 Solver workspace: N^3 rows, 3 columns.
 * @param hands     At least 3 handles; the solves run on hands[0].
 * @param repeats   Number of timed solves averaged into solveMs; at least 1.
 * @return Both times, in milliseconds.
 */
template <typename Real, typename SegX, typename SegY, typename SegZ>
ThomasTiming timeThomas(const BoundaryConfigHost<Real, SegX, SegY, SegZ>& boundary,
                        SimpleArray<Real>& x,
                        const SimpleArray<Real>& rhs,
                        Mat<Real>& sizeOfBX3,
                        Handles& hands,
                        size_t repeats) {
    ThomasTiming timing;
    x.fill(0, hands[0]);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    EigenDecompThomas<Real, SegZ> ed(boundary, hands, sizeOfBX3);
    ed.solve(x, rhs, hands[0]);
    cudaDeviceSynchronize();
    timing.buildAndSolveMs = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repeats; ++i) ed.solve(x, rhs, hands[0]);
    cudaDeviceSynchronize();
    timing.solveMs = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count() / static_cast<double>(repeats);

    return timing;
}

/**
 * @brief The largest cube side N whose Thomas benchmark buffers fit in the GPU's free memory.
 *
 * The benchmark keeps five arrays of N^3 values on the GPU: the right-hand side, the solution, and
 * the solver's three-column workspace. reserveBytes stays free for everything else the solver
 * allocates while it runs (the per-axis eigenvector matrices and their inverses, cuBLAS workspace)
 * and for allocation padding.
 *
 * @param reserveBytes Bytes of free memory to leave unclaimed.
 * @return The largest N that fits, or 0 if none does.
 */
template <typename Real>
size_t largestThomasGrid(size_t reserveBytes) {
    size_t freeBytes = 0, totalBytes = 0;
    cudaMemGetInfo(&freeBytes, &totalBytes);
    if (freeBytes <= reserveBytes) return 0;

    const size_t maxTotal = (freeBytes - reserveBytes) / (5 * sizeof(Real));
    size_t n = static_cast<size_t>(std::cbrt(static_cast<double>(maxTotal)));
    while ((n + 1) * (n + 1) * (n + 1) <= maxTotal) ++n;
    while (n > 0 && n * n * n > maxTotal) --n;
    return n;
}

/**
 * @brief The all-Neumann, variable-segment boundary of SolverRuntimes' "Var" case, with every
 *        spacing set to 1, built on caller-supplied spacing buffers.
 * @param dx   At least n + 1 values; the first n + 1 are overwritten with 1.
 * @param dy   As dx.
 * @param dz   As dx.
 * @param n    Grid side length.
 * @param hand Handle for the fills.
 * @return The boundary configuration.
 */
template <typename Real>
auto makeVariableUnitSpacingOn(SimpleArray<Real>& dx, SimpleArray<Real>& dy, SimpleArray<Real>& dz, size_t n, Handle& hand) {
    auto dxSub = dx.subArray(0, n + 1); dxSub.fill(1.0, hand);
    auto dySub = dy.subArray(0, n + 1); dySub.fill(1.0, hand);
    auto dzSub = dz.subArray(0, n + 1); dzSub.fill(1.0, hand);

    AxisSegmentHost<VariableSegment<Real>> varX({0, true}, {0, true}, dxSub);
    AxisSegmentHost<VariableSegment<Real>> varY({0, true}, {0, true}, dySub);
    AxisSegmentHost<VariableSegment<Real>> varZ({0, true}, {0, true}, dzSub);

    return BoundaryConfigHost<Real,
                              VariableSegment<Real>,
                              VariableSegment<Real>,
                              VariableSegment<Real>>(varX, varY, varZ);
}

/**
 * @brief Prints one ThomasOnly row; a negative residual prints as "skipped".
 * @param n        Grid side length.
 * @param unif     Uniform-segment, all-Dirichlet times.
 * @param residual ||L*x_unif - b||, or negative if it was not computed.
 * @param var      Variable-segment, all-Neumann times.
 */
void printThomasRow(size_t n, const ThomasTiming& unif, double residual, const ThomasTiming& var) {
    std::cout << std::setw(8)  << n
              << std::setw(14) << n * n * n
              << std::setw(22) << std::fixed << std::setprecision(3) << unif.buildAndSolveMs
              << std::setw(18) << std::fixed << std::setprecision(3) << unif.solveMs;
    if (residual < 0) std::cout << std::setw(20) << "skipped";
    else std::cout << std::setw(20) << std::scientific << std::setprecision(3) << residual;
    std::cout << std::setw(22) << std::fixed << std::setprecision(3) << var.buildAndSolveMs
              << std::setw(18) << std::fixed << std::setprecision(3) << var.solveMs
              << std::endl;
}

/**
 * @brief Times only the Thomas solver, on cubic grids up to the largest that fits in GPU memory.
 *
 * Unlike SolverRuntimes, this allocates its buffers once, sized from the free memory at launch,
 * and spends that memory on the Thomas solver alone: no EigenDecomp3d run, and no separate
 * solution buffer per case. The grid sizes follow SolverRuntimes (steps of 2 up to 200, then 1),
 * so rows line up with earlier results.
 *
 * Columns:
 * - Unif: uniform segments, all Dirichlet (makeUniformNeumann, whose name is out of date). Same
 *   setup as SolverRuntimes' "Thomas Unif".
 * - Var: variable segments with unit spacing, all Neumann. Same setup as SolverRuntimes'
 *   "Thomas Var".
 * - "build+solve" is timed as SolverRuntimes times it; "solve" is the average of `repeats` solves
 *   on the built solver.
 *
 * The residual check needs a banded Laplacian of seven diagonals plus a result vector, about eight
 * times N^3 values, far more than the solver itself. Before each check the free memory is compared
 * with that need plus 64 MB; once it no longer fits, the check is skipped for the rest of the run
 * and the timings continue. It is not left to fail: a failed allocation there has caused an
 * illegal memory access rather than a catchable error. Var gets no residual: its right-hand side is random
 * and not zero-mean, so the singular all-Neumann system has no exact solution to check against,
 * and the timing does not depend on it.
 */
TEST(Benchmark, ThomasOnly) {
    using Real = double;

    Handles hands(3);
    const Real3d delta(1, 1, 1);
    const size_t repeats = 3;
    const size_t reserveBytes = size_t{256} << 20;

    std::cout << DeviceMemory() << std::endl;

    const size_t maxN = largestThomasGrid<Real>(reserveBytes);
    ASSERT_GE(maxN, size_t{2}) << "Not enough free GPU memory for even a 2x2x2 grid.";
    const size_t maxTotal = maxN * maxN * maxN;
    std::cout << "Buffers sized for N up to " << maxN << " (" << maxTotal << " points)" << std::endl;

    auto rhs        = SimpleArray<Real>::create(maxTotal, hands[0]);
    auto x          = SimpleArray<Real>::create(maxTotal, hands[0]);
    auto sizeOfBX3  = Mat<Real>::create(maxTotal, 3, hands[0]);
    auto dx         = SimpleArray<Real>::create(maxN + 1, hands[0]);
    auto dy         = SimpleArray<Real>::create(maxN + 1, hands[0]);
    auto dz         = SimpleArray<Real>::create(maxN + 1, hands[0]);
    auto normResult = Singleton<Real>::create(hands[0]);

    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<Real> dist(-1, 1);
        std::vector<Real> host(maxTotal);
        for (Real& v : host) v = dist(rng);
        rhs.set(host.data(), hands[0]);
        hands[0].synch();
    }

    std::cout << std::setw(8)  << "N"
              << std::setw(14) << "N^3"
              << std::setw(22) << "Unif build+solve"
              << std::setw(18) << "Unif solve"
              << std::setw(20) << "||L*x_unif - b||"
              << std::setw(22) << "Var build+solve"
              << std::setw(18) << "Var solve"
              << "   (ms)" << std::endl;
    std::cout << std::string(125, '-') << std::endl;

    bool residualFits = true;

    for (size_t n = 2; n <= maxN; n += (n < 200 ? 2 : 1)) {
        const size_t total = n * n * n;
        const GridDim dim(n, n, n);

        auto rhsSub       = rhs.subArray(0, total);
        auto xSub         = x.subArray(0, total);
        auto sizeOfBX3Sub = sizeOfBX3.subMat(0, 0, total, 3);

        const auto uniformBoundary  = makeUniformNeumann<Real>(delta, dim);
        const auto variableBoundary = makeVariableUnitSpacingOn<Real>(dx, dy, dz, n, hands[0]);

        ThomasTiming unif, var;
        double residual = -1;

        try {
            unif = timeThomas(uniformBoundary, xSub, rhsSub, sizeOfBX3Sub, hands, repeats);
        } catch (...) {
            cudaGetLastError();
            std::cout << "OOM in EigenDecompThomas (Uniform) at N=" << n << std::endl;
            break;
        }

        if (residualFits) {
            size_t freeBytes = 0, totalBytes = 0;
            cudaMemGetInfo(&freeBytes, &totalBytes);
            const size_t residualBytes = 8 * total * sizeof(Real) + (size_t{64} << 20);
            if (residualBytes > freeBytes) {
                residualFits = false;
                std::cout << "Residual check no longer fits in memory from N=" << n
                          << "; timings continue without it." << std::endl;
            } else {
                residual = computeResidualNorm(uniformBoundary.forDevice(), xSub, rhsSub, normResult, hands);
            }
        }

        try {
            var = timeThomas(variableBoundary, xSub, rhsSub, sizeOfBX3Sub, hands, repeats);
        } catch (...) {
            cudaGetLastError();
            std::cout << "OOM in EigenDecompThomas (Variable) at N=" << n << std::endl;
            break;
        }

        printThomasRow(n, unif, residual, var);
    }
}
