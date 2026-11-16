/**
 * @file WorkspaceContractTest.cpp
 * @brief Checks the memory contract of EigenDecompThomas and EigenDecomp3d.
 *
 * - With a workspace of its own, solve() leaves b exactly as it was.
 * - b's own memory may be given to the solver as its workspace. solve() then gives the same x
 *   and overwrites b. The CFD's SolvedField and the Fortran wrapper rely on this.
 * - EigenDecompThomas only: x may not be the workspace. solve() throws std::invalid_argument and
 *   leaves b as it was.
 *
 * The grids are not cubes, and one has the minimum of two layers, so that a mix-up of axes, or of
 * the first and last steps of the Thomas sweep, shows up.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "solvers/EigenDecomp/EigenDecompThomas.cuh"
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"
#include "poisson/Poisson.cuh"
#include "poisson/BoundaryConfig.cuh"
#include "deviceArrays/headers/Handles.h"

namespace {

/**
 * @brief Copies a device array to the host and waits for the copy to finish.
 * @param array The device array.
 * @param hand The stream the copy runs on.
 * @return The array's values.
 */
template<typename Real>
std::vector<Real> toHost(const SimpleArray<Real>& array, Handle& hand) {
    std::vector<Real> host(array.size());
    array.get(host.data(), hand);
    hand.synch();
    return host;
}

/**
 * @brief Spacings for one axis, in the form each segment kind expects. Same rules as
 * generateAxisDeltas in test.cpp: a single spacing for the uniform kinds, n + 1 random spacings
 * for VariableDeltaLapl, and spacings built from n random cell widths for FluxLapl.
 * @param n Number of nodes along the axis.
 * @param kind The axis's segment kind.
 * @param rng Random source.
 * @return The axis's spacings.
 */
template<typename Real>
std::vector<Real> axisDeltas(size_t n, eigen::LaplOperatorT kind, std::mt19937& rng) {
    if (!eigen::hasVariableDelta(kind) || n <= 1) return {Real(1)};
    std::uniform_real_distribution<Real> dist(0.5, 2.0);
    std::vector<Real> deltas(n + 1);
    if (kind == eigen::FluxLapl) {
        std::vector<Real> widths(n);
        for (auto& width : widths) width = dist(rng);
        deltas[0] = widths[0] / 2;
        for (size_t i = 1; i < n; ++i) deltas[i] = (widths[i - 1] + widths[i]) / 2;
        deltas[n] = widths[n - 1] / 2;
    } else {
        for (auto& delta : deltas) delta = dist(rng);
    }
    return deltas;
}

/**
 * @brief Largest absolute value in an array.
 * @param values The array.
 * @return max |values[i]|.
 */
template<typename Real>
Real maxAbs(const std::vector<Real>& values) {
    Real largest = 0;
    for (Real value : values) largest = std::max(largest, std::abs(value));
    return largest;
}

/**
 * @brief Largest absolute difference between two arrays of the same length.
 * @return max |a[i] - b[i]|.
 */
template<typename Real>
Real maxAbsDiff(const std::vector<Real>& a, const std::vector<Real>& b) {
    Real largest = 0;
    for (size_t i = 0; i < a.size(); ++i) largest = std::max(largest, std::abs(a[i] - b[i]));
    return largest;
}

/**
 * @brief Euclidean distance between two arrays of the same length.
 * @return ||a - b||.
 */
template<typename Real>
Real distance(const std::vector<Real>& a, const std::vector<Real>& b) {
    Real sumOfSquares = 0;
    for (size_t i = 0; i < a.size(); ++i) sumOfSquares += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(sumOfSquares);
}

/**
 * @brief Runs the memory-contract checks for one solver, one grid, one choice of segment kinds and
 * one choice of boundary conditions.
 *
 * The right-hand side is b = L x0 for a random x0, so it is consistent even when every boundary
 * is Neumann and L is singular. Boundary values are zero, so no boundary correction is needed.
 * The residual tolerance is the one laplacianBattery in test.cpp uses.
 *
 * The solution found with b as the workspace is compared with the one found with a separate
 * workspace. The arithmetic is the same in both, so the two should agree to the last bit; the
 * check allows a few rounding errors in case cuBLAS picks a different kernel for different
 * buffers. A mistake in the aliasing would show up as a difference of the size of x itself.
 *
 * @param thomas True to check EigenDecompThomas, false for EigenDecomp3d.
 * @param dim The grid. Must have at least 2 layers.
 * @param kinds The segment kind of each axis.
 * @param allNeumann True for Neumann conditions on every boundary (singular L), false for
 * Dirichlet conditions on every boundary.
 * @param hands At least 3 handles.
 */
void checkWorkspaceContract(bool thomas, const GridDim& dim, const XYZ<eigen::LaplOperatorT>& kinds, bool allNeumann, Handles& hands) {
    using Real = double;
    Handle& hand = hands[0];
    std::mt19937 rng(static_cast<unsigned>(dim.size() + 10 * kinds.z + allNeumann));

    const XYZ<std::vector<Real>> deltas(
        axisDeltas<Real>(dim.cols, kinds.x, rng),
        axisDeltas<Real>(dim.rows, kinds.y, rng),
        axisDeltas<Real>(dim.layers, kinds.z, rng)
    );
    const XYZ<bool> neumann(allNeumann, allNeumann, allNeumann);
    const XYZ<Real> zero(0, 0, 0);

    std::ostringstream where;
    where << (thomas ? " EigenDecompThomas," : " EigenDecomp3d,") << " dim = " << dim << ", kinds = " << kinds << ", allNeumann = " << allNeumann;

    buildBoundaryConfigAndLaunch<Real>(dim, deltas, neumann, neumann, zero, zero, kinds, hand, [&](const auto& boundaryHost) {
        using SegZ = typename std::decay_t<decltype(boundaryHost.z)>::SegmentType;
        const size_t n = dim.size();
        auto makeSolver = [&](SimpleArray<Real> workspace) -> std::unique_ptr<EigenDecompSolver<Real>> {
            if (thomas) return std::make_unique<EigenDecompThomas<Real, SegZ>>(boundaryHost, hands, workspace);
            return std::make_unique<EigenDecomp3d<Real>>(boundaryHost.forDevice(), hands, workspace);
        };

        std::uniform_real_distribution<Real> dist(-1, 1);
        std::vector<Real> x0(n);
        for (auto& value : x0) value = dist(rng);

        auto laplacian = poisson::laplacian<Real>(boundaryHost.forDevice(), hand);
        auto xTrue = SimpleArray<Real>::create(x0, hand);
        auto b = SimpleArray<Real>::create(n, hand);
        laplacian.bandedMult(xTrue, b, &hand, GPUScalar<Real>::get(1, hand), GPUScalar<Real>::get(0, hand), false);
        const std::vector<Real> bBefore = toHost(b, hand);

        auto workspace = SimpleArray<Real>::create(n, hand);
        auto xSeparate = SimpleArray<Real>::create(n, hand);
        const auto separate = makeSolver(workspace);
        separate->solve(xSeparate, b, hand);
        EXPECT_EQ(toHost(b, hand), bBefore) << "solve() changed b, which was not its workspace." << where.str();

        if (thomas) {
            EXPECT_THROW(separate->solve(workspace, b, hand), std::invalid_argument) << where.str();
            EXPECT_EQ(toHost(b, hand), bBefore) << "b changed when solve() rejected x as the workspace." << where.str();
        }

        auto xShared = SimpleArray<Real>::create(n, hand);
        const auto shared = makeSolver(b);
        shared->solve(xShared, b, hand);

        const std::vector<Real> fromSeparate = toHost(xSeparate, hand);
        const std::vector<Real> fromShared = toHost(xShared, hand);
        EXPECT_LE(maxAbsDiff(fromShared, fromSeparate), 64 * std::numeric_limits<Real>::epsilon() * maxAbs(fromSeparate))
            << "Giving b as the workspace changed the solution." << where.str();

        auto lx = SimpleArray<Real>::create(n, hand);
        laplacian.bandedMult(xShared, lx, &hand, GPUScalar<Real>::get(1, hand), GPUScalar<Real>::get(0, hand), false);
        EXPECT_LE(distance(toHost(lx, hand), bBefore), 1e-11 * std::sqrt(static_cast<Real>(n)))
            << "||L x - b|| with b as the workspace." << where.str();
    });
}

/**
 * @brief Runs checkWorkspaceContract for every segment kind along Z, for Dirichlet and for
 * all-Neumann (singular) boundaries, on two grids. X and Y use variable and flux-form spacing, so
 * their eigenvector transforms are not orthonormal.
 * @param thomas True to check EigenDecompThomas, false for EigenDecomp3d.
 */
void checkEveryKindAlongZ(bool thomas) {
    Handles hands(3);
    const GridDim dims[] = {GridDim(7, 6, 5), GridDim(4, 3, 2)};
    const eigen::LaplOperatorT zKinds[] = {
        eigen::UniformDeltaNodeCenteredLapl, eigen::UniformDeltaStaggeredLapl, eigen::VariableDeltaLapl, eigen::FluxLapl
    };
    for (const GridDim& dim : dims)
        for (eigen::LaplOperatorT zKind : zKinds)
            for (bool allNeumann : {false, true})
                checkWorkspaceContract(thomas, dim, XYZ<eigen::LaplOperatorT>(eigen::VariableDeltaLapl, eigen::FluxLapl, zKind), allNeumann, hands);
}

} // namespace

TEST(WorkspaceContract, EigenDecompThomas) { checkEveryKindAlongZ(true); }

TEST(WorkspaceContract, EigenDecomp3d) { checkEveryKindAlongZ(false); }
