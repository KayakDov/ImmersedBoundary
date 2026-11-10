/**
 * @file CfdTestPhysicsHelpers.h
 * @brief Shared host-only helpers for GoogleTest files that check a
 * production FiniteDifferenceOperators/ElectromagneticForce function
 * against a hand-written, ghost-padded reference using the project's own
 * BoundaryKind and Spacing<T> types (test_operators_host.cpp,
 * test_electromagnetic_force.cpp).
 *
 * Kept separate from CfdTestHelpers.h, which every host-only .cpp test
 * includes -- including ones with nothing to do with boundary conditions
 * or mesh spacing, such as test_config.cpp. BoundaryKind and Spacing<T>
 * are only reachable through physics/Operators.cuh's own include chain,
 * which pulls in math/XYZ.cuh and, with it, <cuda_runtime.h>; folding
 * FillTempGhosts/Sp into CfdTestHelpers.h would force that chain onto
 * every host-only test, not just the ones that already need it.
 */
#ifndef CFDSOLVER_TESTS_CFDTESTPHYSICSHELPERS_H
#define CFDSOLVER_TESTS_CFDTESTPHYSICSHELPERS_H

#include "CfdTestHelpers.h"
#include "physics/Operators.cuh"
#include <cmath>
#include <vector>

namespace cfdtest {

/// @brief Fills a PaddedField's temperature ghosts on every face: 0 for
/// Dirichlet, a mirror of the nearest real cell for Neumann.
inline void FillTempGhosts(PaddedField& t, BoundaryKind bx, BoundaryKind by, BoundaryKind bz) {
    bool nx = bx == BoundaryKind::Neumann, ny = by == BoundaryKind::Neumann, nz = bz == BoundaryKind::Neumann;
    int nx1 = t.nx - 1, ny1 = t.ny - 1, nz1 = t.nz - 1;
    for (int j = 0; j <= t.ny; ++j) for (int k = 0; k <= t.nz; ++k) { t.at(0, j, k) = nx ? t.at(1, j, k) : 0.0; t.at(t.nx, j, k) = nx ? t.at(nx1, j, k) : 0.0; }
    for (int i = 0; i <= t.nx; ++i) for (int k = 0; k <= t.nz; ++k) { t.at(i, 0, k) = ny ? t.at(i, 1, k) : 0.0; t.at(i, t.ny, k) = ny ? t.at(i, ny1, k) : 0.0; }
    for (int i = 0; i <= t.nx; ++i) for (int j = 0; j <= t.ny; ++j) { t.at(i, j, 0) = nz ? t.at(i, j, 1) : 0.0; t.at(i, j, t.nz) = nz ? t.at(i, j, nz1) : 0.0; }
}

/// @brief Shared mesh-spacing arrays (not boundary-related -- plain 0-indexed coefficient arrays).
struct Sp {
    std::vector<double> hx12, hy12, hz12, hpx, hpy, hpz;
    explicit Sp(int n) : hx12(n + 2), hy12(n + 2), hz12(n + 2), hpx(n + 2), hpy(n + 2), hpz(n + 2) {
        for (int i = 0; i < n + 2; ++i) {
            hx12[i] = 0.5 + 0.3 * std::sin(0.37 * i + 1.0); hy12[i] = 0.5 + 0.3 * std::sin(0.41 * i + 2.0); hz12[i] = 0.5 + 0.3 * std::sin(0.29 * i + 3.0);
            hpx[i] = 0.5 + 0.3 * std::sin(0.53 * i + 4.0);  hpy[i] = 0.5 + 0.3 * std::sin(0.61 * i + 5.0);  hpz[i] = 0.5 + 0.3 * std::sin(0.71 * i + 6.0);
        }
    }
    Spacing<double> view() const { return Spacing<double>{hx12.data(), hy12.data(), hz12.data(), hpx.data(), hpy.data(), hpz.data()}; }
};

} // namespace cfdtest

#endif //CFDSOLVER_TESTS_CFDTESTPHYSICSHELPERS_H
