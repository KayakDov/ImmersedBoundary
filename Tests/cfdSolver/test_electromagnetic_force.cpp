/**
 * @file test_electromagnetic_force.cpp
 * @brief GoogleTest direct verification of ElectromagneticForce: each
 * function's actual shipped implementation (not a hand-copied duplicate)
 * checked against a Fortran-literal, ghost-padded reference, across every
 * grid-size and boundary-condition combination this project tests.
 *
 * Before this file, ElectromagneticForce.cuh -- the code at the center of
 * the Hartmann=1000 Potential bisection (task #41) -- had no direct unit
 * coverage: it was verified only by manual code reading and by the
 * end-to-end golden-value comparison. This closes that gap the same way
 * test_operators_host.cpp closes it for FiniteDifferenceOperators.
 */

#include <gtest/gtest.h>
#include "physics/ElectromagneticForce.cuh"
#include "CfdTestHelpers.h"
#include "CfdTestPhysicsHelpers.h"
#include <random>
#include <vector>

using std::vector;

namespace {

std::mt19937 rng(20261108);
std::uniform_real_distribution<double> dist(-1.0, 1.0);
double R() { return dist(rng); }

/// potentialRightHandSide's domain (i in [1,nx], j in [1,ny1], k in [1,nz])
/// never crosses a velocity ghost -- VMx's own z-extent is nz1 and VMz's
/// own x-extent is nx1, so i+1/k+1 always land on a real cell. The
/// ghost-fill is still applied so this test depends on the same
/// production boundary semantics as every other test in this suite, not
/// on that fact holding.
void CheckPotentialRightHandSide(int nx1, int ny1, int nz1, const cfdtest::Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    cfdtest::PaddedField vx(nx1, ny2, nz2), vz(nx1 + 1, ny2, nz1), rhsRef(nx, ny2, nz);
    vector<double> vxU((size_t)nx * ny1 * nz1), vzU((size_t)nx1 * ny1 * nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vx.at(i, j, k) = v; vxU[cfdtest::Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vz.at(i, j, k) = v; vzU[cfdtest::Uidx(i, j, k, nx1, ny1)] = v; }
    cfdtest::FillVelGhostsZero(vx); cfdtest::FillVelGhostsZero(vz);

    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) {
        double dVxDz = (vx.at(i, j, k + 1) - vx.at(i, j, k)) / sp.hpz[k];
        double dVzDx = (vz.at(i + 1, j, k) - vz.at(i, j, k)) / sp.hpx[i];
        rhsRef.at(i, j, k) = dVzDx - dVxDz;
    }

    FieldView<double> vxv(vxU.data(), nx, ny1, nz1), vzv(vzU.data(), nx1, ny1, nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k)
        mismatches.Check(rhsRef.at(i, j, k), ElectromagneticForce::potentialRightHandSide(i, j, k, vxv, vzv, sp.view()));
}

/// applyForceX's potential reads (i,j,k) and (i,j,k-1) cross potential's
/// own z ghost at both k=1 (reads k-1=0) and k=nz1 (one past potential's
/// real z-extent nz) -- the BC sweep below is exercising real boundary
/// logic, not an unreachable corner.
void CheckApplyForceX(int nx1, int ny1, int nz1, BoundaryKind bx, BoundaryKind by, BoundaryKind bz,
                       const cfdtest::Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    cfdtest::PaddedField vx(nx1, ny2, nz2), potential(nx + 1, ny2, nz + 1), rhsRef(nx1, ny2, nz2);
    vector<double> vxU((size_t)nx * ny1 * nz1), potentialU((size_t)nx * ny1 * nz), rhsImpl((size_t)nx * ny1 * nz1);
    AxisBoundaries bc{bx, by, bz};
    double coefficient = 1.7;

    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vx.at(i, j, k) = v; vxU[cfdtest::Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); potential.at(i, j, k) = v; potentialU[cfdtest::Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); rhsRef.at(i, j, k) = v; rhsImpl[cfdtest::Uidx(i, j, k, nx, ny1)] = v; }
    cfdtest::FillVelGhostsZero(vx);
    cfdtest::FillTempGhosts(potential, bx, by, bz);

    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) {
        double dPotentialDz = (potential.at(i, j, k) - potential.at(i, j, k - 1)) / sp.hz12[k - 1];
        rhsRef.at(i, j, k) += coefficient * (dPotentialDz + vx.at(i, j, k));
    }

    FieldView<double> rhsv(rhsImpl.data(), nx, ny1, nz1), vxv(vxU.data(), nx, ny1, nz1), potv(potentialU.data(), nx, ny1, nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        ElectromagneticForce::applyForceX(i, j, k, rhsv, vxv, potv, bc, sp.view(), coefficient);

    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        mismatches.Check(rhsRef.at(i, j, k), rhsImpl[cfdtest::Uidx(i, j, k, nx, ny1)]);
}

/// applyForceZ's potential reads (i,j,k) and (i-1,j,k) cross potential's
/// own x ghost at both i=1 and i=nx1 (one past potential's real x-extent
/// nx) -- mirrors CheckApplyForceX's reasoning on the other axis.
void CheckApplyForceZ(int nx1, int ny1, int nz1, BoundaryKind bx, BoundaryKind by, BoundaryKind bz,
                       const cfdtest::Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, nz = nz1 - 1, ny2 = ny1 + 1;
    cfdtest::PaddedField vz(nx1 + 1, ny2, nz1), potential(nx + 1, ny2, nz + 1), rhsRef(nx1, ny2, nz1);
    vector<double> vzU((size_t)nx1 * ny1 * nz), potentialU((size_t)nx * ny1 * nz), rhsImpl((size_t)nx1 * ny1 * nz);
    AxisBoundaries bc{bx, by, bz};
    double coefficient = 1.7;

    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vz.at(i, j, k) = v; vzU[cfdtest::Uidx(i, j, k, nx1, ny1)] = v; }
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); potential.at(i, j, k) = v; potentialU[cfdtest::Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); rhsRef.at(i, j, k) = v; rhsImpl[cfdtest::Uidx(i, j, k, nx1, ny1)] = v; }
    cfdtest::FillVelGhostsZero(vz);
    cfdtest::FillTempGhosts(potential, bx, by, bz);

    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) {
        double dPotentialDx = (potential.at(i, j, k) - potential.at(i - 1, j, k)) / sp.hx12[i - 1];
        rhsRef.at(i, j, k) -= coefficient * (dPotentialDx - vz.at(i, j, k));
    }

    FieldView<double> rhsv(rhsImpl.data(), nx1, ny1, nz), vzv(vzU.data(), nx1, ny1, nz), potv(potentialU.data(), nx, ny1, nz);
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k)
        ElectromagneticForce::applyForceZ(i, j, k, rhsv, vzv, potv, bc, sp.view(), coefficient);

    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k)
        mismatches.Check(rhsRef.at(i, j, k), rhsImpl[cfdtest::Uidx(i, j, k, nx1, ny1)]);
}

/// @brief Checks subtractReferenceValue against a direct host subtraction of the same scalar from every cell.
void CheckSubtractReferenceValue(int nx1, int ny1, int nz1, cfdtest::MismatchCollector& mismatches) {
    vector<double> potentialRef((size_t)nx1 * ny1 * nz1), potentialImpl((size_t)nx1 * ny1 * nz1);
    for (auto& v : potentialRef) v = R();
    potentialImpl = potentialRef;
    double referenceValue = R();

    for (auto& v : potentialRef) v -= referenceValue;
    FieldView<double> potv(potentialImpl.data(), nx1, ny1, nz1);
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        ElectromagneticForce::subtractReferenceValue(i, j, k, potv, &referenceValue);

    for (size_t idx = 0; idx < potentialRef.size(); ++idx) mismatches.Check(potentialRef[idx], potentialImpl[idx]);
}

} // namespace

TEST(ElectromagneticForceTest, PotentialRightHandSide) {
    cfdtest::MismatchCollector mismatches("potentialRightHandSide");
    for (int nx1 : cfdtest::TestGridSizes()) for (int ny1 : cfdtest::TestGridSizes()) for (int nz1 : cfdtest::TestGridSizes()) {
        cfdtest::Sp sp(std::max({nx1, ny1, nz1}) + 2);
        CheckPotentialRightHandSide(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(ElectromagneticForceTest, ApplyForceX) {
    cfdtest::MismatchCollector mismatches("applyForceX");
    BoundaryKind bcVals[] = {BoundaryKind::Dirichlet, BoundaryKind::Neumann};
    for (int nx1 : cfdtest::TestGridSizes()) for (int ny1 : cfdtest::TestGridSizes()) for (int nz1 : cfdtest::TestGridSizes()) {
        cfdtest::Sp sp(std::max({nx1, ny1, nz1}) + 2);
        for (auto bx : bcVals) for (auto by : bcVals) for (auto bz : bcVals)
            CheckApplyForceX(nx1, ny1, nz1, bx, by, bz, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(ElectromagneticForceTest, ApplyForceZ) {
    cfdtest::MismatchCollector mismatches("applyForceZ");
    BoundaryKind bcVals[] = {BoundaryKind::Dirichlet, BoundaryKind::Neumann};
    for (int nx1 : cfdtest::TestGridSizes()) for (int ny1 : cfdtest::TestGridSizes()) for (int nz1 : cfdtest::TestGridSizes()) {
        cfdtest::Sp sp(std::max({nx1, ny1, nz1}) + 2);
        for (auto bx : bcVals) for (auto by : bcVals) for (auto bz : bcVals)
            CheckApplyForceZ(nx1, ny1, nz1, bx, by, bz, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(ElectromagneticForceTest, SubtractReferenceValue) {
    cfdtest::MismatchCollector mismatches("subtractReferenceValue");
    for (int nx1 : cfdtest::TestGridSizes()) for (int ny1 : cfdtest::TestGridSizes()) for (int nz1 : cfdtest::TestGridSizes())
        CheckSubtractReferenceValue(nx1, ny1, nz1, mismatches);
    mismatches.Finish();
}
