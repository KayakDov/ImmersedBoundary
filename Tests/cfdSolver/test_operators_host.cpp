/**
 * @file test_operators_host.cpp
 * @brief GoogleTest direct verification of FiniteDifferenceOperators: each
 * operator's actual shipped implementation (not a hand-copied duplicate)
 * checked against a Fortran-literal, ghost-padded reference, across every
 * grid-size and boundary-condition combination this project tests.
 */

#include <gtest/gtest.h>
#include "physics/Operators.cuh"
#include "CfdTestHelpers.h"
#include "CfdTestPhysicsHelpers.h"
#include <algorithm>
#include <random>
#include <vector>

using std::vector;

namespace {

std::mt19937 rng(777);
std::uniform_real_distribution<double> dist(-1.0, 1.0);
double R() { return dist(rng); }

/// @brief Checks temperatureAdvection against a ghost-padded, non-simplified reference, across every boundary-condition combination.
void CheckTemperatureAdvection(int nx1, int ny1, int nz1, BoundaryKind bx, BoundaryKind by, BoundaryKind bz,
                                const cfdtest::Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    cfdtest::PaddedField tmpr(nx1 + 1, ny2, nz2), rhsRef(nx1 + 1, ny2, nz2);
    cfdtest::PaddedField vx(nx1, ny2, nz2), vy(nx1 + 1, ny1, nz2), vz(nx1 + 1, ny2, nz1);
    vector<double> tmprU((size_t)nx1 * ny1 * nz1);
    vector<double> vxU((size_t)nx * ny1 * nz1), vyU((size_t)nx1 * ny * nz1), vzU((size_t)nx1 * ny1 * nz);
    vector<double> buoyancyByX(nx1 + 2);
    for (auto& v : buoyancyByX) v = R();

    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double t = R(); tmpr.at(i, j, k) = t; tmprU[cfdtest::Uidx(i, j, k, nx1, ny1)] = t; }
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vx.at(i, j, k) = v; vxU[cfdtest::Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vy.at(i, j, k) = v; vyU[cfdtest::Uidx(i, j, k, nx1, ny)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vz.at(i, j, k) = v; vzU[cfdtest::Uidx(i, j, k, nx1, ny1)] = v; }
    cfdtest::FillTempGhosts(tmpr, bx, by, bz);
    cfdtest::FillVelGhostsZero(vx); cfdtest::FillVelGhostsZero(vy); cfdtest::FillVelGhostsZero(vz);

    /* Reference: Teta read directly (NOT through ghost-fill) at every
       absolute x index, exactly matching the real Fortran init loop's
       direct-formula boundary coverage. */
    auto tetaAt = [&](int i) { return buoyancyByX[i]; };

    for (int i = 1; i <= nx1; ++i) { double px = 0.5 / sp.hx12[i - 1];
        for (int j = 1; j <= ny1; ++j) { double py = 0.5 / sp.hy12[j - 1];
            for (int k = 1; k <= nz1; ++k) { double pz = 0.5 / sp.hz12[k - 1];
                double a1 = -vy.at(i, j - 1, k) * py, a4 = vy.at(i, j, k) * py;
                double a2 = -vx.at(i - 1, j, k) * px, a5 = vx.at(i, j, k) * px;
                double a6 = -vz.at(i, j, k - 1) * pz, a7 = vz.at(i, j, k) * pz;
                double a3 = -(a1 + a2 + a4 + a5 + a6 + a7);
                rhsRef.at(i, j, k) = a1 * (tmpr.at(i, j - 1, k) + tetaAt(i)) + a2 * (tmpr.at(i - 1, j, k) + tetaAt(i - 1))
                    + a3 * (tmpr.at(i, j, k) + tetaAt(i)) + a4 * (tmpr.at(i, j + 1, k) + tetaAt(i))
                    + a5 * (tmpr.at(i + 1, j, k) + tetaAt(i + 1)) + a6 * (tmpr.at(i, j, k - 1) + tetaAt(i)) + a7 * (tmpr.at(i, j, k + 1) + tetaAt(i));
            }}}

    FieldView<double> tv(tmprU.data(), nx1, ny1, nz1);
    FieldView<double> vxv(vxU.data(), nx, ny1, nz1), vyv(vyU.data(), nx1, ny, nz1), vzv(vzU.data(), nx1, ny1, nz);
    AxisBoundaries bc{bx, by, bz};

    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        mismatches.Check(rhsRef.at(i, j, k), FiniteDifferenceOperators::temperatureAdvection(i, j, k, tv, buoyancyByX.data(), vxv, vyv, vzv, sp.view(), bc));
}

/// @brief Checks velocityXAdvection against a ghost-padded reference.
void CheckVelocityXAdvection(int nx1, int ny1, int nz1, const cfdtest::Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    cfdtest::PaddedField vx(nx1, ny2, nz2), vy(nx1 + 1, ny1, nz2), vz(nx1 + 1, ny2, nz1), rhsRef(nx1, ny2, nz2);
    vector<double> vxU((size_t)nx * ny1 * nz1), vyU((size_t)nx1 * ny * nz1), vzU((size_t)nx1 * ny1 * nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vx.at(i, j, k) = v; vxU[cfdtest::Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vy.at(i, j, k) = v; vyU[cfdtest::Uidx(i, j, k, nx1, ny)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vz.at(i, j, k) = v; vzU[cfdtest::Uidx(i, j, k, nx1, ny1)] = v; }
    cfdtest::FillVelGhostsZero(vx); cfdtest::FillVelGhostsZero(vy); cfdtest::FillVelGhostsZero(vz);

    for (int i = 1; i <= nx; ++i) { double px = 0.25 / sp.hpx[i];
        for (int j = 1; j <= ny1; ++j) { double py = px / sp.hy12[j - 1], py1 = sp.hx12[i - 1] * py, py2 = sp.hx12[i] * py;
            for (int k = 1; k <= nz1; ++k) { double pz = px / sp.hz12[k - 1], pz1 = sp.hx12[i - 1] * pz, pz2 = sp.hx12[i] * pz;
                double a1 = -py2 * vy.at(i + 1, j - 1, k) - py1 * vy.at(i, j - 1, k), a4 = py2 * vy.at(i + 1, j, k) + py1 * vy.at(i, j, k);
                double a6 = -pz2 * vz.at(i + 1, j, k - 1) - pz1 * vz.at(i, j, k - 1), a7 = pz2 * vz.at(i + 1, j, k) + pz1 * vz.at(i, j, k);
                double a2 = -vx.at(i - 1, j, k) * px, a5 = vx.at(i + 1, j, k) * px, a3 = -(a1 + a4 + a6 + a7);
                rhsRef.at(i, j, k) = a1 * vx.at(i, j - 1, k) + a2 * vx.at(i - 1, j, k) + a3 * vx.at(i, j, k)
                    + a4 * vx.at(i, j + 1, k) + a5 * vx.at(i + 1, j, k) + a6 * vx.at(i, j, k - 1) + a7 * vx.at(i, j, k + 1);
            }}}
    FieldView<double> vxv(vxU.data(), nx, ny1, nz1), vyv(vyU.data(), nx1, ny, nz1), vzv(vzU.data(), nx1, ny1, nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        mismatches.Check(rhsRef.at(i, j, k), FiniteDifferenceOperators::velocityXAdvection(i, j, k, vxv, vyv, vzv, sp.view()));
}

/// @brief Checks velocityYAdvection against a ghost-padded reference.
void CheckVelocityYAdvection(int nx1, int ny1, int nz1, const cfdtest::Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    cfdtest::PaddedField vx(nx1, ny2, nz2), vy(nx1 + 1, ny1, nz2), vz(nx1 + 1, ny2, nz1), rhsRef(nx1 + 1, ny1, nz2);
    vector<double> vxU((size_t)nx * ny1 * nz1), vyU((size_t)nx1 * ny * nz1), vzU((size_t)nx1 * ny1 * nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vx.at(i, j, k) = v; vxU[cfdtest::Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vy.at(i, j, k) = v; vyU[cfdtest::Uidx(i, j, k, nx1, ny)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vz.at(i, j, k) = v; vzU[cfdtest::Uidx(i, j, k, nx1, ny1)] = v; }
    cfdtest::FillVelGhostsZero(vx); cfdtest::FillVelGhostsZero(vy); cfdtest::FillVelGhostsZero(vz);

    for (int j = 1; j <= ny; ++j) { double py = 0.25 / sp.hpy[j];
        for (int i = 1; i <= nx1; ++i) { double px = py / sp.hx12[i - 1], px1 = sp.hy12[j - 1] * px, px2 = sp.hy12[j] * px;
            for (int k = 1; k <= nz1; ++k) { double pz = py / sp.hz12[k - 1], pz1 = sp.hy12[j - 1] * pz, pz2 = sp.hy12[j] * pz;
                double a1 = -vy.at(i, j - 1, k) * py, a4 = vy.at(i, j + 1, k) * py;
                double a2 = -px2 * vx.at(i - 1, j + 1, k) - px1 * vx.at(i - 1, j, k), a5 = px2 * vx.at(i, j + 1, k) + px1 * vx.at(i, j, k);
                double a6 = -pz2 * vz.at(i, j + 1, k - 1) - pz1 * vz.at(i, j, k - 1), a7 = pz2 * vz.at(i, j + 1, k) + pz1 * vz.at(i, j, k);
                double a3 = -(a2 + a5 + a6 + a7);
                rhsRef.at(i, j, k) = a1 * vy.at(i, j - 1, k) + a2 * vy.at(i - 1, j, k) + a3 * vy.at(i, j, k)
                    + a4 * vy.at(i, j + 1, k) + a5 * vy.at(i + 1, j, k) + a6 * vy.at(i, j, k - 1) + a7 * vy.at(i, j, k + 1);
            }}}
    FieldView<double> vxv(vxU.data(), nx, ny1, nz1), vyv(vyU.data(), nx1, ny, nz1), vzv(vzU.data(), nx1, ny1, nz);
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k)
        mismatches.Check(rhsRef.at(i, j, k), FiniteDifferenceOperators::velocityYAdvection(i, j, k, vxv, vyv, vzv, sp.view()));
}

/// @brief Checks velocityZAdvection against a ghost-padded reference.
void CheckVelocityZAdvection(int nx1, int ny1, int nz1, const cfdtest::Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    cfdtest::PaddedField vx(nx1, ny2, nz2), vy(nx1 + 1, ny1, nz2), vz(nx1 + 1, ny2, nz1), rhsRef(nx1 + 1, ny2, nz1);
    vector<double> vxU((size_t)nx * ny1 * nz1), vyU((size_t)nx1 * ny * nz1), vzU((size_t)nx1 * ny1 * nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vx.at(i, j, k) = v; vxU[cfdtest::Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vy.at(i, j, k) = v; vyU[cfdtest::Uidx(i, j, k, nx1, ny)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vz.at(i, j, k) = v; vzU[cfdtest::Uidx(i, j, k, nx1, ny1)] = v; }
    cfdtest::FillVelGhostsZero(vx); cfdtest::FillVelGhostsZero(vy); cfdtest::FillVelGhostsZero(vz);

    for (int k = 1; k <= nz; ++k) { double pz = 0.25 / sp.hpz[k];
        for (int i = 1; i <= nx1; ++i) { double px = pz / sp.hx12[i - 1], px1 = sp.hz12[k - 1] * px, px2 = sp.hz12[k] * px;
            for (int j = 1; j <= ny1; ++j) { double py = pz / sp.hy12[j - 1], py1 = sp.hz12[k - 1] * py, py2 = sp.hz12[k] * py;
                double a6 = -vz.at(i, j, k - 1) * pz, a7 = vz.at(i, j, k + 1) * pz;
                double a2 = -px2 * vx.at(i - 1, j, k + 1) - px1 * vx.at(i - 1, j, k), a5 = px2 * vx.at(i, j, k + 1) + px1 * vx.at(i, j, k);
                double a1 = -py2 * vy.at(i, j - 1, k + 1) - py1 * vy.at(i, j - 1, k), a4 = py2 * vy.at(i, j, k + 1) + py1 * vy.at(i, j, k);
                double a3 = -(a2 + a5 + a1 + a4);
                rhsRef.at(i, j, k) = a1 * vz.at(i, j - 1, k) + a2 * vz.at(i - 1, j, k) + a3 * vz.at(i, j, k)
                    + a4 * vz.at(i, j + 1, k) + a5 * vz.at(i + 1, j, k) + a6 * vz.at(i, j, k - 1) + a7 * vz.at(i, j, k + 1);
            }}}
    FieldView<double> vxv(vxU.data(), nx, ny1, nz1), vyv(vyU.data(), nx1, ny, nz1), vzv(vzU.data(), nx1, ny1, nz);
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k)
        mismatches.Check(rhsRef.at(i, j, k), FiniteDifferenceOperators::velocityZAdvection(i, j, k, vxv, vyv, vzv, sp.view()));
}

/// @brief Checks divergence against a ghost-padded reference.
void CheckDivergence(int nx1, int ny1, int nz1, const cfdtest::Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    cfdtest::PaddedField vx(nx1, ny2, nz2), vy(nx1 + 1, ny1, nz2), vz(nx1 + 1, ny2, nz1), rhsRef(nx1 + 1, ny1 + 1, nz1 + 1);
    vector<double> vxU((size_t)nx * ny1 * nz1), vyU((size_t)nx1 * ny * nz1), vzU((size_t)nx1 * ny1 * nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vx.at(i, j, k) = v; vxU[cfdtest::Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vy.at(i, j, k) = v; vyU[cfdtest::Uidx(i, j, k, nx1, ny)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vz.at(i, j, k) = v; vzU[cfdtest::Uidx(i, j, k, nx1, ny1)] = v; }
    cfdtest::FillVelGhostsZero(vx); cfdtest::FillVelGhostsZero(vy); cfdtest::FillVelGhostsZero(vz);

    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        rhsRef.at(i, j, k) = (vx.at(i, j, k) - vx.at(i - 1, j, k)) / sp.hx12[i - 1]
            + (vy.at(i, j, k) - vy.at(i, j - 1, k)) / sp.hy12[j - 1]
            + (vz.at(i, j, k) - vz.at(i, j, k - 1)) / sp.hz12[k - 1];
    FieldView<double> vxv(vxU.data(), nx, ny1, nz1), vyv(vyU.data(), nx1, ny, nz1), vzv(vzU.data(), nx1, ny1, nz);
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        mismatches.Check(rhsRef.at(i, j, k), FiniteDifferenceOperators::divergence(i, j, k, vxv, vyv, vzv, sp.view()));
}

/// @brief Checks pressureGradientX/Y/Z against a direct ghost-padded finite difference, for all three axes in one pass.
void CheckPressureGradient(int nx1, int ny1, int nz1, const cfdtest::Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1;
    vector<double> presU((size_t)nx1 * ny1 * nz1);
    cfdtest::PaddedField p(nx1 + 1, ny1 + 1, nz1 + 1);
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); p.at(i, j, k) = v; presU[cfdtest::Uidx(i, j, k, nx1, ny1)] = v; }
    FieldView<double> pv(presU.data(), nx1, ny1, nz1);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        mismatches.Check((p.at(i + 1, j, k) - p.at(i, j, k)) / sp.hpx[i], FiniteDifferenceOperators::pressureGradientX(i, j, k, pv, sp.view()));
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k)
        mismatches.Check((p.at(i, j + 1, k) - p.at(i, j, k)) / sp.hpy[j], FiniteDifferenceOperators::pressureGradientY(i, j, k, pv, sp.view()));
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k)
        mismatches.Check((p.at(i, j, k + 1) - p.at(i, j, k)) / sp.hpz[k], FiniteDifferenceOperators::pressureGradientZ(i, j, k, pv, sp.view()));
}

} // namespace

TEST(FiniteDifferenceOperatorsTest, TemperatureAdvection) {
    cfdtest::MismatchCollector mismatches("temperatureAdvection");
    BoundaryKind bcVals[] = {BoundaryKind::Dirichlet, BoundaryKind::Neumann};
    for (int nx1 : cfdtest::TestGridSizes()) for (int ny1 : cfdtest::TestGridSizes()) for (int nz1 : cfdtest::TestGridSizes()) {
        cfdtest::Sp sp(std::max({nx1, ny1, nz1}) + 2);
        for (auto bx : bcVals) for (auto by : bcVals) for (auto bz : bcVals)
            CheckTemperatureAdvection(nx1, ny1, nz1, bx, by, bz, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(FiniteDifferenceOperatorsTest, VelocityXAdvection) {
    cfdtest::MismatchCollector mismatches("velocityXAdvection");
    for (int nx1 : cfdtest::TestGridSizes()) for (int ny1 : cfdtest::TestGridSizes()) for (int nz1 : cfdtest::TestGridSizes()) {
        cfdtest::Sp sp(std::max({nx1, ny1, nz1}) + 2);
        CheckVelocityXAdvection(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(FiniteDifferenceOperatorsTest, VelocityYAdvection) {
    cfdtest::MismatchCollector mismatches("velocityYAdvection");
    for (int nx1 : cfdtest::TestGridSizes()) for (int ny1 : cfdtest::TestGridSizes()) for (int nz1 : cfdtest::TestGridSizes()) {
        cfdtest::Sp sp(std::max({nx1, ny1, nz1}) + 2);
        CheckVelocityYAdvection(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(FiniteDifferenceOperatorsTest, VelocityZAdvection) {
    cfdtest::MismatchCollector mismatches("velocityZAdvection");
    for (int nx1 : cfdtest::TestGridSizes()) for (int ny1 : cfdtest::TestGridSizes()) for (int nz1 : cfdtest::TestGridSizes()) {
        cfdtest::Sp sp(std::max({nx1, ny1, nz1}) + 2);
        CheckVelocityZAdvection(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(FiniteDifferenceOperatorsTest, Divergence) {
    cfdtest::MismatchCollector mismatches("divergence");
    for (int nx1 : cfdtest::TestGridSizes()) for (int ny1 : cfdtest::TestGridSizes()) for (int nz1 : cfdtest::TestGridSizes()) {
        cfdtest::Sp sp(std::max({nx1, ny1, nz1}) + 2);
        CheckDivergence(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(FiniteDifferenceOperatorsTest, PressureGradient) {
    cfdtest::MismatchCollector mismatches("pressureGradient");
    for (int nx1 : cfdtest::TestGridSizes()) for (int ny1 : cfdtest::TestGridSizes()) for (int nz1 : cfdtest::TestGridSizes()) {
        cfdtest::Sp sp(std::max({nx1, ny1, nz1}) + 2);
        CheckPressureGradient(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}
