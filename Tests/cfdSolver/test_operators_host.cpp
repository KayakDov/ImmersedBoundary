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
#include <algorithm>
#include <random>
#include <vector>

using std::vector;

namespace {

struct PaddedField {
    int nx, ny, nz;
    vector<double> data;
    PaddedField(int x, int y, int z) : nx(x), ny(y), nz(z), data((size_t)(x + 1) * (y + 1) * (z + 1), 0.0) {}
    double& at(int i, int j, int k) { return data[(size_t)i + (size_t)(nx + 1) * ((size_t)j + (size_t)(ny + 1) * (size_t)k)]; }
    double at(int i, int j, int k) const { return data[(size_t)i + (size_t)(nx + 1) * ((size_t)j + (size_t)(ny + 1) * (size_t)k)]; }
};

std::mt19937 rng(777);
std::uniform_real_distribution<double> dist(-1.0, 1.0);
double R() { return dist(rng); }

void FillTempGhosts(PaddedField& t, BoundaryKind bx, BoundaryKind by, BoundaryKind bz) {
    bool nx = bx == BoundaryKind::Neumann, ny = by == BoundaryKind::Neumann, nz = bz == BoundaryKind::Neumann;
    int nx1 = t.nx - 1, ny1 = t.ny - 1, nz1 = t.nz - 1;
    for (int j = 0; j <= t.ny; ++j) for (int k = 0; k <= t.nz; ++k) { t.at(0, j, k) = nx ? t.at(1, j, k) : 0.0; t.at(t.nx, j, k) = nx ? t.at(nx1, j, k) : 0.0; }
    for (int i = 0; i <= t.nx; ++i) for (int k = 0; k <= t.nz; ++k) { t.at(i, 0, k) = ny ? t.at(i, 1, k) : 0.0; t.at(i, t.ny, k) = ny ? t.at(i, ny1, k) : 0.0; }
    for (int i = 0; i <= t.nx; ++i) for (int j = 0; j <= t.ny; ++j) { t.at(i, j, 0) = nz ? t.at(i, j, 1) : 0.0; t.at(i, j, t.nz) = nz ? t.at(i, j, nz1) : 0.0; }
}

void FillVelGhostsZero(PaddedField& v) {
    for (int j = 0; j <= v.ny; ++j) for (int k = 0; k <= v.nz; ++k) { v.at(0, j, k) = 0.0; v.at(v.nx, j, k) = 0.0; }
    for (int i = 0; i <= v.nx; ++i) for (int k = 0; k <= v.nz; ++k) { v.at(i, 0, k) = 0.0; v.at(i, v.ny, k) = 0.0; }
    for (int i = 0; i <= v.nx; ++i) for (int j = 0; j <= v.ny; ++j) { v.at(i, j, 0) = 0.0; v.at(i, j, v.nz) = 0.0; }
}

struct Sp {
    vector<double> hx12, hy12, hz12, hpx, hpy, hpz;
    explicit Sp(int n) : hx12(n + 2), hy12(n + 2), hz12(n + 2), hpx(n + 2), hpy(n + 2), hpz(n + 2) {
        for (int i = 0; i < n + 2; ++i) {
            hx12[i] = 0.5 + 0.3 * std::sin(0.37 * i + 1.0); hy12[i] = 0.5 + 0.3 * std::sin(0.41 * i + 2.0); hz12[i] = 0.5 + 0.3 * std::sin(0.29 * i + 3.0);
            hpx[i] = 0.5 + 0.3 * std::sin(0.53 * i + 4.0);  hpy[i] = 0.5 + 0.3 * std::sin(0.61 * i + 5.0);  hpz[i] = 0.5 + 0.3 * std::sin(0.71 * i + 6.0);
        }
    }
    Spacing<double> view() const { return Spacing<double>{hx12.data(), hy12.data(), hz12.data(), hpx.data(), hpy.data(), hpz.data()}; }
};

size_t Uidx(int i, int j, int k, int nx, int ny) { return (size_t)(i - 1) + (size_t)nx * ((size_t)(j - 1) + (size_t)ny * (size_t)(k - 1)); }

const std::vector<int>& TestGridSizes() {
    static const std::vector<int> sizes = {5, 8, 13};
    return sizes;
}

void CheckTemperatureAdvection(int nx1, int ny1, int nz1, BoundaryKind bx, BoundaryKind by, BoundaryKind bz,
                                const Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    PaddedField tmpr(nx1 + 1, ny2, nz2), rhsRef(nx1 + 1, ny2, nz2);
    PaddedField vx(nx1, ny2, nz2), vy(nx1 + 1, ny1, nz2), vz(nx1 + 1, ny2, nz1);
    vector<double> tmprU((size_t)nx1 * ny1 * nz1);
    vector<double> vxU((size_t)nx * ny1 * nz1), vyU((size_t)nx1 * ny * nz1), vzU((size_t)nx1 * ny1 * nz);
    vector<double> buoyancyByX(nx1 + 2);
    for (auto& v : buoyancyByX) v = R();

    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double t = R(); tmpr.at(i, j, k) = t; tmprU[Uidx(i, j, k, nx1, ny1)] = t; }
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vx.at(i, j, k) = v; vxU[Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vy.at(i, j, k) = v; vyU[Uidx(i, j, k, nx1, ny)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vz.at(i, j, k) = v; vzU[Uidx(i, j, k, nx1, ny1)] = v; }
    FillTempGhosts(tmpr, bx, by, bz);
    FillVelGhostsZero(vx); FillVelGhostsZero(vy); FillVelGhostsZero(vz);

    // Reference: Teta read directly (NOT through ghost-fill) at every
    // absolute x index, exactly matching the real Fortran init loop's
    // direct-formula boundary coverage.
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

void CheckVelocityXAdvection(int nx1, int ny1, int nz1, const Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    PaddedField vx(nx1, ny2, nz2), vy(nx1 + 1, ny1, nz2), vz(nx1 + 1, ny2, nz1), rhsRef(nx1, ny2, nz2);
    vector<double> vxU((size_t)nx * ny1 * nz1), vyU((size_t)nx1 * ny * nz1), vzU((size_t)nx1 * ny1 * nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vx.at(i, j, k) = v; vxU[Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vy.at(i, j, k) = v; vyU[Uidx(i, j, k, nx1, ny)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vz.at(i, j, k) = v; vzU[Uidx(i, j, k, nx1, ny1)] = v; }
    FillVelGhostsZero(vx); FillVelGhostsZero(vy); FillVelGhostsZero(vz);

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

void CheckVelocityYAdvection(int nx1, int ny1, int nz1, const Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    PaddedField vx(nx1, ny2, nz2), vy(nx1 + 1, ny1, nz2), vz(nx1 + 1, ny2, nz1), rhsRef(nx1 + 1, ny1, nz2);
    vector<double> vxU((size_t)nx * ny1 * nz1), vyU((size_t)nx1 * ny * nz1), vzU((size_t)nx1 * ny1 * nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vx.at(i, j, k) = v; vxU[Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vy.at(i, j, k) = v; vyU[Uidx(i, j, k, nx1, ny)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vz.at(i, j, k) = v; vzU[Uidx(i, j, k, nx1, ny1)] = v; }
    FillVelGhostsZero(vx); FillVelGhostsZero(vy); FillVelGhostsZero(vz);

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

void CheckVelocityZAdvection(int nx1, int ny1, int nz1, const Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    PaddedField vx(nx1, ny2, nz2), vy(nx1 + 1, ny1, nz2), vz(nx1 + 1, ny2, nz1), rhsRef(nx1 + 1, ny2, nz1);
    vector<double> vxU((size_t)nx * ny1 * nz1), vyU((size_t)nx1 * ny * nz1), vzU((size_t)nx1 * ny1 * nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vx.at(i, j, k) = v; vxU[Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vy.at(i, j, k) = v; vyU[Uidx(i, j, k, nx1, ny)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vz.at(i, j, k) = v; vzU[Uidx(i, j, k, nx1, ny1)] = v; }
    FillVelGhostsZero(vx); FillVelGhostsZero(vy); FillVelGhostsZero(vz);

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

void CheckDivergence(int nx1, int ny1, int nz1, const Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    PaddedField vx(nx1, ny2, nz2), vy(nx1 + 1, ny1, nz2), vz(nx1 + 1, ny2, nz1), rhsRef(nx1 + 1, ny1 + 1, nz1 + 1);
    vector<double> vxU((size_t)nx * ny1 * nz1), vyU((size_t)nx1 * ny * nz1), vzU((size_t)nx1 * ny1 * nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vx.at(i, j, k) = v; vxU[Uidx(i, j, k, nx, ny1)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vy.at(i, j, k) = v; vyU[Uidx(i, j, k, nx1, ny)] = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vz.at(i, j, k) = v; vzU[Uidx(i, j, k, nx1, ny1)] = v; }
    FillVelGhostsZero(vx); FillVelGhostsZero(vy); FillVelGhostsZero(vz);

    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        rhsRef.at(i, j, k) = (vx.at(i, j, k) - vx.at(i - 1, j, k)) / sp.hx12[i - 1]
            + (vy.at(i, j, k) - vy.at(i, j - 1, k)) / sp.hy12[j - 1]
            + (vz.at(i, j, k) - vz.at(i, j, k - 1)) / sp.hz12[k - 1];
    FieldView<double> vxv(vxU.data(), nx, ny1, nz1), vyv(vyU.data(), nx1, ny, nz1), vzv(vzU.data(), nx1, ny1, nz);
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        mismatches.Check(rhsRef.at(i, j, k), FiniteDifferenceOperators::divergence(i, j, k, vxv, vyv, vzv, sp.view()));
}

void CheckPressureGradient(int nx1, int ny1, int nz1, const Sp& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1;
    vector<double> presU((size_t)nx1 * ny1 * nz1);
    PaddedField p(nx1 + 1, ny1 + 1, nz1 + 1);
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); p.at(i, j, k) = v; presU[Uidx(i, j, k, nx1, ny1)] = v; }
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
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Sp sp(std::max({nx1, ny1, nz1}) + 2);
        for (auto bx : bcVals) for (auto by : bcVals) for (auto bz : bcVals)
            CheckTemperatureAdvection(nx1, ny1, nz1, bx, by, bz, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(FiniteDifferenceOperatorsTest, VelocityXAdvection) {
    cfdtest::MismatchCollector mismatches("velocityXAdvection");
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Sp sp(std::max({nx1, ny1, nz1}) + 2);
        CheckVelocityXAdvection(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(FiniteDifferenceOperatorsTest, VelocityYAdvection) {
    cfdtest::MismatchCollector mismatches("velocityYAdvection");
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Sp sp(std::max({nx1, ny1, nz1}) + 2);
        CheckVelocityYAdvection(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(FiniteDifferenceOperatorsTest, VelocityZAdvection) {
    cfdtest::MismatchCollector mismatches("velocityZAdvection");
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Sp sp(std::max({nx1, ny1, nz1}) + 2);
        CheckVelocityZAdvection(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(FiniteDifferenceOperatorsTest, Divergence) {
    cfdtest::MismatchCollector mismatches("divergence");
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Sp sp(std::max({nx1, ny1, nz1}) + 2);
        CheckDivergence(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(FiniteDifferenceOperatorsTest, PressureGradient) {
    cfdtest::MismatchCollector mismatches("pressureGradient");
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Sp sp(std::max({nx1, ny1, nz1}) + 2);
        CheckPressureGradient(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}
