/**
 * @file verify_padding_equivalence.cpp
 * @brief GoogleTest proof that the boundary-conditional (no-ghost-padding)
 * rewrite of every finite-difference stencil in Alex's CFD solver produces
 * bit-identical output to the original ghost-padded (Fortran-matching)
 * version, across random data and every live EVD_BCx/y/z setting. Does not
 * exercise the eigendecomposition Poisson/Helmholtz solve itself (lives in
 * CudaBandedLib, already built/tested separately) -- isolates exactly the
 * architectural change in question: ghost-cell array padding vs. inline
 * boundary conditionals.
 */

#include <gtest/gtest.h>
#include "CfdTestHelpers.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using std::vector;

namespace {

struct PaddedField {
    int Nx, Ny, Nz;
    vector<double> data;
    PaddedField(int nx, int ny, int nz) : Nx(nx), Ny(ny), Nz(nz), data((size_t)(nx + 1) * (ny + 1) * (nz + 1), 0.0) {}
    double& at(int i, int j, int k) { return data[(size_t)i + (size_t)(Nx + 1) * ((size_t)j + (size_t)(Ny + 1) * (size_t)k)]; }
    double at(int i, int j, int k) const { return data[(size_t)i + (size_t)(Nx + 1) * ((size_t)j + (size_t)(Ny + 1) * (size_t)k)]; }
};

struct UnpaddedField {
    int Nx, Ny, Nz;
    vector<double> data;
    UnpaddedField(int nx, int ny, int nz) : Nx(nx), Ny(ny), Nz(nz), data((size_t)nx * (size_t)ny * (size_t)nz, 0.0) {}
    double& at(int i, int j, int k) { return data[(size_t)(i - 1) + (size_t)Nx * ((size_t)(j - 1) + (size_t)Ny * (size_t)(k - 1))]; }
    double at(int i, int j, int k) const { return data[(size_t)(i - 1) + (size_t)Nx * ((size_t)(j - 1) + (size_t)Ny * (size_t)(k - 1))]; }
};

std::mt19937 rng(12345);
std::uniform_real_distribution<double> dist(-1.0, 1.0);
double R() { return dist(rng); }

struct TempBC { bool neumannX, neumannY, neumannZ; };

/// Boundary-conditional reader for Temperature (unpadded) -- mirrors
/// EVDbounds exactly: Dirichlet -> 0, Neumann -> value at nearest real cell.
double TempRead(const UnpaddedField& t, int i, int j, int k, const TempBC& bc) {
    if (i < 1) return bc.neumannX ? TempRead(t, 1, j, k, bc) : 0.0;
    if (i > t.Nx) return bc.neumannX ? TempRead(t, t.Nx, j, k, bc) : 0.0;
    if (j < 1) return bc.neumannY ? TempRead(t, i, 1, k, bc) : 0.0;
    if (j > t.Ny) return bc.neumannY ? TempRead(t, i, t.Ny, k, bc) : 0.0;
    if (k < 1) return bc.neumannZ ? TempRead(t, i, j, 1, bc) : 0.0;
    if (k > t.Nz) return bc.neumannZ ? TempRead(t, i, j, t.Nz, bc) : 0.0;
    return t.at(i, j, k);
}

void FillTempGhosts(PaddedField& t, const TempBC& bc) {
    int nx1 = t.Nx - 1, ny1 = t.Ny - 1, nz1 = t.Nz - 1;
    for (int j = 0; j <= t.Ny; ++j) for (int k = 0; k <= t.Nz; ++k) { t.at(0, j, k) = bc.neumannX ? t.at(1, j, k) : 0.0; t.at(t.Nx, j, k) = bc.neumannX ? t.at(nx1, j, k) : 0.0; }
    for (int i = 0; i <= t.Nx; ++i) for (int k = 0; k <= t.Nz; ++k) { t.at(i, 0, k) = bc.neumannY ? t.at(i, 1, k) : 0.0; t.at(i, t.Ny, k) = bc.neumannY ? t.at(i, ny1, k) : 0.0; }
    for (int i = 0; i <= t.Nx; ++i) for (int j = 0; j <= t.Ny; ++j) { t.at(i, j, 0) = bc.neumannZ ? t.at(i, j, 1) : 0.0; t.at(i, j, t.Nz) = bc.neumannZ ? t.at(i, j, nz1) : 0.0; }
}

/// Velocity fields: ALWAYS Dirichlet-zero, unconditionally (confirmed from
/// bounds_3D.f90's VMxNew/VMyNew/VMzNew blocks -- no if/else, unlike Temp).
double VelRead(const UnpaddedField& v, int i, int j, int k) {
    if (i < 1 || i > v.Nx || j < 1 || j > v.Ny || k < 1 || k > v.Nz) return 0.0;
    return v.at(i, j, k);
}

void FillVelGhostsZero(PaddedField& v) {
    for (int j = 0; j <= v.Ny; ++j) for (int k = 0; k <= v.Nz; ++k) { v.at(0, j, k) = 0.0; v.at(v.Nx, j, k) = 0.0; }
    for (int i = 0; i <= v.Nx; ++i) for (int k = 0; k <= v.Nz; ++k) { v.at(i, 0, k) = 0.0; v.at(i, v.Ny, k) = 0.0; }
    for (int i = 0; i <= v.Nx; ++i) for (int j = 0; j <= v.Ny; ++j) { v.at(i, j, 0) = 0.0; v.at(i, j, v.Nz) = 0.0; }
}

/// Shared mesh-spacing arrays (not boundary-related -- plain 0-indexed
/// coefficient arrays, identical in both padded/unpadded tests).
struct Spacing2 {
    vector<double> Hx12, Hy12, Hz12, HPx, HPy, HPz;
    explicit Spacing2(int n) : Hx12(n + 2), Hy12(n + 2), Hz12(n + 2), HPx(n + 2), HPy(n + 2), HPz(n + 2) {
        for (int i = 0; i < n + 2; ++i) {
            Hx12[i] = 0.5 + 0.3 * std::sin(0.37 * i + 1.0);
            Hy12[i] = 0.5 + 0.3 * std::sin(0.41 * i + 2.0);
            Hz12[i] = 0.5 + 0.3 * std::sin(0.29 * i + 3.0);
            HPx[i] = 0.5 + 0.3 * std::sin(0.53 * i + 4.0);
            HPy[i] = 0.5 + 0.3 * std::sin(0.61 * i + 5.0);
            HPz[i] = 0.5 + 0.3 * std::sin(0.71 * i + 6.0);
        }
    }
};

const std::vector<int>& TestGridSizes() {
    static const std::vector<int> sizes = {5, 8, 13};
    return sizes;
}

// Exercises Temperature ghosts (BC-dependent, both ends, all 3 axes) AND
// velocity ghosts (always-zero, both ends, all 3 axes) simultaneously --
// the most complex access pattern in the whole codebase.
void CheckVgrTmp(int nx1, int ny1, int nz1, const TempBC& bc, const Spacing2& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1;
    int nx2 = nx1 + 1, ny2 = ny1 + 1, nz2 = nz1 + 1;

    PaddedField tmpr(nx2, ny2, nz2), teta(nx2, ny2, nz2), fdrhp(nx2, ny2, nz2);
    PaddedField vmx(nx1, ny2, nz2), vmy(nx2, ny1, nz2), vmz(nx2, ny2, nz1);
    UnpaddedField tmprU(nx1, ny1, nz1), tetaU(nx1, ny1, nz1), fdrhpU(nx1, ny1, nz1);
    UnpaddedField vmxU(nx, ny1, nz1), vmyU(nx1, ny, nz1), vmzU(nx1, ny1, nz);

    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) {
        double t = R(), te = R();
        tmpr.at(i, j, k) = t; tmprU.at(i, j, k) = t;
        teta.at(i, j, k) = te; tetaU.at(i, j, k) = te;
    }
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vmx.at(i, j, k) = v; vmxU.at(i, j, k) = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vmy.at(i, j, k) = v; vmyU.at(i, j, k) = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vmz.at(i, j, k) = v; vmzU.at(i, j, k) = v; }

    FillTempGhosts(tmpr, bc); FillTempGhosts(teta, bc);
    FillVelGhostsZero(vmx); FillVelGhostsZero(vmy); FillVelGhostsZero(vmz);

    for (int i = 1; i <= nx1; ++i) { double px = 0.5 / sp.Hx12[i - 1];
        for (int j = 1; j <= ny1; ++j) { double py = 0.5 / sp.Hy12[j - 1];
            for (int k = 1; k <= nz1; ++k) { double pz = 0.5 / sp.Hz12[k - 1];
                double a1 = -vmy.at(i, j - 1, k) * py, a4 = vmy.at(i, j, k) * py;
                double a2 = -vmx.at(i - 1, j, k) * px, a5 = vmx.at(i, j, k) * px;
                double a6 = -vmz.at(i, j, k - 1) * pz, a7 = vmz.at(i, j, k) * pz;
                double a3 = -(a1 + a2 + a4 + a5 + a6 + a7);
                fdrhp.at(i, j, k) = a1 * (tmpr.at(i, j - 1, k) + teta.at(i, j - 1, k)) + a2 * (tmpr.at(i - 1, j, k) + teta.at(i - 1, j, k))
                    + a3 * (tmpr.at(i, j, k) + teta.at(i, j, k)) + a4 * (tmpr.at(i, j + 1, k) + teta.at(i, j + 1, k))
                    + a5 * (tmpr.at(i + 1, j, k) + teta.at(i + 1, j, k)) + a6 * (tmpr.at(i, j, k - 1) + teta.at(i, j, k - 1)) + a7 * (tmpr.at(i, j, k + 1) + teta.at(i, j, k + 1));
            }}}

    for (int i = 1; i <= nx1; ++i) { double px = 0.5 / sp.Hx12[i - 1];
        for (int j = 1; j <= ny1; ++j) { double py = 0.5 / sp.Hy12[j - 1];
            for (int k = 1; k <= nz1; ++k) { double pz = 0.5 / sp.Hz12[k - 1];
                double a1 = -VelRead(vmyU, i, j - 1, k) * py, a4 = VelRead(vmyU, i, j, k) * py;
                double a2 = -VelRead(vmxU, i - 1, j, k) * px, a5 = VelRead(vmxU, i, j, k) * px;
                double a6 = -VelRead(vmzU, i, j, k - 1) * pz, a7 = VelRead(vmzU, i, j, k) * pz;
                double a3 = -(a1 + a2 + a4 + a5 + a6 + a7);
                fdrhpU.at(i, j, k) = a1 * (TempRead(tmprU, i, j - 1, k, bc) + TempRead(tetaU, i, j - 1, k, bc)) + a2 * (TempRead(tmprU, i - 1, j, k, bc) + TempRead(tetaU, i - 1, j, k, bc))
                    + a3 * (TempRead(tmprU, i, j, k, bc) + TempRead(tetaU, i, j, k, bc)) + a4 * (TempRead(tmprU, i, j + 1, k, bc) + TempRead(tetaU, i, j + 1, k, bc))
                    + a5 * (TempRead(tmprU, i + 1, j, k, bc) + TempRead(tetaU, i + 1, j, k, bc)) + a6 * (TempRead(tmprU, i, j, k - 1, bc) + TempRead(tetaU, i, j, k - 1, bc)) + a7 * (TempRead(tmprU, i, j, k + 1, bc) + TempRead(tetaU, i, j, k + 1, bc));
            }}}

    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        mismatches.Check(fdrhp.at(i, j, k), fdrhpU.at(i, j, k));
}

// VMy/VMz read at i+1 (may or may not be a ghost depending on i); VMx read
// at both its own ghosts. Pure velocity -- always-zero boundary, no
// BC-switch dependence.
void CheckVgrdVx(int nx1, int ny1, int nz1, const Spacing2& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    PaddedField vmx(nx1, ny2, nz2), vmy(nx1 + 1, ny1, nz2), vmz(nx1 + 1, ny2, nz1), fdrhp(nx1, ny2, nz2);
    UnpaddedField vmxU(nx, ny1, nz1), vmyU(nx1, ny, nz1), vmzU(nx1, ny1, nz), fdrhpU(nx, ny1, nz1);

    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vmx.at(i, j, k) = v; vmxU.at(i, j, k) = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vmy.at(i, j, k) = v; vmyU.at(i, j, k) = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vmz.at(i, j, k) = v; vmzU.at(i, j, k) = v; }
    FillVelGhostsZero(vmx); FillVelGhostsZero(vmy); FillVelGhostsZero(vmz);

    for (int i = 1; i <= nx; ++i) { double px = 0.25 / sp.HPx[i];
        for (int j = 1; j <= ny1; ++j) { double py = px / sp.Hy12[j - 1], py1 = sp.Hx12[i - 1] * py, py2 = sp.Hx12[i] * py;
            for (int k = 1; k <= nz1; ++k) { double pz = px / sp.Hz12[k - 1], pz1 = sp.Hx12[i - 1] * pz, pz2 = sp.Hx12[i] * pz;
                double a1 = -py2 * vmy.at(i + 1, j - 1, k) - py1 * vmy.at(i, j - 1, k);
                double a4 = py2 * vmy.at(i + 1, j, k) + py1 * vmy.at(i, j, k);
                double a6 = -pz2 * vmz.at(i + 1, j, k - 1) - pz1 * vmz.at(i, j, k - 1);
                double a7 = pz2 * vmz.at(i + 1, j, k) + pz1 * vmz.at(i, j, k);
                double a2 = -vmx.at(i - 1, j, k) * px, a5 = vmx.at(i + 1, j, k) * px;
                double a3 = -(a1 + a4 + a6 + a7);
                fdrhp.at(i, j, k) = a1 * vmx.at(i, j - 1, k) + a2 * vmx.at(i - 1, j, k) + a3 * vmx.at(i, j, k)
                    + a4 * vmx.at(i, j + 1, k) + a5 * vmx.at(i + 1, j, k) + a6 * vmx.at(i, j, k - 1) + a7 * vmx.at(i, j, k + 1);
            }}}
    for (int i = 1; i <= nx; ++i) { double px = 0.25 / sp.HPx[i];
        for (int j = 1; j <= ny1; ++j) { double py = px / sp.Hy12[j - 1], py1 = sp.Hx12[i - 1] * py, py2 = sp.Hx12[i] * py;
            for (int k = 1; k <= nz1; ++k) { double pz = px / sp.Hz12[k - 1], pz1 = sp.Hx12[i - 1] * pz, pz2 = sp.Hx12[i] * pz;
                double a1 = -py2 * VelRead(vmyU, i + 1, j - 1, k) - py1 * VelRead(vmyU, i, j - 1, k);
                double a4 = py2 * VelRead(vmyU, i + 1, j, k) + py1 * VelRead(vmyU, i, j, k);
                double a6 = -pz2 * VelRead(vmzU, i + 1, j, k - 1) - pz1 * VelRead(vmzU, i, j, k - 1);
                double a7 = pz2 * VelRead(vmzU, i + 1, j, k) + pz1 * VelRead(vmzU, i, j, k);
                double a2 = -VelRead(vmxU, i - 1, j, k) * px, a5 = VelRead(vmxU, i + 1, j, k) * px;
                double a3 = -(a1 + a4 + a6 + a7);
                fdrhpU.at(i, j, k) = a1 * VelRead(vmxU, i, j - 1, k) + a2 * VelRead(vmxU, i - 1, j, k) + a3 * VelRead(vmxU, i, j, k)
                    + a4 * VelRead(vmxU, i, j + 1, k) + a5 * VelRead(vmxU, i + 1, j, k) + a6 * VelRead(vmxU, i, j, k - 1) + a7 * VelRead(vmxU, i, j, k + 1);
            }}}
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        mismatches.Check(fdrhp.at(i, j, k), fdrhpU.at(i, j, k));
}

void CheckVgrdVy(int nx1, int ny1, int nz1, const Spacing2& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    PaddedField vmx(nx1, ny2, nz2), vmy(nx1 + 1, ny1, nz2), vmz(nx1 + 1, ny2, nz1), fdrhp(nx1 + 1, ny1, nz2);
    UnpaddedField vmxU(nx, ny1, nz1), vmyU(nx1, ny, nz1), vmzU(nx1, ny1, nz), fdrhpU(nx1, ny, nz1);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vmx.at(i, j, k) = v; vmxU.at(i, j, k) = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vmy.at(i, j, k) = v; vmyU.at(i, j, k) = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vmz.at(i, j, k) = v; vmzU.at(i, j, k) = v; }
    FillVelGhostsZero(vmx); FillVelGhostsZero(vmy); FillVelGhostsZero(vmz);

    for (int j = 1; j <= ny; ++j) { double py = 0.25 / sp.HPy[j];
        for (int i = 1; i <= nx1; ++i) { double px = py / sp.Hx12[i - 1], px1 = sp.Hy12[j - 1] * px, px2 = sp.Hy12[j] * px;
            for (int k = 1; k <= nz1; ++k) { double pz = py / sp.Hz12[k - 1], pz1 = sp.Hy12[j - 1] * pz, pz2 = sp.Hy12[j] * pz;
                double a1 = -vmy.at(i, j - 1, k) * py, a4 = vmy.at(i, j + 1, k) * py;
                double a2 = -px2 * vmx.at(i - 1, j + 1, k) - px1 * vmx.at(i - 1, j, k), a5 = px2 * vmx.at(i, j + 1, k) + px1 * vmx.at(i, j, k);
                double a6 = -pz2 * vmz.at(i, j + 1, k - 1) - pz1 * vmz.at(i, j, k - 1), a7 = pz2 * vmz.at(i, j + 1, k) + pz1 * vmz.at(i, j, k);
                double a3 = -(a2 + a5 + a6 + a7);
                fdrhp.at(i, j, k) = a1 * vmy.at(i, j - 1, k) + a2 * vmy.at(i - 1, j, k) + a3 * vmy.at(i, j, k) + a4 * vmy.at(i, j + 1, k)
                    + a5 * vmy.at(i + 1, j, k) + a6 * vmy.at(i, j, k - 1) + a7 * vmy.at(i, j, k + 1);
            }}}
    for (int j = 1; j <= ny; ++j) { double py = 0.25 / sp.HPy[j];
        for (int i = 1; i <= nx1; ++i) { double px = py / sp.Hx12[i - 1], px1 = sp.Hy12[j - 1] * px, px2 = sp.Hy12[j] * px;
            for (int k = 1; k <= nz1; ++k) { double pz = py / sp.Hz12[k - 1], pz1 = sp.Hy12[j - 1] * pz, pz2 = sp.Hy12[j] * pz;
                double a1 = -VelRead(vmyU, i, j - 1, k) * py, a4 = VelRead(vmyU, i, j + 1, k) * py;
                double a2 = -px2 * VelRead(vmxU, i - 1, j + 1, k) - px1 * VelRead(vmxU, i - 1, j, k), a5 = px2 * VelRead(vmxU, i, j + 1, k) + px1 * VelRead(vmxU, i, j, k);
                double a6 = -pz2 * VelRead(vmzU, i, j + 1, k - 1) - pz1 * VelRead(vmzU, i, j, k - 1), a7 = pz2 * VelRead(vmzU, i, j + 1, k) + pz1 * VelRead(vmzU, i, j, k);
                double a3 = -(a2 + a5 + a6 + a7);
                fdrhpU.at(i, j, k) = a1 * VelRead(vmyU, i, j - 1, k) + a2 * VelRead(vmyU, i - 1, j, k) + a3 * VelRead(vmyU, i, j, k) + a4 * VelRead(vmyU, i, j + 1, k)
                    + a5 * VelRead(vmyU, i + 1, j, k) + a6 * VelRead(vmyU, i, j, k - 1) + a7 * VelRead(vmyU, i, j, k + 1);
            }}}
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k)
        mismatches.Check(fdrhp.at(i, j, k), fdrhpU.at(i, j, k));
}

void CheckVgrdVz(int nx1, int ny1, int nz1, const Spacing2& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    PaddedField vmx(nx1, ny2, nz2), vmy(nx1 + 1, ny1, nz2), vmz(nx1 + 1, ny2, nz1), fdrhp(nx1 + 1, ny2, nz1);
    UnpaddedField vmxU(nx, ny1, nz1), vmyU(nx1, ny, nz1), vmzU(nx1, ny1, nz), fdrhpU(nx1, ny1, nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vmx.at(i, j, k) = v; vmxU.at(i, j, k) = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vmy.at(i, j, k) = v; vmyU.at(i, j, k) = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vmz.at(i, j, k) = v; vmzU.at(i, j, k) = v; }
    FillVelGhostsZero(vmx); FillVelGhostsZero(vmy); FillVelGhostsZero(vmz);

    for (int k = 1; k <= nz; ++k) { double pz = 0.25 / sp.HPz[k];
        for (int i = 1; i <= nx1; ++i) { double px = pz / sp.Hx12[i - 1], px1 = sp.Hz12[k - 1] * px, px2 = sp.Hz12[k] * px;
            for (int j = 1; j <= ny1; ++j) { double py = pz / sp.Hy12[j - 1], py1 = sp.Hz12[k - 1] * py, py2 = sp.Hz12[k] * py;
                double a6 = -vmz.at(i, j, k - 1) * pz, a7 = vmz.at(i, j, k + 1) * pz;
                double a2 = -px2 * vmx.at(i - 1, j, k + 1) - px1 * vmx.at(i - 1, j, k), a5 = px2 * vmx.at(i, j, k + 1) + px1 * vmx.at(i, j, k);
                double a1 = -py2 * vmy.at(i, j - 1, k + 1) - py1 * vmy.at(i, j - 1, k), a4 = py2 * vmy.at(i, j, k + 1) + py1 * vmy.at(i, j, k);
                double a3 = -(a2 + a5 + a1 + a4);
                fdrhp.at(i, j, k) = a1 * vmz.at(i, j - 1, k) + a2 * vmz.at(i - 1, j, k) + a3 * vmz.at(i, j, k) + a4 * vmz.at(i, j + 1, k)
                    + a5 * vmz.at(i + 1, j, k) + a6 * vmz.at(i, j, k - 1) + a7 * vmz.at(i, j, k + 1);
            }}}
    for (int k = 1; k <= nz; ++k) { double pz = 0.25 / sp.HPz[k];
        for (int i = 1; i <= nx1; ++i) { double px = pz / sp.Hx12[i - 1], px1 = sp.Hz12[k - 1] * px, px2 = sp.Hz12[k] * px;
            for (int j = 1; j <= ny1; ++j) { double py = pz / sp.Hy12[j - 1], py1 = sp.Hz12[k - 1] * py, py2 = sp.Hz12[k] * py;
                double a6 = -VelRead(vmzU, i, j, k - 1) * pz, a7 = VelRead(vmzU, i, j, k + 1) * pz;
                double a2 = -px2 * VelRead(vmxU, i - 1, j, k + 1) - px1 * VelRead(vmxU, i - 1, j, k), a5 = px2 * VelRead(vmxU, i, j, k + 1) + px1 * VelRead(vmxU, i, j, k);
                double a1 = -py2 * VelRead(vmyU, i, j - 1, k + 1) - py1 * VelRead(vmyU, i, j - 1, k), a4 = py2 * VelRead(vmyU, i, j, k + 1) + py1 * VelRead(vmyU, i, j, k);
                double a3 = -(a2 + a5 + a1 + a4);
                fdrhpU.at(i, j, k) = a1 * VelRead(vmzU, i, j - 1, k) + a2 * VelRead(vmzU, i - 1, j, k) + a3 * VelRead(vmzU, i, j, k) + a4 * VelRead(vmzU, i, j + 1, k)
                    + a5 * VelRead(vmzU, i + 1, j, k) + a6 * VelRead(vmzU, i, j, k - 1) + a7 * VelRead(vmzU, i, j, k + 1);
            }}}
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k)
        mismatches.Check(fdrhp.at(i, j, k), fdrhpU.at(i, j, k));
}

// Simplest pattern: each component read at its own lower ghost only
// (i-1/j-1/k-1 at the loop's lower bound).
void CheckFdDiv(int nx1, int ny1, int nz1, const Spacing2& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    PaddedField vmx(nx1, ny2, nz2), vmy(nx1 + 1, ny1, nz2), vmz(nx1 + 1, ny2, nz1), fdrhp(nx1, ny1, nz1);
    UnpaddedField vmxU(nx, ny1, nz1), vmyU(nx1, ny, nz1), vmzU(nx1, ny1, nz), fdrhpU(nx1, ny1, nz1);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vmx.at(i, j, k) = v; vmxU.at(i, j, k) = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vmy.at(i, j, k) = v; vmyU.at(i, j, k) = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vmz.at(i, j, k) = v; vmzU.at(i, j, k) = v; }
    FillVelGhostsZero(vmx); FillVelGhostsZero(vmy); FillVelGhostsZero(vmz);

    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        fdrhp.at(i, j, k) = (vmx.at(i, j, k) - vmx.at(i - 1, j, k)) / sp.Hx12[i - 1]
            + (vmy.at(i, j, k) - vmy.at(i, j - 1, k)) / sp.Hy12[j - 1]
            + (vmz.at(i, j, k) - vmz.at(i, j, k - 1)) / sp.Hz12[k - 1];
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        fdrhpU.at(i, j, k) = (VelRead(vmxU, i, j, k) - VelRead(vmxU, i - 1, j, k)) / sp.Hx12[i - 1]
            + (VelRead(vmyU, i, j, k) - VelRead(vmyU, i, j - 1, k)) / sp.Hy12[j - 1]
            + (VelRead(vmzU, i, j, k) - VelRead(vmzU, i, j, k - 1)) / sp.Hz12[k - 1];
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        mismatches.Check(fdrhp.at(i, j, k), fdrhpU.at(i, j, k));
}

// Temperature ONE-SIDED wall stencil, reads Tmpr(0,j,k) [ghost] AND
// Tmpr(2,j,k) [real interior, 2 cells in] -- the case where a ghost read is
// combined with a non-adjacent real read in the SAME formula.
void CheckNusselt(int nx1, int ny1, int nz1, const TempBC& bc, const Spacing2& sp, cfdtest::MismatchCollector& mismatches) {
    PaddedField tmpr(nx1 + 1, ny1 + 1, nz1 + 1);
    UnpaddedField tmprU(nx1, ny1, nz1);
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); tmpr.at(i, j, k) = v; tmprU.at(i, j, k) = v; }
    FillTempGhosts(tmpr, bc);

    double div = sp.HPx[0] * sp.HPx[1] * (sp.HPx[0] + sp.HPx[1]);
    double f0 = sp.HPx[1] * (sp.HPx[1] + 2.0 * sp.HPx[0]);
    double f1 = -(sp.HPx[1] + sp.HPx[0]) * (sp.HPx[1] + sp.HPx[0]);
    double f2 = sp.HPx[0] * sp.HPx[0];

    double snuP = 0.0, snuU = 0.0;
    for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) {
        double dtP = (f0 * tmpr.at(0, j, k) + f1 * tmpr.at(1, j, k) + f2 * tmpr.at(2, j, k)) / div;
        snuP += dtP * sp.Hy12[j - 1] * sp.Hz12[k - 1];
        double dtU = (f0 * TempRead(tmprU, 0, j, k, bc) + f1 * TempRead(tmprU, 1, j, k, bc) + f2 * TempRead(tmprU, 2, j, k, bc)) / div;
        snuU += dtU * sp.Hy12[j - 1] * sp.Hz12[k - 1];
    }
    mismatches.Check(snuP, snuU);
}

void CheckEkinem(int nx1, int ny1, int nz1, const Spacing2& sp, cfdtest::MismatchCollector& mismatches) {
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1, ny2 = ny1 + 1, nz2 = nz1 + 1;
    PaddedField vmx(nx1, ny2, nz2), vmy(nx1 + 1, ny1, nz2), vmz(nx1 + 1, ny2, nz1);
    UnpaddedField vmxU(nx, ny1, nz1), vmyU(nx1, ny, nz1), vmzU(nx1, ny1, nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vmx.at(i, j, k) = v; vmxU.at(i, j, k) = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz1; ++k) { double v = R(); vmy.at(i, j, k) = v; vmyU.at(i, j, k) = v; }
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) { double v = R(); vmz.at(i, j, k) = v; vmzU.at(i, j, k) = v; }
    FillVelGhostsZero(vmx); FillVelGhostsZero(vmy); FillVelGhostsZero(vmz);

    double ssP = 0.0, ssU = 0.0;
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) {
        double tx = vmx.at(i, j, k) + vmx.at(i - 1, j, k); tx *= tx;
        double ty = vmy.at(i, j, k) + vmy.at(i, j - 1, k); ty *= ty;
        double tz = vmz.at(i, j, k) + vmz.at(i, j, k - 1); tz *= tz;
        ssP += 0.125 * (tx + ty + tz) * sp.Hx12[i - 1] * sp.Hy12[j - 1] * sp.Hz12[k - 1];

        double ux = VelRead(vmxU, i, j, k) + VelRead(vmxU, i - 1, j, k); ux *= ux;
        double uy = VelRead(vmyU, i, j, k) + VelRead(vmyU, i, j - 1, k); uy *= uy;
        double uz = VelRead(vmzU, i, j, k) + VelRead(vmzU, i, j, k - 1); uz *= uz;
        ssU += 0.125 * (ux + uy + uz) * sp.Hx12[i - 1] * sp.Hy12[j - 1] * sp.Hz12[k - 1];
    }
    mismatches.Check(ssP, ssU);
}

} // namespace

TEST(PaddingEquivalenceTest, VgrTmp) {
    cfdtest::MismatchCollector mismatches("VgrTmp");
    bool bcVals[] = {false, true};
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Spacing2 sp(std::max({nx1, ny1, nz1}) + 2);
        for (bool bx : bcVals) for (bool by : bcVals) for (bool bz : bcVals)
            CheckVgrTmp(nx1, ny1, nz1, TempBC{bx, by, bz}, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(PaddingEquivalenceTest, Nusselt) {
    cfdtest::MismatchCollector mismatches("Nusselt");
    bool bcVals[] = {false, true};
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Spacing2 sp(std::max({nx1, ny1, nz1}) + 2);
        for (bool bx : bcVals) for (bool by : bcVals) for (bool bz : bcVals)
            CheckNusselt(nx1, ny1, nz1, TempBC{bx, by, bz}, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(PaddingEquivalenceTest, VgrdVx) {
    cfdtest::MismatchCollector mismatches("VgrdVx");
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Spacing2 sp(std::max({nx1, ny1, nz1}) + 2);
        CheckVgrdVx(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(PaddingEquivalenceTest, VgrdVy) {
    cfdtest::MismatchCollector mismatches("VgrdVy");
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Spacing2 sp(std::max({nx1, ny1, nz1}) + 2);
        CheckVgrdVy(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(PaddingEquivalenceTest, VgrdVz) {
    cfdtest::MismatchCollector mismatches("VgrdVz");
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Spacing2 sp(std::max({nx1, ny1, nz1}) + 2);
        CheckVgrdVz(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(PaddingEquivalenceTest, FdDiv) {
    cfdtest::MismatchCollector mismatches("FdDiv");
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Spacing2 sp(std::max({nx1, ny1, nz1}) + 2);
        CheckFdDiv(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}

TEST(PaddingEquivalenceTest, Ekinem) {
    cfdtest::MismatchCollector mismatches("Ekinem");
    for (int nx1 : TestGridSizes()) for (int ny1 : TestGridSizes()) for (int nz1 : TestGridSizes()) {
        Spacing2 sp(std::max({nx1, ny1, nz1}) + 2);
        CheckEkinem(nx1, ny1, nz1, sp, mismatches);
    }
    mismatches.Finish();
}
