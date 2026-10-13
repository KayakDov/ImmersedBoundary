/**
 * @file test_diagnostics.cpp
 * @brief GoogleTest verification of Diagnostics (kinetic energy, Nusselt,
 * Nusselt_middle, Dist2D's relative-difference reduction) against the
 * exact formulas in ekinem_3D.f90 and sublid_3D.f90.
 */

#include <gtest/gtest.h>
#include "physics/Diagnostics.cuh"
#include "CfdTestHelpers.h"
#include <functional>
#include <random>
#include <vector>

namespace {

std::mt19937 rng(42);
std::uniform_real_distribution<double> dist(-1.0, 1.0);
double R() { return dist(rng); }

double VelocityAt(const std::vector<double>& d, int i, int j, int k, int nx, int ny, int nz) {
    if (i < 1 || i > nx || j < 1 || j > ny || k < 1 || k > nz) return 0.0;
    return d[(size_t)(i - 1) + (size_t)nx * ((size_t)(j - 1) + (size_t)ny * (size_t)(k - 1))];
}

/// Shared grid extents and mesh spacing for every sub-test below -- matches
/// the single Spacing<double> object the original test built once in main().
class DiagnosticsTest : public ::testing::Test {
protected:
    int nx1 = 7, ny1 = 8, nz1 = 6;
    int nx = nx1 - 1, ny = ny1 - 1, nz = nz1 - 1;
    std::vector<double> hx12, hy12, hz12, hpx, hpy, hpz;

    void SetUp() override {
        hx12.resize(20); hy12.resize(20); hz12.resize(20);
        hpx.resize(20); hpy.resize(20); hpz.resize(20);
        for (int i = 0; i < 20; ++i) {
            hx12[i] = 0.3 + 0.1 * i; hy12[i] = 0.4 + 0.1 * i; hz12[i] = 0.5 + 0.1 * i;
            hpx[i] = 0.6 + 0.1 * i;  hpy[i] = 0.7 + 0.1 * i;  hpz[i] = 0.8 + 0.1 * i;
        }
    }

    Spacing<double> spacing() const {
        return Spacing<double>{hx12.data(), hy12.data(), hz12.data(), hpx.data(), hpy.data(), hpz.data()};
    }
};

} // namespace

TEST_F(DiagnosticsTest, KineticEnergyMatchesEkinemFormula) {
    std::vector<double> vxData((size_t)nx * ny1 * nz1), vyData((size_t)nx1 * ny * nz1), vzData((size_t)nx1 * ny1 * nz);
    for (auto& v : vxData) v = R();
    for (auto& v : vyData) v = R();
    for (auto& v : vzData) v = R();

    double referenceSum = 0.0;
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) {
        double sx = VelocityAt(vxData, i, j, k, nx, ny1, nz1) + VelocityAt(vxData, i - 1, j, k, nx, ny1, nz1);
        double sy = VelocityAt(vyData, i, j, k, nx1, ny, nz1) + VelocityAt(vyData, i, j - 1, k, nx1, ny, nz1);
        double sz = VelocityAt(vzData, i, j, k, nx1, ny1, nz) + VelocityAt(vzData, i, j, k - 1, nx1, ny1, nz);
        referenceSum += 0.125 * (sx * sx + sy * sy + sz * sz) * hx12[i - 1] * hy12[j - 1] * hz12[k - 1];
    }

    FieldView<double> vx(vxData.data(), nx, ny1, nz1), vy(vyData.data(), nx1, ny, nz1), vz(vzData.data(), nx1, ny1, nz);
    double implSum = 0.0;
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        implSum += Diagnostics::kineticEnergyCell(i, j, k, vx, vy, vz, spacing());

    EXPECT_TRUE(cfdtest::IsClose(referenceSum, implSum)) << "reference=" << referenceSum << " actual=" << implSum;
}

TEST_F(DiagnosticsTest, NusseltMatchesFormulaAndFinalCombination) {
    std::vector<double> tmprData((size_t)nx1 * ny1 * nz1);
    for (auto& v : tmprData) v = R();
    FieldView<double> tmpr(tmprData.data(), nx1, ny1, nz1);
    AxisBoundaries bc{BoundaryKind::Dirichlet, BoundaryKind::Neumann, BoundaryKind::Neumann};

    std::function<double(int, int, int)> tempAt = [&](int i, int j, int k) -> double {
        BoundaryKind kinds[3] = {bc.x, bc.y, bc.z};
        int idxs[3] = {i, j, k};
        int lo[3] = {1, 1, 1}, hi[3] = {nx1, ny1, nz1};
        for (int ax = 0; ax < 3; ++ax) {
            if (idxs[ax] < lo[ax] || idxs[ax] > hi[ax]) {
                if (kinds[ax] == BoundaryKind::Dirichlet) return 0.0;
                int mirrored[3] = {i, j, k};
                mirrored[ax] = idxs[ax] < lo[ax] ? lo[ax] : hi[ax];
                return tempAt(mirrored[0], mirrored[1], mirrored[2]);
            }
        }
        return tmprData[(size_t)(i - 1) + (size_t)nx1 * ((size_t)(j - 1) + (size_t)ny1 * (size_t)(k - 1))];
    };

    double div = hpx[0] * hpx[1] * (hpx[0] + hpx[1]);
    double f0 = hpx[1] * (hpx[1] + 2.0 * hpx[0]);
    double f1 = -(hpx[1] + hpx[0]) * (hpx[1] + hpx[0]);
    double f2 = hpx[0] * hpx[0];

    double referenceSum = 0.0;
    for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) {
        double dT = (f0 * tempAt(0, j, k) + f1 * tempAt(1, j, k) + f2 * tempAt(2, j, k)) / div;
        referenceSum += dT * hy12[j - 1] * hz12[k - 1];
    }

    double implSum = 0.0;
    for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        implSum += Diagnostics::nusseltCell(j, k, tmpr, bc, spacing());

    EXPECT_TRUE(cfdtest::IsClose(referenceSum, implSum)) << "reference=" << referenceSum << " actual=" << implSum;

    double aspectRatio = 1.3, widthRatio = 0.9;
    double referenceNusselt = 1.0 / aspectRatio + referenceSum / widthRatio;
    double implNusselt = Diagnostics::nusseltFromSum(implSum, aspectRatio, widthRatio);
    EXPECT_TRUE(cfdtest::IsClose(referenceNusselt, implNusselt)) << "reference=" << referenceNusselt << " actual=" << implNusselt;
}

TEST_F(DiagnosticsTest, NusseltMiddleMatchesFormulaAndFinalCombination) {
    std::vector<double> tmprData((size_t)nx1 * ny1 * nz1);
    for (auto& v : tmprData) v = R();
    FieldView<double> tmpr(tmprData.data(), nx1, ny1, nz1);
    AxisBoundaries bc{BoundaryKind::Dirichlet, BoundaryKind::Neumann, BoundaryKind::Neumann};

    std::function<double(int, int, int)> tempAt = [&](int i, int j, int k) -> double {
        BoundaryKind kinds[3] = {bc.x, bc.y, bc.z};
        int idxs[3] = {i, j, k};
        int lo[3] = {1, 1, 1}, hi[3] = {nx1, ny1, nz1};
        for (int ax = 0; ax < 3; ++ax) {
            if (idxs[ax] < lo[ax] || idxs[ax] > hi[ax]) {
                if (kinds[ax] == BoundaryKind::Dirichlet) return 0.0;
                int mirrored[3] = {i, j, k};
                mirrored[ax] = idxs[ax] < lo[ax] ? lo[ax] : hi[ax];
                return tempAt(mirrored[0], mirrored[1], mirrored[2]);
            }
        }
        return tmprData[(size_t)(i - 1) + (size_t)nx1 * ((size_t)(j - 1) + (size_t)ny1 * (size_t)(k - 1))];
    };

    int ny2 = ny1 + 1;
    int midplaneY = ny2 / 2;

    double referenceSum = 0.0;
    for (int k = 1; k <= nz1; ++k) {
        double dT = (tempAt(0, midplaneY, k) - tempAt(1, midplaneY, k)) / hpx[0];
        referenceSum += dT * hz12[k - 1];
    }

    double implSum = 0.0;
    for (int k = 1; k <= nz1; ++k)
        implSum += Diagnostics::nusseltMiddleCell(k, midplaneY, tmpr, bc, spacing());

    EXPECT_TRUE(cfdtest::IsClose(referenceSum, implSum)) << "reference=" << referenceSum << " actual=" << implSum;

    double aspectRatio = 1.3;
    double referenceNusseltMiddle = 1.0 / aspectRatio + referenceSum;
    double implNusseltMiddle = Diagnostics::nusseltMiddleFromSum(implSum, aspectRatio);
    EXPECT_TRUE(cfdtest::IsClose(referenceNusseltMiddle, implNusseltMiddle))
        << "reference=" << referenceNusseltMiddle << " actual=" << implNusseltMiddle;
}

TEST_F(DiagnosticsTest, MaxRelativeDifferenceMatchesDist2DFormula) {
    std::vector<double> aData((size_t)nx1 * ny1 * nz1), bData((size_t)nx1 * ny1 * nz1);
    for (auto& v : aData) v = R() * 2.0;
    for (auto& v : bData) v = R() * 0.05;
    FieldView<double> a(aData.data(), nx1, ny1, nz1), b(bData.data(), nx1, ny1, nz1);

    double referenceMax = 0.0;
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k) {
        size_t idx = (size_t)(i - 1) + (size_t)nx1 * ((size_t)(j - 1) + (size_t)ny1 * (size_t)(k - 1));
        double diffVal = std::fabs(aData[idx] - bData[idx]);
        if (std::fabs(bData[idx]) > 0.1) diffVal /= std::fabs(bData[idx]);
        referenceMax = std::max(referenceMax, diffVal);
    }

    double implMax = 0.0;
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        implMax = std::max(implMax, Diagnostics::relativeDifferenceCell(i, j, k, a, b));

    EXPECT_TRUE(cfdtest::IsClose(referenceMax, implMax)) << "reference=" << referenceMax << " actual=" << implMax;
}
