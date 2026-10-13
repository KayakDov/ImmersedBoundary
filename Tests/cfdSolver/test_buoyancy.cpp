/**
 * @file test_buoyancy.cpp
 * @brief GoogleTest verification of FiniteDifferenceOperators::buoyancyForceZ
 * against the unsimplified Fortran formula (time_step_Q2D.f90's buoyancy
 * term), confirming both the formula itself and the 2*buoyancyByX[i]
 * simplification used in the real implementation.
 */

#include <gtest/gtest.h>
#include "physics/Operators.cuh"
#include "CfdTestHelpers.h"
#include <random>
#include <vector>

namespace {
std::mt19937 rng(123);
std::uniform_real_distribution<double> dist(-1.0, 1.0);
double R() { return dist(rng); }
} // namespace

TEST(BuoyancyForceZTest, MatchesUnsimplifiedFortranFormula) {
    // Nx1=7, Ny1=8, Nz=6 (so Nz1=7) -- RHSz's own real range is (Nx1,Ny1,Nz).
    int nx1 = 7, ny1 = 8, nz = 6, nz1 = nz + 1;
    std::vector<double> rhsZRefData((size_t)nx1 * ny1 * nz), rhsZImplData((size_t)nx1 * ny1 * nz);
    std::vector<double> tmprNewData((size_t)nx1 * ny1 * nz1);
    std::vector<double> buoyancyByX(nx1 + 2);

    for (auto& v : rhsZRefData) v = R();
    rhsZImplData = rhsZRefData; // same starting RHSz content for both
    for (auto& v : tmprNewData) v = R();
    for (auto& v : buoyancyByX) v = R();

    double buGr = 1.7;

    auto idx = [&](int i, int j, int k) {
        return (size_t)(i - 1) + (size_t)nx1 * ((size_t)(j - 1) + (size_t)ny1 * (size_t)(k - 1));
    };

    // Reference: direct Fortran-literal formula, Teta read via its own two
    // terms (NOT pre-simplified), to independently confirm the
    // 2*buoyancyByX[i] simplification used in the real implementation.
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k) {
        double tetaK = buoyancyByX[i];  // Teta(i,j,k)   -- constant in j,k
        double tetaK1 = buoyancyByX[i]; // Teta(i,j,k+1) -- same, constant in j,k
        double tmprK = tmprNewData[idx(i, j, k)];
        double tmprK1 = tmprNewData[idx(i, j, k + 1)];
        rhsZRefData[idx(i, j, k)] -= 0.5 * buGr * (tmprK + tmprK1) + 0.5 * buGr * (tetaK + tetaK1);
    }

    FieldView<double> rhsZ(rhsZImplData.data(), nx1, ny1, nz);
    FieldView<double> tmprNew(tmprNewData.data(), nx1, ny1, nz1);
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k)
        FiniteDifferenceOperators::buoyancyForceZ(i, j, k, rhsZ, tmprNew, buoyancyByX.data(), buGr);

    cfdtest::MismatchCollector mismatches("buoyancyForceZ");
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k)
        mismatches.Check(rhsZRefData[idx(i, j, k)], rhsZImplData[idx(i, j, k)]);
    mismatches.Finish();
}
