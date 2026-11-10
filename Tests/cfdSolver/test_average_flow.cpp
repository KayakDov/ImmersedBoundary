/**
 * @file test_average_flow.cpp
 * @brief GoogleTest direct verification of AverageFlow::run(): its actual
 * shipped Y-average / stream-function / 2D-diagnostics pipeline, checked
 * against a hand-derived, Fortran-literal reference.
 *
 * This is the first test to actually build and run it. The reference
 * below reuses BoundaryConditions::readVelocity/readTemperature -- the same
 * production boundary resolution AverageFlow.cu itself calls -- so it checks
 * the Y-average/PsiInt/Nusselt_2D/Ekinem_2D arithmetic and indexing, not a
 * second hand-rolled boundary model. The two Tecplot files AverageFlow::run
 * writes are checked with the project's existing comparator
 * (cfd_golden::compareTables), never a new one: the reference is written in
 * the same Tecplot point format and the two files are diffed against each
 * other.
 */
#include <gtest/gtest.h>
#include "setup/AverageFlow.h"
#include "GoldenSupport.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

std::mt19937 rng(20261108);
std::uniform_real_distribution<double> dist(-1.0, 1.0);
double R() { return dist(rng); }

/// @brief A 2D host array indexed like AverageFlow.cu's own private
/// Array2D. Reproduced here rather than reused because that type is an
/// anonymous-namespace implementation detail with no public header.
struct Array2D {
    Array2D(int ni, int nk) : nk_(nk), data_((size_t)(ni + 1) * (size_t)(nk + 1), 0.0) {}
    double& operator()(int i, int k) { return data_[(size_t)i * (size_t)(nk_ + 1) + (size_t)k]; }
    double operator()(int i, int k) const { return data_[(size_t)i * (size_t)(nk_ + 1) + (size_t)k]; }
    int nk_;
    std::vector<double> data_;
};

/// @brief One synthetic scenario's inputs, sized exactly to
/// AverageFlow::run's documented per-array extents.
struct Scenario {
    int nx, ny, nz;
    AxisBoundaries temperatureBc;
    std::vector<double> vx, vz, t;
    std::vector<double> hx12, hy12, hz12, hpx, nodeHalfX, nodeHalfZ;
    double aspectRatio = 0, widthRatio = 0, gravitationalScale = 0;
};

/// @brief Builds random but correctly-sized inputs for one (nx,ny,nz,bc)
/// combination.
Scenario MakeScenario(int nx, int ny, int nz, AxisBoundaries bc) {
    Scenario s{nx, ny, nz, bc, {}, {}, {}, {}, {}, {}, {}, {}, {}};
    const auto fill = [](std::vector<double>& v, size_t n) { v.resize(n); for (auto& x : v) x = R(); };
    const auto fillPositive = [](std::vector<double>& v, size_t n) { v.resize(n); for (auto& x : v) x = 0.1 + std::abs(R()); };
    fill(s.vx, (size_t)nx * (size_t)(ny + 1) * (size_t)(nz + 1));
    fill(s.vz, (size_t)(nx + 1) * (size_t)(ny + 1) * (size_t)nz);
    fill(s.t, (size_t)(nx + 1) * (size_t)(ny + 1) * (size_t)(nz + 1));
    fillPositive(s.hx12, (size_t)nx + 1);
    fillPositive(s.hy12, (size_t)ny + 1);
    fillPositive(s.hz12, (size_t)nz + 1);
    fillPositive(s.hpx, (size_t)nx + 2);
    fill(s.nodeHalfX, (size_t)nx + 3);
    fill(s.nodeHalfZ, (size_t)nz + 3);
    s.aspectRatio = 1.0 + std::abs(R());
    s.widthRatio = 1.0 + std::abs(R());
    s.gravitationalScale = 0.1 + std::abs(R());
    return s;
}

/// @brief The Y-averaged temperature, stream function, and 2D diagnostics,
/// independently computed from AverageFlow.cu's own documented formulas.
struct Reference {
    Array2D tr, psi;
    double nusselt2D, ekinem2D;
};

/// @brief Reference transcription of Average_flow/PsiInt/Nusselt_2D/
/// Ekinem_2D, line-for-line against AverageFlow.cu, using the real
/// BoundaryConditions calls AverageFlow.cu itself uses.
Reference ComputeReference(const Scenario& s) {
    const int nx1 = s.nx + 1, nx2 = s.nx + 2, ny1 = s.ny + 1, nz1 = s.nz + 1, nz2 = s.nz + 2;
    FieldView<double> vxView(const_cast<double*>(s.vx.data()), s.nx, ny1, nz1);
    FieldView<double> vzView(const_cast<double*>(s.vz.data()), nx1, ny1, s.nz);
    FieldView<double> tView(const_cast<double*>(s.t.data()), nx1, ny1, nz1);

    Array2D ux(nx1, nz2), uz(nx2, nz1), tr(nx2, nz2);
    for (int i = 0; i <= nx1; ++i)
        for (int k = 0; k <= nz2; ++k) {
            double sum = 0;
            for (int j = 1; j <= ny1; ++j) sum += BoundaryConditions::readVelocity(vxView, i, j, k) * s.hy12[j - 1];
            ux(i, k) = sum / s.widthRatio;
        }
    for (int i = 0; i <= nx2; ++i)
        for (int k = 0; k <= nz1; ++k) {
            double sum = 0;
            for (int j = 1; j <= ny1; ++j) sum += BoundaryConditions::readVelocity(vzView, i, j, k) * s.hy12[j - 1];
            uz(i, k) = sum / s.widthRatio;
        }
    for (int i = 0; i <= nx2; ++i)
        for (int k = 0; k <= nz2; ++k) {
            double sum = 0;
            for (int j = 1; j <= ny1; ++j)
                sum += BoundaryConditions::readTemperature(tView, i, j, k, s.temperatureBc) * s.hy12[j - 1];
            tr(i, k) = sum / s.widthRatio;
        }

    Array2D psi(nx2, nz2);
    for (int i = 0; i <= nx2; ++i) {
        psi(i, 0) = 0;
        psi(i, nz2) = 0;
    }
    for (int k = 1; k <= nz1; ++k) {
        psi(0, k) = 0;
        psi(nx2, k) = 0;
        for (int i = 1; i <= nx1; ++i) {
            const double vz = (uz(i, k) + uz(i - 1, k) + uz(i, k - 1) + uz(i - 1, k - 1)) / 4.0;
            psi(i, k) = psi(i - 1, k) + vz * s.hpx[i - 1];
        }
    }

    double snu = 0;
    for (int j = 1; j <= nz1; ++j) snu += (tr(0, j) - tr(1, j)) / s.hpx[0] * s.hz12[j - 1];
    const double nusselt2D = 1.0 / s.aspectRatio + snu;

    double ekinem2D = 0;
    for (int i = 1; i <= nx1; ++i)
        for (int k = 1; k <= nz1; ++k) {
            const double uxTerm = ux(i, k) + ux(i - 1, k);
            const double uzTerm = uz(i, k) + uz(i, k - 1);
            ekinem2D += 0.25 * (uxTerm * uxTerm + uzTerm * uzTerm) * s.hx12[i - 1] * s.hz12[k - 1];
        }

    return Reference{tr, psi, nusselt2D, ekinem2D};
}

/// @brief Writes one field as a Tecplot point-format file, in exactly the
/// format cfd_golden::readTable(tecplot=true) expects -- so the existing
/// comparator, not a new one, is what checks it against AverageFlow::run's
/// real output file.
void WriteReferenceTecplot(const std::string& path, const Array2D& field, const std::vector<double>& r,
                            const std::vector<double>& z, int nr, int nz, const char* name) {
    std::FILE* out = std::fopen(path.c_str(), "w");
    ASSERT_NE(out, nullptr);
    std::fprintf(out, " VARIABLES = \"R\",\"Z\",\"%s\"\n", name);
    std::fprintf(out, "ZONE F=POINT, I=%4d, J=%4d\n", nr + 1, nz + 1);
    for (int k = 0; k <= nz; ++k)
        for (int i = 0; i <= nr; ++i)
            std::fprintf(out, "%12.5E %12.5E %12.5E\n", r[i], z[k], field(i, k));
    std::fclose(out);
}

/// @brief Formats a double the same way AverageFlow::run's convOut mirror
/// does (`%.15g`), so compareNumber below compares two printed tokens on
/// equal terms instead of a printed token against a raw double.
std::string Format15g(double value) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.15g", value);
    return buf;
}

/// @brief Reads one `label  = value` line and returns its numeric token
/// verbatim (via cfd_golden::trim, not a new trimmer), so the comparison
/// sees exactly what was printed.
std::string ReadMirroredToken(std::ifstream& in) {
    std::string line;
    EXPECT_TRUE(static_cast<bool>(std::getline(in, line)));
    const auto eq = line.find('=');
    EXPECT_NE(eq, std::string::npos);
    return cfd_golden::trim(line.substr(eq + 1));
}

/// @brief Runs AverageFlow::run on one scenario and checks its two output
/// files, and optionally its Conv.out mirror, against the reference.
void CheckScenario(int nx, int ny, int nz, AxisBoundaries bc, bool mirrorToConvOut) {
    const Scenario s = MakeScenario(nx, ny, nz, bc);
    const Reference ref = ComputeReference(s);
    const auto dir = cfd_golden::uniqueRunDirectory("average_flow");
    const auto psiPath = dir / "Psi_Yaverage.dat", tmprPath = dir / "Tmpr_Yaverage.dat";
    const auto refPsiPath = dir / "Psi_reference.dat", refTmprPath = dir / "Tmpr_reference.dat";
    const auto convPath = dir / "Conv.out";

    std::FILE* convOut = mirrorToConvOut ? std::fopen(convPath.string().c_str(), "w") : nullptr;
    ASSERT_TRUE(!mirrorToConvOut || convOut != nullptr);
    AverageFlow::run<double>(nx, ny, nz, s.vx, s.vz, s.t, s.temperatureBc, s.hx12, s.hy12, s.hz12, s.hpx,
                              s.nodeHalfX, s.nodeHalfZ, s.aspectRatio, s.widthRatio, s.gravitationalScale,
                              psiPath.string(), tmprPath.string(), convOut);
    if (convOut) std::fclose(convOut);

    WriteReferenceTecplot(refPsiPath.string(), ref.psi, s.nodeHalfX, s.nodeHalfZ, nx + 2, nz + 2, "Psi");
    WriteReferenceTecplot(refTmprPath.string(), ref.tr, s.nodeHalfX, s.nodeHalfZ, nx + 2, nz + 2, "Tmpr");

    const auto psiCmp = cfd_golden::compareTables(refPsiPath, psiPath, /*tecplot=*/true);
    EXPECT_EQ(psiCmp.mismatches, 0u) << psiCmp.summary();
    const auto tmprCmp = cfd_golden::compareTables(refTmprPath, tmprPath, /*tecplot=*/true);
    EXPECT_EQ(tmprCmp.mismatches, 0u) << tmprCmp.summary();

    if (!mirrorToConvOut) return;
    std::ifstream in(convPath);
    ASSERT_TRUE(static_cast<bool>(in));
    const std::string nusseltToken = ReadMirroredToken(in);
    const std::string ekinemToken = ReadMirroredToken(in);
    cfd_golden::Comparison scalarCmp;
    cfd_golden::compareNumber(scalarCmp, "2D Nusselt number", nusseltToken, Format15g(ref.nusselt2D), true);
    cfd_golden::compareNumber(scalarCmp, "2D kinetic energy", ekinemToken, Format15g(ref.ekinem2D), true);
    EXPECT_EQ(scalarCmp.mismatches, 0u) << scalarCmp.summary();
}

} // namespace

TEST(AverageFlowTest, DirichletTemperatureWithConvOutMirror) {
    CheckScenario(4, 3, 5, AxisBoundaries::fill(BoundaryKind::Dirichlet), /*mirrorToConvOut=*/true);
}

TEST(AverageFlowTest, MixedNeumannDirichletTemperatureWithoutConvOutMirror) {
    CheckScenario(6, 2, 3, AxisBoundaries(BoundaryKind::Neumann, BoundaryKind::Dirichlet, BoundaryKind::Neumann),
                  /*mirrorToConvOut=*/false);
}

TEST(AverageFlowTest, ThrowsWhenOutputPathCannotBeOpened) {
    const Scenario s = MakeScenario(3, 2, 3, AxisBoundaries::fill(BoundaryKind::Dirichlet));
    EXPECT_THROW(
        AverageFlow::run<double>(3, 2, 3, s.vx, s.vz, s.t, s.temperatureBc, s.hx12, s.hy12, s.hz12, s.hpx,
                                  s.nodeHalfX, s.nodeHalfZ, s.aspectRatio, s.widthRatio, s.gravitationalScale,
                                  "/nonexistent_directory_for_average_flow_test/Psi.dat", "/tmp/unused_tmpr_av.dat",
                                  nullptr),
        std::runtime_error);
}
