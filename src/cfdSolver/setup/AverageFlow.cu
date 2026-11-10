/**
 * @file AverageFlow.cu
 * @ingroup cfd_solver
 * @brief Implements AverageFlow::run().
 *
 * Pure host arithmetic and file I/O -- no device code, no kernel
 * launches. Compile-verified, linked, and run directly by
 * test_average_flow.cpp (3/3 passing) and as part of the combined
 * host-only suite; also checked by hand against modfv_3D_Q2D.f90's
 * Average_flow/PsiInt/Nusselt_2D/Ekinem_2D and Point_Write_2D, line by
 * line, including every array's exact index range.
 */

#include "AverageFlow.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {

/**
 * @brief A 2D host array indexed exactly like Fortran's own
 * `Real(kind=8) :: Arr(0:ni,0:nk)` declarations -- constructed with the
 * two *maximum* indices, not sizes, so every loop bound below is copied
 * from the original source with no translation.
 */
template<typename Real>
class Array2D {
public:
    Array2D(int ni, int nk) : nk_(nk), data_((size_t)(ni + 1) * (size_t)(nk + 1), Real(0)) {}

    Real& operator()(int i, int k) { return data_[(size_t)i * (size_t)(nk_ + 1) + (size_t)k]; }
    Real operator()(int i, int k) const { return data_[(size_t)i * (size_t)(nk_ + 1) + (size_t)k]; }

private:
    int nk_;
    std::vector<Real> data_;
};

/**
 * @brief Direct port of Point_Write_2D: writes one 2D field as a TECPLOT
 * point-format file.
 *
 * Fortran's exact E12.5 mantissa convention (leading zero before the
 * decimal point: 0.d1d2d3d4d5E+-ee) isn't reproduced -- this file is read
 * by TECPLOT for visualization, never re-read by this program or
 * cross-checked byte-for-byte against Fortran's own output, so the
 * ordinary C scientific-notation convention (d1.d2d3d4d5E+-ee, equally
 * valid TECPLOT input) is used instead. Same reasoning as this project's
 * Conv.out header, which is a parameter echo rather than a byte-replica of
 * a header whose exact source wasn't available.
 */
template<typename Real>
void writeTecplot2D(const std::string& path, const Array2D<Real>& field,
                     const std::vector<Real>& r, const std::vector<Real>& z,
                     int nr, int nz, const char* name) {
    std::FILE* out = std::fopen(path.c_str(), "w");
    if (!out)
        throw std::runtime_error("AverageFlow: could not open " + path + " for writing");

    std::fprintf(out, " VARIABLES = \"R\",\"Z\",\"%s\"\n", name);
    std::fprintf(out, "ZONE F=POINT, I=%4d, J=%4d\n", nr + 1, nz + 1);

    for (int k = 0; k <= nz; ++k) {
        for (int i = 0; i <= nr; ++i) {
            std::fprintf(out, "%12.5E %12.5E %12.5E\n", (double)r[i], (double)z[k], (double)field(i, k));
        }
    }

    std::fclose(out);
}

} // namespace

template<typename Real>
void AverageFlow::run(int nx, int ny, int nz,
                       const std::vector<Real>& vxCurrent, const std::vector<Real>& vzCurrent,
                       const std::vector<Real>& temperatureCurrent, AxisBoundaries temperatureBc,
                       const std::vector<Real>& hx12, const std::vector<Real>& hy12,
                       const std::vector<Real>& hz12, const std::vector<Real>& hpx,
                       const std::vector<Real>& nodeHalfX, const std::vector<Real>& nodeHalfZ,
                       Real aspectRatio, Real widthRatio, Real gravitationalScale,
                       const std::string& psiPath, const std::string& tmprPath, std::FILE* convOut) {
    int nx1 = nx + 1, nx2 = nx + 2;
    int ny1 = ny + 1;
    int nz1 = nz + 1, nz2 = nz + 2;

    /* Host views over already-downloaded data -- FieldView is just
     * pointer arithmetic, so it works identically whether the pointer it
     * wraps is device or host memory; here it's host. */
    FieldView<Real> vxView(const_cast<Real*>(vxCurrent.data()), nx, ny1, nz1);
    FieldView<Real> vzView(const_cast<Real*>(vzCurrent.data()), nx1, ny1, nz);
    FieldView<Real> tView(const_cast<Real*>(temperatureCurrent.data()), nx1, ny1, nz1);

    /* Ux/Uz/Tr are each a Hy12-weighted sum over the source field's full
     * y-range (j=1..Ny1 -- no boundary resolution needed on y),
     * normalized by widthRatio. The i/k ranges below are each output
     * array's own (wider) extent, so most i/k combinations fall outside
     * the source field's own real range and resolve through the same
     * BoundaryConditions functions the solve itself uses to read a ghost
     * cell -- velocity unconditionally 0 (no-slip), temperature 0
     * (Dirichlet) or mirrored (Neumann) per temperatureBc. */
    Array2D<Real> ux(nx1, nz2), uz(nx2, nz1), tr(nx2, nz2);

    for (int i = 0; i <= nx1; ++i) {
        for (int k = 0; k <= nz2; ++k) {
            Real sum = Real(0);
            for (int j = 1; j <= ny1; ++j)
                sum += BoundaryConditions::readVelocity(vxView, i, j, k) * hy12[j - 1];
            ux(i, k) = sum / widthRatio;
        }
    }

    for (int i = 0; i <= nx2; ++i) {
        for (int k = 0; k <= nz1; ++k) {
            Real sum = Real(0);
            for (int j = 1; j <= ny1; ++j)
                sum += BoundaryConditions::readVelocity(vzView, i, j, k) * hy12[j - 1];
            uz(i, k) = sum / widthRatio;
        }
    }

    for (int i = 0; i <= nx2; ++i) {
        for (int k = 0; k <= nz2; ++k) {
            Real sum = Real(0);
            for (int j = 1; j <= ny1; ++j)
                sum += BoundaryConditions::readTemperature(tView, i, j, k, temperatureBc) * hy12[j - 1];
            tr(i, k) = sum / widthRatio;
        }
    }

    /* Zero on all four domain edges by construction. Every Uz(...) read
     * below is within Uz's own just-computed (0:Nx2,0:Nz1) extent -- no
     * further boundary resolution needed. */
    Array2D<Real> psi(nx2, nz2);

    for (int i = 0; i <= nx2; ++i) {
        psi(i, 0) = Real(0);
        psi(i, nz2) = Real(0);
    }
    for (int k = 1; k <= nz1; ++k) {
        psi(0, k) = Real(0);
        psi(nx2, k) = Real(0);
        for (int i = 1; i <= nx1; ++i) {
            Real vz = (uz(i, k) + uz(i - 1, k) + uz(i, k - 1) + uz(i - 1, k - 1)) / Real(4);
            psi(i, k) = psi(i - 1, k) + vz * hpx[i - 1];
        }
    }

    /* A one-sided finite difference at x=0, using Tr's own
     * Dirichlet-resolved wall row, integrated over z. */
    Real snu = Real(0);
    for (int j = 1; j <= nz1; ++j) {
        Real dt = (tr(0, j) - tr(1, j)) / hpx[0];
        snu += dt * hz12[j - 1];
    }
    Real nusselt2D = Real(1) / aspectRatio + snu;

    Real ekinem2D = Real(0);
    for (int i = 1; i <= nx1; ++i) {
        for (int k = 1; k <= nz1; ++k) {
            Real uxTerm = ux(i, k) + ux(i - 1, k);
            Real uzTerm = uz(i, k) + uz(i, k - 1);
            ekinem2D += Real(0.25) * (uxTerm * uxTerm + uzTerm * uzTerm) * hx12[i - 1] * hz12[k - 1];
        }
    }

    std::printf("      2D Nusselt number  = %.15g\n", (double)nusselt2D);
    std::printf("      2D kinetic energy  = %.15g\n", (double)ekinem2D);
    if (convOut) {
        std::fprintf(convOut, "      2D Nusselt number  = %.15g\n", (double)nusselt2D);
        std::fprintf(convOut, "      2D kinetic energy  = %.15g\n", (double)ekinem2D);
        std::fflush(convOut);
    }

    Real maxAbsPsi = Real(0);
    for (int i = 0; i <= nx2; ++i)
        for (int k = 0; k <= nz2; ++k)
            maxAbsPsi = std::max(maxAbsPsi, std::abs(psi(i, k)));
    std::printf(" MaxPsi_average= %.15g\n", (double)(maxAbsPsi / gravitationalScale));

    writeTecplot2D(psiPath, psi, nodeHalfX, nodeHalfZ, nx2, nz2, "Psi");
    writeTecplot2D(tmprPath, tr, nodeHalfX, nodeHalfZ, nx2, nz2, "Tmpr");
}

template void AverageFlow::run<double>(int, int, int, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, AxisBoundaries,
    const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    double, double, double, const std::string&, const std::string&, std::FILE*);

template void AverageFlow::run<float>(int, int, int, const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, AxisBoundaries,
    const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
    float, float, float, const std::string&, const std::string&, std::FILE*);
