/**
 * @file AverageFlow.h
 * @ingroup cfd_solver
 * @brief Declares AverageFlow, the port of modfv_3D_Q2D.f90's
 * Average_flow.
 */

#ifndef CFDSOLVER_AVERAGEFLOW_H
#define CFDSOLVER_AVERAGEFLOW_H

#include <cstdio>
#include <string>
#include <vector>
#include "../fields/BoundaryConditions.cuh"

/**
 * @ingroup cfd_solver
 * @brief Port of modfv_3D_Q2D.f90's Average_flow, including its three
 * internal procedures (PsiInt, Nusselt_2D, Ekinem_2D).
 *
 * Y-averages VMx/VMz/Tmpr into 2D (x,z) fields (Ux/Uz/Tr), reconstructs a
 * stream function from the Y-averaged VMz by trapezoidal integration in x
 * (PsiInt), reports the resulting 2D Nusselt number and kinetic energy
 * (Nusselt_2D/Ekinem_2D), and writes the Y-averaged stream function and
 * temperature as TECPLOT point-format files (Point_Write_2D).
 *
 * Unconditional in the original: called once, at the very end of
 * ConvMain_3D_Q2D's run, right after the 3D Nusselt/kinetic-energy report
 * -- not gated by any config flag here either. The original reads
 * whatever VMx/VMz/Tmpr currently hold at that point, which is the
 * time-average when time-averaging is on (CfdSimulation's
 * timeAveragingEnabled: see CfdSimulation::finalizeTimeAverage(), which
 * performs that substitution in place before this function's caller in
 * run() downloads vx_/vz_/temperature_) and the plain instantaneous field
 * otherwise. This function itself has no opinion on which: it is given
 * whatever its caller downloaded, same as the original reads whatever
 * VMx/VMz/Tmpr happen to hold.
 *
 * Pure host-side post-processing: takes already-downloaded field data, not
 * device views. None of this needs GPU acceleration -- Fortran's own
 * Average_flow is plain CPU code, run once per simulation after the
 * GPU-accelerated time-stepping has finished.
 */
class AverageFlow {
public:
    /**
     * @brief Runs the Y-average / stream-function / 2D-diagnostics
     * pipeline and writes its two TECPLOT output files.
     *
     * @tparam Real Floating-point type.
     * @param nx Grid's Nx (the solved-field interior count, not Nx+1).
     * @param ny Grid's Ny.
     * @param nz Grid's Nz.
     * @param vxCurrent VMx's current-timestep data, downloaded from the
     * device. Real extent (1:Nx,1:Ny+1,1:Nz+1), i.e. nx*(ny+1)*(nz+1)
     * elements.
     * @param vzCurrent VMz's current-timestep data. Real extent
     * (1:Nx+1,1:Ny+1,1:Nz), i.e. (nx+1)*(ny+1)*nz elements.
     * @param temperatureCurrent Tmpr's current-timestep data. Real extent
     * (1:Nx+1,1:Ny+1,1:Nz+1), i.e. (nx+1)*(ny+1)*(nz+1) elements.
     * @param temperatureBc Temperature's configured per-axis boundary
     * conditions -- needed to resolve Tr's x/z boundary rows the same way
     * BoundaryConditions::readTemperature resolves them during the solve.
     * Never Teta-inclusive -- see CfdSimulation.cu's Average_flow call site
     * for why Tmpr/temperature_ is always Teta-free by the time anything
     * reads it, time-averaging on or off.
     * @param hx12 Mesh::hostHx12(): cell spacing on x, index 0..Nx.
     * @param hy12 Mesh::hostHy12(): cell spacing on y, index 0..Ny.
     * @param hz12 Mesh::hostHz12(): cell spacing on z, index 0..Nz.
     * @param hpx Mesh::hostHpx(): control-volume spacing on x, index
     * 0..Nx+1.
     * @param nodeHalfX Mesh::hostNodeHalfX(): x half-node positions, index
     * 0..Nx+2 -- written out as both output files' "R" column.
     * @param nodeHalfZ Mesh::hostNodeHalfZ(): z half-node positions, index
     * 0..Nz+2 -- written out as both output files' "Z" column.
     * @param aspectRatio Config's physics.aspect_ratio (AspRa).
     * @param widthRatio Config's physics.width_ratio (WidRa).
     * @param gravitationalScale 1/sqrt(Grashof) (DGr) -- only used to scale
     * the console-only MaxPsi_average report.
     * @param psiPath Where to write the Y-averaged stream function
     * (Psi_Yaverage.dat). Truncated, not appended -- matches the original,
     * which opens this file with no position='append' (unlike Serie.dat/
     * History.dat/Conv.out), consistent with it being a final-state
     * snapshot rather than an accumulating log.
     * @param tmprPath Where to write the Y-averaged temperature
     * (Tmpr_Yaverage.dat). Same truncate semantics as psiPath.
     * @param convOut Optional: also mirror the 2D Nusselt/kinetic-energy
     * report into this already-open file (Conv.out), the same way
     * CfdSimulation::printFinalReport() mirrors the 3D report. Pass
     * nullptr to skip.
     * @throws std::runtime_error if either output file can't be opened for
     * writing.
     */
    template<typename Real>
    static void run(int nx, int ny, int nz,
                     const std::vector<Real>& vxCurrent, const std::vector<Real>& vzCurrent,
                     const std::vector<Real>& temperatureCurrent, AxisBoundaries temperatureBc,
                     const std::vector<Real>& hx12, const std::vector<Real>& hy12,
                     const std::vector<Real>& hz12, const std::vector<Real>& hpx,
                     const std::vector<Real>& nodeHalfX, const std::vector<Real>& nodeHalfZ,
                     Real aspectRatio, Real widthRatio, Real gravitationalScale,
                     const std::string& psiPath, const std::string& tmprPath, std::FILE* convOut);
};

#endif //CFDSOLVER_AVERAGEFLOW_H
