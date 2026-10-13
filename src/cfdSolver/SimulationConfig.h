/**
 * @file SimulationConfig.h
 * @brief The master configuration -- replaces Conv.dat plus every
 * previously-hardcoded Fortran constant that controlled program behavior.
 *
 * @details
 * Every field here corresponds to something the original Fortran code
 * either read from Conv.dat or had wired into the source directly (and
 * could therefore only be changed by editing and recompiling a .f90 file
 * -- something Alex can no longer do). Each field names, in its comment,
 * exactly which Fortran file/variable it replaces, so the mapping back to
 * the original code is never ambiguous.
 *
 * Three deliberate clarity improvements over Conv.dat's own conventions,
 * called out explicitly since they change what a given value *means*
 * relative to the original file:
 *
 *   1. Conv.dat's EVD_BCx/y/z and EVD_Pot_X/Y/Z were bare 0/1 integers
 *      (0 - Dirichlet, 1 - Neumann, per Conv.dat's own inline comment).
 *      Here they're BoundaryKind::Dirichlet/Neumann -- same meaning, but a
 *      typo in the JSON ("neuman") is a parse error instead of a silently
 *      wrong simulation.
 *
 *   2. Icheck and I_Fourier both used inverted, error-code-style Fortran
 *      conventions (0 means the feature is ON). Here they're
 *      runConservationChecks / timeAveraging.enabled -- true literally
 *      means "on".
 *
 *   3. Iprint did double duty in Fortran: its magnitude set the print
 *      cadence, and its *sign* separately signaled "read initial values
 *      from Conv.ddd instead of starting fresh" (init_3D.f90's
 *      `If (Iprint < 0)` branch). That overload is split into two
 *      independent fields here: printEvery (always positive) and
 *      restartFile (null for a fresh start, a path to resume from).
 *
 * Deliberately NOT exposed: Ckor (fixed at 1.5 in init_3D.f90). This is a
 * numerical-method constant from the projection scheme's own derivation,
 * not a physical or tunable parameter -- changing it would silently break
 * the method's correctness rather than let Alex explore something
 * meaningful, so it stays an implementation detail. Likewise Conv.dat's
 * "Number of CPUs" field is not carried forward: it was read by
 * ConvMain_3D_Q2D.f90 but never actually used anywhere (thread count came
 * from omp_get_num_procs(), not from that field) -- a dead parameter in
 * the original code, not a real setting.
 */

#ifndef CFDSOLVER_SIMULATIONCONFIG_H
#define CFDSOLVER_SIMULATIONCONFIG_H

#include <string>
#include <vector>
#include <optional>
#include <cstdint>

// BoundaryKind (Dirichlet/Neumann) is defined once, in fields/BoundaryConditions.cuh
// -- not re-declared here. BoundaryConditions.cuh is host-compilable (it stubs out
// __host__/__device__/__forceinline__ when not built under nvcc), so pulling it into
// this otherwise CUDA-independent header costs nothing and keeps the enum from being
// defined twice in any translation unit that includes both files, which every one of
// Mesh.cu/CfdSimulation.cu/GpuSolverSetup.cu does.
#include "fields/BoundaryConditions.cuh"

/// Which subset of the physics to advance. Matches Conv.dat's Iexcl:
/// Iexcl < 0 -> TemperatureOnly, Iexcl > 0 -> FlowOnly, Iexcl == 0 -> Full.
enum class SolveMode { Full, TemperatureOnly, FlowOnly };

/// Per-field GPU placement and solver variant -- was hardcoded per-call in
/// AlexCudaCompatibility.f90's Initialize_GPU_Solvers (gpuInd=0/1 literals,
/// thomas=.true./.false. literals). Now one of these per field, all in the
/// config file instead of in source that would need recompiling.
struct FieldGpuConfig {
    int device = 0;              ///< Which physical GPU (0-based, matches nvidia-smi -L / CUDA_VISIBLE_DEVICES numbering)
    bool useThomasAlgorithm = true; ///< true: Thomas-algorithm-optimized 3D solve (faster, slightly less accurate). false: full 3D eigendecomposition.
};

struct SimulationConfig {
    // ===================================================================
    // grid -- Conv.dat's Nx, Ny, Nz
    // ===================================================================
    int Nx = 0, Ny = 0, Nz = 0;

    // ===================================================================
    // physics -- Conv.dat's AspRa, WidRa, Gr, Bi, Prandtl, Hartmann
    // (read via ConvMain_3D_Q2D.f90's first Read(1,101,...) call)
    // ===================================================================
    double aspectRatio = 1.0;   ///< AspRa
    double widthRatio = 1.0;    ///< WidRa
    double grashof = 0.0;       ///< Gr
    double biot = 0.0;          ///< Bi
    double prandtl = 0.0;       ///< Prandtl
    double hartmann = 0.0;      ///< Hartmann

    // ===================================================================
    // numerics -- Conv.dat's Eps, EpsCnv, ItMax, Htime, Time, Iprint
    // ===================================================================
    double eps = 1e-7;               ///< Eps (unused by the live solve path in the original -- see ConvMain_3D_Q2D.f90; kept for parity, not currently read by anything)
    double epsConvergence = 1e-6;    ///< EpsCnv -- the max(RNSx,RNSy,RNSz,RTmpr) < EpsCnv early-exit test in Solution_time_Q2D.f90
    int maxIterations = 9000;        ///< ItMax
    double timeStep = 1e-4;          ///< Htime
    double timeInterval = 0.1;       ///< Time (Conv.dat's field is reused as both input and, internally, as Time+Tstart -- here it's just the requested interval)
    int printEvery = 1000;           ///< Iprint's magnitude only -- see the restart/checkpoint split above

    // ===================================================================
    // diagnostics -- Conv.dat's Icheck, Iexcl
    // ===================================================================
    bool runConservationChecks = false; ///< Icheck==0 in Fortran (inverted here -- see class comment). Expensive: re-runs FdDiv/GradPx/y/z/VgrTmp/VgrdVx/y/z an extra time per printEvery step purely to print diagnostics (check_3D.f90) -- off by default, matching every run this project has actually used.
    SolveMode solveMode = SolveMode::Full; ///< Iexcl

    // ===================================================================
    // temperature -- Conv.dat's Istat, EVD_BCx, EVD_BCy, EVD_BCz
    // ===================================================================
    bool temperatureStationary = false; ///< Istat==0 in Fortran (inverted -- true here means stationary, matching Istat==0's actual effect: dt_temp=0 in time_step_Q2D.f90, dropping the time-derivative term)
    BoundaryKind temperatureBoundaryX = BoundaryKind::Dirichlet; ///< EVD_BCx
    BoundaryKind temperatureBoundaryY = BoundaryKind::Dirichlet; ///< EVD_BCy
    BoundaryKind temperatureBoundaryZ = BoundaryKind::Dirichlet; ///< EVD_BCz

    // ===================================================================
    // potential -- Conv.dat's EVD_Pot_X, EVD_Pot_Y, EVD_Pot_Z
    // ===================================================================
    BoundaryKind potentialBoundaryX = BoundaryKind::Neumann; ///< EVD_Pot_X
    BoundaryKind potentialBoundaryY = BoundaryKind::Neumann; ///< EVD_Pot_Y
    BoundaryKind potentialBoundaryZ = BoundaryKind::Neumann; ///< EVD_Pot_Z

    // ===================================================================
    // time_averaging -- Conv.dat's I_Fourier, N_Fourier, Omega1/Omega2
    // (Solution_time_Q2D.f90's Fourier_sum/Plot_Amplitude feature)
    // ===================================================================
    bool timeAveragingEnabled = false; ///< I_Fourier==0 in Fortran (inverted -- see class comment)
    std::vector<double> fourierFrequencies; ///< was fixed at exactly 2 values (Omega1, Omega2) by Conv.dat's layout; any length now. Empty is valid (plain time-averaging with no Fourier decomposition).

    // ===================================================================
    // gpu -- was hardcoded per-call in AlexCudaCompatibility.f90
    // ===================================================================
    FieldGpuConfig gpuTemperature{0, true};
    FieldGpuConfig gpuVx{0, true};
    FieldGpuConfig gpuVy{1, true};
    FieldGpuConfig gpuVz{1, true};
    FieldGpuConfig gpuPressure{0, false};  ///< Pressure is a pure Poisson solve (helmholtzShift=0); Thomas doesn't apply the same way -- see EigenDecompForFortran/AlexCudaCompatibility.f90's original PressureHandle construction, which always passed thomas=.false.
    FieldGpuConfig gpuPotential{1, false}; ///< Same reasoning as Pressure -- pure Poisson.

    // ===================================================================
    // mesh -- MeshStretch.f90's hardcoded `s = 10.d0` tanh-stretching factor
    // ===================================================================
    double meshStretchingFactor = 10.0;

    // ===================================================================
    // restart -- replaces Iprint<0's dual-purpose sign trick (see class comment)
    // ===================================================================
    std::optional<std::string> restartFile; ///< null/absent: fresh start (init_3D.f90's "crazily estimated" 1e-6 branch). A path: resume from that checkpoint (init_3D.f90's Read(3,...) branch).
    int checkpointEvery = 1000; ///< How often to write a resumable checkpoint (Outp, outp_3D.f90 -- was tied to Iprint's magnitude in the original; independent here).

    // ===================================================================
    // output -- was hardcoded relative filenames (Tmpr.dat, Pres.dat, ...)
    // scattered through ConvMain_3D_Q2D.f90's Open statements
    // ===================================================================
    std::string outputDirectory = ".";

    /**
     * @brief Loads and validates a SimulationConfig from a JSON file.
     * @param path Path to the master config file.
     * @throws std::runtime_error with a specific, human-readable message
     *         naming the missing/malformed field -- never silently
     *         defaults a required physical parameter.
     */
    static SimulationConfig loadFromFile(const std::string& path);
};

#endif //CFDSOLVER_SIMULATIONCONFIG_H
