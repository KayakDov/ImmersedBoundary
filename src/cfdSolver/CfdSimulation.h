/**
 * @file CfdSimulation.h
 * @ingroup cfd_solver
 * @brief CfdSimulation: the main orchestrator, replacing ConvMain_3D_Q2D.f90,
 * Solution_time_Q2D.f90, and time_step_Q2D.f90.
 */

#ifndef CFDSOLVER_CFDSIMULATION_H
#define CFDSOLVER_CFDSIMULATION_H

#include <cstdio>
#include <string>
#include <memory>
#include "setup/GpuSolverSetup.h"
#include "physics/ElectromagneticForce.cuh"
#include "physics/Diagnostics.cuh"
#include "setup/CheckpointIO.h"

/**
 * @ingroup cfd_solver
 * @brief One cavity-convection simulation: owns its mesh, its six solved
 * fields, and the scratch buffers its timestep needs; exposes run() as the
 * only entry point a caller needs.
 *
 * Two features present in the original Fortran are deliberately not
 * implemented, and are refused with a clear error at construction rather
 * than silently ignored:
 *
 *   - time_averaging.enabled (I_Fourier==0 in the original): a separate
 *     running time-average and Fourier-mode decomposition, confirmed
 *     disabled in every Conv.dat this project has ever used. Its
 *     accumulation math is straightforward, but writing it with the same
 *     rigor as everything else here -- new state for six fields, new
 *     output formats for Amplitudes.dat and Average_velocity.ddd -- is a
 *     genuinely separate, bounded piece of work deserving its own pass.
 *
 *   - diagnostics.run_conservation_checks (Icheck==0 in the original): a
 *     pure side-channel diagnostic (check_3D.f90) that recomputes FdDiv/
 *     GradPx/y/z/VgrTmp/VgrdVx/y/z a second time purely to print
 *     divergence and conservation numbers -- never affects the simulation
 *     itself, and likewise confirmed disabled everywhere this project has
 *     run.
 *
 * run() also writes three of the original's own always-on output files,
 * under cfg_.outputDirectory, alongside the existing checkpoint:
 * Serie.dat and History.dat (every iteration, unconditionally -- port of
 * Solution_time_Q2D.f90's Ek/Anu and point-probe Write(25,*)/Write(26,*))
 * and Conv.out (mirrors printIterationReport()'s and printFinalReport()'s
 * stdout text; its one-line header is a parameter echo, not a replica of
 * the original's own header block, whose exact source wasn't available
 * when this was written).
 *
 * SolveMode (Iexcl in the original) is parsed and stored, matching what
 * the original config accepted, but -- confirmed directly from time_step_
 * Q2D.f90 -- Iexcl is read and reported in the original source but never
 * actually referenced by the timestep computation itself, in Alex's own
 * code, not just this port. It is not given a behavior here that the
 * original did not actually have either.
 *
 * Potential has no Old/Current/New time-level triad (it is fully
 * recomputed from the current velocity field every step, never an
 * independent degree of freedom), so it is held as a plain SolvedField,
 * not wrapped in TimeDependentField. Pressure's accumulated value (Prs)
 * is likewise a plain GpuField, separate from the pressure-*correction*
 * solve (Dprs): `Prs = Prs + Dprs` each step, confirmed directly from
 * time_step_Q2D.f90 -- they were never the same buffer.
 *
 * @tparam Real Floating-point type.
 */
template<typename Real>
class CfdSimulation {
public:
    /**
     * @brief Constructs a simulation: generates the mesh, builds the six
     * solvers, and allocates every scratch buffer.
     * @param cfg The simulation configuration.
     * @throws std::runtime_error if cfg requests an unimplemented feature
     * (see class comment).
     */
    explicit CfdSimulation(SimulationConfig cfg);

    CfdSimulation(const CfdSimulation&) = delete;             ///< Not copyable: owns unique device allocations.
    CfdSimulation& operator=(const CfdSimulation&) = delete;   ///< Not copyable: owns unique device allocations.
    CfdSimulation(CfdSimulation&&) = default;                  ///< Movable.
    CfdSimulation& operator=(CfdSimulation&&) = default;       ///< Movable.

    /**
     * @brief Sets every field to its starting state: either the small
     * constants init_3D.f90 used for a fresh start, or a checkpoint's
     * contents if cfg.restartFile is set. Must be called once, before
     * run().
     */
    void initialize();

    /**
     * @brief Runs the simulation to completion: advances timesteps until
     * cfg.timeInterval or cfg.maxIterations is reached, or the
     * convergence criterion is met, printing and checkpointing at
     * cfg.printEvery, then prints the final diagnostic report.
     */
    void run();

private:
    /**
     * @brief Everything the constructor needs to initialize every member
     * with its real, final value, built by prepare() before any member
     * exists. Exists so the constructor can do multi-step setup (config
     * validation, mesh generation, peer-access enablement, solver
     * construction) without resorting to placeholder member values that
     * get immediately discarded and replaced -- every CfdSimulation member
     * is constructed exactly once, with its real value, via the
     * delegating constructor below.
     */
    struct Ingredients {
        SimulationConfig cfg;
        // Owned via unique_ptr, not by value: Handle is deliberately
        // neither copyable nor movable (see handle.h) -- every caller in
        // the codebase takes a Handle&/Handle*. prepare() heap-allocates
        // this once with std::make_unique<Handle>(...) and only the
        // pointer travels from here into CfdSimulation's hand_ member.
        std::unique_ptr<Handle> hand;
        Mesh<Real> mesh;
        GpuField<Real> buoyancyByX;
        GpuField<Real> temperatureOld, temperatureCurrent;
        GpuField<Real> vxOld, vxCurrent;
        GpuField<Real> vyOld, vyCurrent;
        GpuField<Real> vzOld, vzCurrent;
        BuiltSolvers<Real> solvers;
        GpuField<Real> pressure;
        GpuField<Real> rhsXScratch, rhsYScratch, rhsZScratch;
        GpuField<Real> reductionScratch;
        GpuField<unsigned long long> maxReductionScratch;
        GpuField<Real> referenceValueScratch;
    };

    static Ingredients prepare(SimulationConfig cfg);
    explicit CfdSimulation(Ingredients ingredients);

    void buildAndLaunchTemperature();
    void buildAndLaunchVy();
    void launchPotential();
    void buildVxVzStencilTerms();
    Real synchTemperatureAndComputeResidual();
    void applyBuoyancyToRhsZ();
    void finishPotentialAndApplyForce();
    void launchVxAndVz();
    void synchVelocities();
    void solvePressureCorrection();
    void correctVelocitiesAndPressure();
    void residuals(Real& rnsX, Real& rnsY, Real& rnsZ, Real& rdp);
    void shiftAllTimeLevels();
    void runOneTimeStep(Real& rnsX, Real& rnsY, Real& rnsZ, Real& rTmpr, Real& rdp);

    void initializeFreshStart();
    void printIterationReport(Real timeNow, Real rnsX, Real rnsY, Real rnsZ, Real rTmpr, Real rdp) const;
    void printFinalReport() const;

    /**
     * @brief Opens Serie.dat, History.dat, and Conv.out under
     * cfg_.outputDirectory (append mode, matching the original's
     * position='append' -- confirmed from the live project's own
     * multi-run Serie.dat/History.dat, which hold several runs' data
     * concatenated). Writes Conv.out's one-line parameter-echo header.
     * Called once, at the top of run().
     * @throws std::runtime_error if any of the three files can't be opened.
     */
    void openOutputFiles();

    /**
     * @brief Joins cfg_.outputDirectory and filename into one path,
     * stripping outputDirectory's trailing slash (if any) so the result
     * never has a doubled separator.
     */
    std::string outputPath(const char* filename) const;

    /**
     * @brief Computes the same global kinetic energy and Nusselt number
     * printFinalReport() computes, plus History.dat's point probe (VMz
     * and Tmpr at probeX_/probeY_/probeZ_) -- port of
     * Solution_time_Q2D.f90's unconditional, every-iteration
     * `Ek = Ekinem(); Anu = Nusselt()` / `VMz(i_x_write,...)` /
     * `Tmpr(i_x_write,...)` calls that feed Serie.dat and History.dat.
     *
     * Explicitly synchs vx_/vy_/vz_/temperature_'s own solve handles
     * before reading their "current" views from *hand_ -- the same
     * synch-before-cross-handle-read rule synchTemperatureAndComputeResidual()
     * and finishPotentialAndApplyForce() already follow. This matters
     * more here: called every iteration, right after
     * shiftAllTimeLevels()'s device-to-device copies (issued on each
     * field's own handle, not waited on there before returning), and most
     * iterations hit no other synchronization point that would have
     * forced the wait incidentally (only ones that also land on
     * printEvery additionally checkpoint, which downloads every field).
     *
     * @param kineticEnergy Set to the volume-integrated kinetic energy.
     * @param nusselt Set to the (3D) Nusselt number.
     * @param probeVz Set to Vz at the history point probe.
     * @param probeTmpr Set to Tmpr at the history point probe.
     */
    void computeGlobalDiagnostics(Real& kineticEnergy, Real& nusselt, Real& probeVz, Real& probeTmpr);

    SimulationConfig cfg_;
    std::unique_ptr<Handle> hand_; ///< Heap-owned; see Ingredients::hand above for why.
    Mesh<Real> mesh_;
    GpuField<Real> buoyancyByX_;

    TimeDependentField<Real> temperature_;
    TimeDependentField<Real> vx_, vy_, vz_;
    SolvedField<Real> potential_;
    GpuField<Real> pressure_;
    SolvedField<Real> pressureCorrection_;

    GpuField<Real> rhsXScratch_, rhsYScratch_, rhsZScratch_;
    GpuField<Real> reductionScratch_; ///< General 1-element scratch; was allocated but unused until computeGlobalDiagnostics() -- now also backs its two sum-reductions and its two point-probe reads, reused sequentially (each download synchronously completes before the buffer's next reuse).
    GpuField<unsigned long long> maxReductionScratch_;
    GpuField<Real> referenceValueScratch_;

    int probeX_ = 0, probeY_ = 0, probeZ_ = 0; ///< History.dat's point-probe indices (Solution_time_Q2D.f90's i_x_write/j_y_write/k_z_write), computed once in the constructor from cfg_ and mesh_.

    /// @brief Closes a FILE* on destruction (nullptr is a no-op, since a
    /// unique_ptr with this deleter default-constructs to nullptr before
    /// openOutputFiles() runs). Lets serieFile_/historyFile_/convOutFile_
    /// use std::unique_ptr for automatic, move-safe cleanup -- the same
    /// reasoning hand_ (above) already uses unique_ptr<Handle> for.
    struct FileCloser { void operator()(FILE* f) const { if (f) std::fclose(f); } };
    std::unique_ptr<FILE, FileCloser> serieFile_;   ///< Every-iteration TimCur/Ek/Anu, port of Serie.dat.
    std::unique_ptr<FILE, FileCloser> historyFile_; ///< Every-iteration TimCur/probeVz/probeTmpr, port of History.dat.
    std::unique_ptr<FILE, FileCloser> convOutFile_; ///< Mirrors printIterationReport()'s/printFinalReport()'s stdout text, port of Conv.out.

    Real elapsedTime_ = Real(0);
    int iteration_ = 0;
};

#endif //CFDSOLVER_CFDSIMULATION_H
