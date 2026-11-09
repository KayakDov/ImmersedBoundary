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
#include "setup/AverageFlow.h"

/**
 * @ingroup cfd_solver
 * @brief One cavity-convection simulation: owns its mesh, its six solved
 * fields, and the scratch buffers its timestep needs; exposes run() as the
 * only entry point a caller needs.
 *
 * One feature present in the original Fortran is deliberately not
 * implemented, and is refused with a clear error at construction rather
 * than silently ignored:
 *
 *   - diagnostics.run_conservation_checks (Icheck==0 in the original): a
 *     pure side-channel diagnostic (check_3D.f90) that recomputes FdDiv/
 *     GradPx/y/z/VgrTmp/VgrdVx/y/z a second time purely to print
 *     divergence and conservation numbers -- never affects the simulation
 *     itself, and likewise confirmed disabled everywhere this project has
 *     run.
 *
 * time_averaging.enabled (I_Fourier==0 in the original) -- a separate
 * running time-average and Fourier-mode decomposition -- IS implemented:
 * see accumulateTimeAverage()/finalizeTimeAverage(). An earlier note here
 * claimed Alex's own Conv.dat ran with I_Fourier==0 (time-averaging "on")
 * the whole time, and that every prior Fortran-vs-C++ comparison was
 * therefore comparing Fortran's time-averaged state against this port's
 * instantaneous one. That claim was checked directly against the live
 * Conv.dat these comparisons actually used and is WRONG: I_Fourier is 1
 * there (confirmed both by its literal value and by Tmpr_av.dat/Vx_av.dat/
 * Vy_av.dat/Vz_av.dat all being 0 bytes in every Fortran run so far --
 * Plot_Amplitude, the only thing that writes them, is itself gated by
 * `If(I_Fourier==0)`). Time-averaging has never actually been engaged on
 * either side of any comparison this project has run; it does not explain
 * the kinetic-energy/MaxPsi_average gap the sweep found. This feature is
 * still implemented, because Alex's source offers it as a real, working
 * toggle (so this port needs the same toggle for parity, and example_
 * config.json's time_averaging.enabled=false already matches Alex's
 * actual, demonstrated default) -- just not as a fix for that gap, whose
 * real cause is still open.
 *
 * run() also writes three of the original's own always-on output files,
 * under cfg_.outputDirectory, alongside the existing checkpoint:
 * Serie.dat and History.dat (every iteration -- port of Solution_time_
 * Q2D.f90's Ek/Anu and point-probe Write(25,*)/Write(26,*), which have no
 * Iprint gate in the original) and Conv.out (mirrors printIterationReport()'s
 * and printFinalReport()'s stdout text; its one-line header is a parameter
 * echo, not a replica of the original's own header block, whose exact
 * source wasn't available when this was written).
 *
 * Serie.dat/History.dat are unconditional *in the original*, and default
 * to on here too (cfg_.writeSerieAndHistory), but are made optional in
 * this port specifically because producing them costs something the
 * original never paid: the original's Ek/Anu were plain CPU arithmetic on
 * already-resident memory, while here computeGlobalDiagnostics() must
 * synch several device handles and run full-volume reduction kernels
 * every single iteration. Setting cfg_.writeSerieAndHistory to false skips
 * that GPU cost entirely, not just the two files -- see its doc comment in
 * SimulationConfig.h. Conv.out has no such cost (it only ever mirrors
 * values already computed for stdout) and stays unconditional.
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
     * cfg.runDuration or cfg.maxIterations is reached, or the
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

        // Time-averaging accumulators (Solution_time_Q2D.f90's VMx_Av/
        // VMy_Av/VMz_Av/Tmp_Av/Prs_Av) -- see accumulateTimeAverage().
        // Always allocated, regardless of cfg.timeAveragingEnabled: each
        // is one more full field (trivial next to the six solved fields
        // already allocated), and allocating unconditionally avoids ever
        // needing an optional/null GpuField. Only the accumulation calls
        // are conditional, in run().
        GpuField<Real> vxTimeAverage, vyTimeAverage, vzTimeAverage, temperatureTimeAverage, pressureTimeAverage;

        // Fourier-mode accumulators (Tmp_Amplitude's i>0/i<0 halves in the
        // original) -- temperature only, one pair of fields per configured
        // frequency. Empty vectors when cfg.fourierFrequencies is empty
        // (the common case: plain time-averaging with no decomposition).
        std::vector<GpuField<Real>> temperatureFourierCos, temperatureFourierSin;
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
    void residuals(XYZ<Real>& residual, Real& rdp);
    void shiftAllTimeLevels();
    void runOneTimeStep(XYZ<Real>& residual, Real& rTmpr, Real& rdp);

    void initializeFreshStart();
    void printIterationReport(Real timeNow, Real targetTime, XYZ<Real> residual, Real rTmpr, Real rdp) const;
    void printFinalReport() const;

    /**
     * @brief One iteration's contribution to the running time-average:
     * `accumulator += current * timeStep` for vx_/vy_/vz_/temperature_/
     * pressure_ (port of Fourier_sum's unconditional five lines), plus,
     * for each configured frequency, the matching cos/sin-weighted
     * temperature contribution (Fourier_sum's `Do i=1,N_Fourier` loop).
     * cos(omega*t)/sin(omega*t) are plain scalars, computed once on the
     * host per call -- no kernel needed for them, only for the per-cell
     * accumulate itself (accumulateScaledKernel).
     *
     * Synchs vx_/vy_/vz_/temperature_'s own solve handles before reading
     * their "current" views from *hand_, same rule and same reason as
     * computeGlobalDiagnostics() -- this is an entirely separate call
     * (cfg_.timeAveragingEnabled is independent of
     * cfg_.writeSerieAndHistory), so it cannot rely on that function
     * having already synched them.
     *
     * Called from run()'s loop only when cfg_.timeAveragingEnabled is
     * true, at the same point the original calls Fourier_sum: right after
     * Serie.dat/History.dat's per-iteration write (if any), using the same
     * now-current elapsedTime_ as Fortran's TimCur.
     */
    void accumulateTimeAverage();

    /**
     * @brief Finishes the time-average: divides every accumulator by
     * cfg_.runDuration (Fourier terms by `2/cfg_.runDuration`, matching
     * Plot_Amplitude's extra factor of 2), writes Vx_av.dat/Vy_av.dat/
     * Vz_av.dat from the INSTANTANEOUS (pre-substitution) velocity --
     * not a typo: Solution_time_Q2D.f90's own Point_Write calls for these
     * three (lines 214-216) pass VMxNew/VMyNew/VMzNew, not VMx_Av/VMy_Av/
     * VMz_Av, even though the file names say "_av" -- and Tmpr_av.dat from
     * the time-averaged temperature plus Teta (buoyancyByX_'s table,
     * broadcast over y/z), matching Tmpr=Tmp_Av;Tmpr=Tmpr+Teta immediately
     * before that one Point_Write call. Pres_av.dat is left empty (its
     * Point_Write call is commented out in the original too). Always also
     * writes Amplitudes.dat and Average_velocity.ddd (both unconditional
     * inside Plot_Amplitude whenever it runs at all, regardless of how
     * many frequencies are configured -- an empty cfg_.fourierFrequencies
     * just means Amplitudes.dat has only its "Average" column, matching
     * N_Fourier==0's degenerate case in the original). Finally uploads the
     * finalized time-averages over vx_/vy_/vz_/temperature_/pressure_'s
     * own current-timestep buffers in place -- Teta-free for temperature_
     * (see below).
     *
     * That last step is deliberately a destructive in-place substitution,
     * mirroring the original's own `VMx = VMx_Av; ...; Tmpr = Tmp_Av`
     * (Solution_time_Q2D.f90 line 210) -- but deliberately WITHOUT line
     * 212's following `Tmpr = Tmpr + Teta`: that addition is specific to
     * the Tmpr_av.dat/Amplitudes.dat "total temperature" display
     * convention (see above), and folding it into the live temperature_
     * buffer would double-count the buoyancy reference profile the next
     * time printFinalReport()'s Nusselt kernel reads it -- that kernel
     * already adds the same profile's contribution analytically (the same
     * `+1/AspRa` term ekinem_3D.f90's Nusselt() uses), exactly like every
     * other reader of temperature_ in this codebase, none of which expect
     * Teta baked in. (ConvMain_3D_Q2D.f90's own redundant re-substitution
     * right after Solution_time returns, which WOULD carry Teta through to
     * its report, is not replicated either -- it reads from two never-
     * declared, never-assigned scalars, `Tmpr_Av` and `VMx_Az`, typos for
     * `Tmp_Av`/`VMz_Av` that read undefined memory; harmless to skip since
     * that whole branch is unreachable in every Conv.dat this project has
     * actually seen, I_Fourier being 1, not 0, in all of them.) This keeps
     * the in-place substitution self-consistent with the rest of the
     * codebase; it is what makes the printFinalReport() call right after
     * this one, and the AverageFlow::run() call at the end of run(),
     * report the time-averaged state correctly with no changes to either
     * of those -- they simply read whatever vx_/vy_/vz_/temperature_
     * currently hold, same as always.
     *
     * Called once, from run(), only when cfg_.timeAveragingEnabled is
     * true -- after the final checkpoint write and before
     * printFinalReport(), matching the original's exact ordering (Outp,
     * then Plot_Amplitude, then the Nusselt/kinetic-energy report): the
     * checkpoint is always the true instantaneous final state, never the
     * time-average, in both the original and here.
     *
     * @note Uses cfg_.runDuration, not the actual elapsedTime_ reached,
     * matching the original's own division by Time_Interval rather than
     * TimCur. For a fresh start (the only case this project has run) these
     * are the same to within one timeStep; a restarted run spanning less
     * than the full configured interval may need this revisited.
     * @note Average_velocity.ddd's byte layout (gfortran's default 4-byte
     * sequential-unformatted record markers) is best-effort and not
     * verified against a real compiled run -- flagged again at its writer.
     */
    void finalizeTimeAverage();

    /**
     * @brief Writes Tmpr.dat/Vx.dat/Vy.dat/Vz.dat/Pres.dat/Potential.dat:
     * a direct port of Solution_time_Q2D.f90's own disabled block (lines
     * 128-133 -- `Tmpr=Tmpr+Teta` plus six Point_Write calls), which Alex
     * wrote in full and then commented out rather than never finishing, so
     * this reproduces it exactly rather than designing something new.
     *
     * Uses the instantaneous (current-timestep) field state throughout,
     * never the time-average: the original block sits before `If(I_Fourier
     * ==0) Call Plot_Amplitude` in the control flow, so even when
     * averaging is also enabled, Solution_time_Q2D.f90's own Tmpr=Tmp_Av
     * substitution (inside Plot_Amplitude, line 210) hasn't happened yet
     * at this point -- confirmed from the file directly, not assumed.
     *
     * Tmpr.dat gets the same Teta (buoyancy-reference) addition Tmpr_av.dat
     * already uses, for the same reason (the original's own `Tmpr=Tmpr+
     * Teta` immediately before its Point_Write call). Vx.dat/Vy.dat/Vz.dat
     * get none (velocity has no buoyancy term). Potential.dat is resolved
     * with BoundaryConditions::readTemperature() against its own boundary
     * config, not readVelocity() -- that function is field-agnostic
     * despite the name (confirmed from its own body: just FieldView and
     * AxisBoundaries, nothing temperature-specific), and Potential, like
     * Temperature, has a real configured Dirichlet/Neumann boundary kind,
     * unlike velocity's unconditional no-slip zero.
     *
     * Pres.dat is the one field here with no ghost resolution at all:
     * pressure_ is allocated at exactly (Nx+1)*(Ny+1)*(Nz+1) -- confirmed
     * against every other real Fortran Prs usage (init_3D.f90, outp_3D.f90
     * both loop i=1,Nx1/j=1,Ny1/k=1,Nz1), not just this one disabled call
     * -- so its real content already has no ghost layer to resolve; the
     * original's own `X12(1:Nx1)` argument slice (not the full
     * ghost-inclusive X12) is the same real-extent-only reasoning, just
     * applied to the coordinate arrays instead of the field.
     *
     * Called once, from run(), only when cfg_.writeFieldSnapshots is true
     * -- after the final checkpoint, before finalizeTimeAverage(), the
     * same relative position the original's own (disabled) block sits in.
     */
    void writeFieldSnapshots();

    /**
     * @brief Creates cfg_.outputDirectory (and any missing parent, a no-op
     * if it already exists -- unlike the original, which only ever ran
     * against a directory Alex had created by hand, this directory now
     * comes from a JSON field anyone can repoint anywhere), then opens
     * Conv.out, and -- only when cfg_.writeSerieAndHistory is true --
     * Serie.dat and History.dat too, all under that directory (append
     * mode, matching the original's position='append' -- confirmed from
     * the live project's own multi-run Serie.dat/History.dat, which hold
     * several runs' data concatenated). Writes Conv.out's one-line
     * parameter-echo header. Called once, at the top of run().
     *
     * When cfg_.writeSerieAndHistory is false, serieFile_/historyFile_ are
     * left null (never opened) -- the signal run()'s loop uses to skip
     * computeGlobalDiagnostics() and its GPU cost entirely, not just the
     * file writes.
     * @throws std::runtime_error if the directory can't be created, or any
     * file that should open can't be.
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
     * Called from run()'s loop only when cfg_.writeSerieAndHistory is
     * true -- unlike the original, where the equivalent computation was
     * free (plain CPU arithmetic, no transfer or synch), this function's
     * cost is real and recurring, so skipping the call is how that cost
     * is actually avoided, not merely how the two output files are.
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

    /// @name Time-averaging state (I_Fourier==0 in the original).
    /// Accumulators during run()'s loop, time-averages after
    /// finalizeTimeAverage() divides them -- same buffer plays both roles,
    /// matching the original's VMx_Av/.../Prs_Av exactly. Always
    /// allocated; only touched when cfg_.timeAveragingEnabled is true.
    ///@{
    GpuField<Real> vxTimeAverage_, vyTimeAverage_, vzTimeAverage_, temperatureTimeAverage_, pressureTimeAverage_;
    std::vector<GpuField<Real>> temperatureFourierCos_, temperatureFourierSin_; ///< One pair per cfg_.fourierFrequencies entry (empty if none configured).
    ///@}

    int probeX_ = 0, probeY_ = 0, probeZ_ = 0; ///< History.dat's point-probe indices (Solution_time_Q2D.f90's i_x_write/j_y_write/k_z_write), computed once in the constructor from cfg_ and mesh_.

    /// @brief Closes a FILE* on destruction (nullptr is a no-op, since a
    /// unique_ptr with this deleter default-constructs to nullptr before
    /// openOutputFiles() runs). Lets serieFile_/historyFile_/convOutFile_
    /// use std::unique_ptr for automatic, move-safe cleanup -- the same
    /// reasoning hand_ (above) already uses unique_ptr<Handle> for.
    struct FileCloser { void operator()(FILE* f) const { if (f) std::fclose(f); } };
    std::unique_ptr<FILE, FileCloser> serieFile_;   ///< Every-iteration TimCur/Ek/Anu, port of Serie.dat. Null when cfg_.writeSerieAndHistory is false (never opened).
    std::unique_ptr<FILE, FileCloser> historyFile_; ///< Every-iteration TimCur/probeVz/probeTmpr, port of History.dat. Null when cfg_.writeSerieAndHistory is false (never opened).
    std::unique_ptr<FILE, FileCloser> convOutFile_; ///< Mirrors printIterationReport()'s/printFinalReport()'s stdout text, port of Conv.out.

    Real elapsedTime_ = Real(0);
    int iteration_ = 0;
};

#endif //CFDSOLVER_CFDSIMULATION_H
