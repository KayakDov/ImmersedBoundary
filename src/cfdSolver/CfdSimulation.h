/**
 * @file CfdSimulation.h
 * @ingroup cfd_solver
 * @brief CfdSimulation: the main orchestrator, replacing ConvMain_3D_Q2D.f90,
 * Solution_time_Q2D.f90, and time_step_Q2D.f90.
 */

#ifndef CFDSOLVER_CFDSIMULATION_H
#define CFDSOLVER_CFDSIMULATION_H

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
    GpuField<Real> reductionScratch_;
    GpuField<unsigned long long> maxReductionScratch_;
    GpuField<Real> referenceValueScratch_;

    Real elapsedTime_ = Real(0);
    int iteration_ = 0;
};

#endif //CFDSOLVER_CFDSIMULATION_H
