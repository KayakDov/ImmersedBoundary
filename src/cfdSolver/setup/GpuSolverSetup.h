/**
 * @file GpuSolverSetup.h
 * @ingroup cfd_solver
 * @brief GpuSolvers: builds all six field solvers from a SimulationConfig
 * and Mesh.
 *
 * Direct, fully GPU-resident replacement for AlexCudaCompatibility.f90's
 * Initialize_GPU_Solvers. Deliberately does not use EigenDecompForFortran
 * -- see SolvedField.h's class comment for why.
 */

#ifndef CFDSOLVER_GPUSOLVERSETUP_H
#define CFDSOLVER_GPUSOLVERSETUP_H

#include "../fields/SolvedField.h"
#include "Mesh.h"

/**
 * @ingroup cfd_solver
 * @brief The six eigendecomposition-solved fields for one simulation, as
 * plain movable values -- CfdSimulation distributes each one into its own
 * ownership (four into TimeDependentField wrappers, two kept directly),
 * so this is intentionally a plain data holder with no accessors of its
 * own, not a class that would compete with CfdSimulation for ownership.
 * @tparam Real Floating-point type.
 */
template<typename Real>
struct BuiltSolvers {
    SolvedField<Real> temperature, vx, vy, vz, pressure, potential;
};

/**
 * @ingroup cfd_solver
 * @brief Builds all six field solvers from a SimulationConfig and Mesh.
 * @tparam Real Floating-point type.
 */
class GpuSolverBuilder {
public:
    /**
     * @brief Builds all six solvers.
     * @tparam Real Floating-point type.
     * @param cfg The simulation configuration.
     * @param mesh The already-generated mesh (its host spacing arrays are
     * read once, here, to construct each solver's boundary configuration).
     * @return The constructed solvers.
     */
    template<typename Real>
    static BuiltSolvers<Real> build(const SimulationConfig& cfg, const Mesh<Real>& mesh);
};

#endif //CFDSOLVER_GPUSOLVERSETUP_H
