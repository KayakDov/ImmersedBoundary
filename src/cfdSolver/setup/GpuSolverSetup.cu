/**
 * @file GpuSolverSetup.cu
 * @ingroup cfd_solver
 * @brief Implements GpuSolvers::build().
 *
 * Every constructor call site below, and every argument to it, is checked
 * directly against the real headers: EigenDecomp2d.h, EigenDecomp3d.cuh,
 * EigenDecompThomas.cuh, poisson/BoundaryConfig.cuh, and
 * wrapper/LaplOperatorType.h.
 */

#include <cmath>
#include "GpuSolverSetup.h"
#include "wrapper/LaplOperatorType.h"
#include "poisson/BoundaryConfig.cuh"
#include "solvers/EigenDecomp/EigenDecomp2d.h"
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"
#include "solvers/EigenDecomp/EigenDecompThomas.cuh"

namespace {

using eigen::FluxLapl;
using eigen::VariableDeltaLapl;

/**
 * @brief Derived scalar coefficients shared by every field's Helmholtz
 * shift. Direct port of AlexCudaCompatibility.f90's Initialize_GPU_Solvers
 * preamble.
 * @tparam Real Floating-point type.
 */
template<typename Real>
struct ShiftCoefficients {
    Real temperature; ///< Temperature's Helmholtz shift.
    Real velocity;     ///< Every velocity component's Helmholtz shift.
};

/**
 * @brief Computes both fields' Helmholtz shift from the configuration.
 * @tparam Real Floating-point type.
 * @param cfg The simulation configuration.
 * @return Temperature's and every velocity component's shift.
 */
template<typename Real>
ShiftCoefficients<Real> computeShiftCoefficients(const SimulationConfig& cfg) {
    Real gravitationalScale = Real(1) / std::sqrt((Real)cfg.grashof);
    Real temperatureDiffusivityScale = cfg.prandtl == 0.0 ? Real(1) : (Real)cfg.prandtl / gravitationalScale;
    Real correctionCoefficient = Real(1.5);
    Real timeDerivativeFlag = cfg.temperatureStationary ? Real(0) : Real(1);

    ShiftCoefficients<Real> shift;
    shift.temperature = correctionCoefficient * timeDerivativeFlag * temperatureDiffusivityScale / (Real)cfg.timeStep;
    shift.velocity = correctionCoefficient / ((Real)cfg.timeStep * gravitationalScale);
    return shift;
}

/// @return Whether kind is the Neumann boundary kind.
bool isNeumann(BoundaryKind kind) { return kind == BoundaryKind::Neumann; }

/**
 * @brief Builds one field's eigendecomposition solver and allocates its
 * buffer.
 *
 * @tparam Real Floating-point type.
 * @param solverDim The eigensolver's own (row, col, layer) role extents --
 * not necessarily in the same order as this field's physical x/y/z storage
 * extents (see storageNx/storageNy/storageNz).
 * @param storageNx This field's physical x extent, kept explicit and
 * separate from solverDim: the CFD kernels address a normal Fortran-order
 * (x-fastest) view, while the solver uses its own permuted role
 * dimensions.
 * @param storageNy This field's physical y extent (see storageNx).
 * @param storageNz This field's physical z extent (see storageNx).
 * @param delta Per-role spacing deltas.
 * @param startIsNeumann Per-role Neumann (true) vs. Dirichlet (false) flag
 * at each role's low boundary.
 * @param endIsNeumann Per-role Neumann/Dirichlet flag at each role's high
 * boundary.
 * @param segType Per-role Laplacian segment kind.
 * @param thomas Whether to use the Thomas-algorithm solver variant.
 * @param helmholtzShift This field's Helmholtz shift.
 * @param gpuIndex Which device to build and allocate on.
 * @return The field's solved buffer, solver, and handles, ready to use.
 *
 * @note Handles is heap-allocated here, via std::unique_ptr, rather than a
 * stack value later moved into SolvedField: Handles is deliberately
 * neither copyable nor movable, so the pointer -- never the Handles
 * itself -- is what travels into the returned SolvedField. Handles 0-2
 * build the eigen data in parallel; handle SolvedField::workingHandle is
 * the one the field solves on.
 */
template<typename Real>
SolvedField<Real> buildFieldSolver(
    GridDim solverDim,
    int storageNx, int storageNy, int storageNz,
    const XYZ<std::vector<Real>>& delta,
    XYZ<bool> startIsNeumann, XYZ<bool> endIsNeumann,
    XYZ<eigen::LaplOperatorT> segType,
    bool thomas, Real helmholtzShift,
    int gpuIndex
) {
    auto hands = std::make_unique<Handles>(SolvedField<Real>::numHandles, GpuIndex(gpuIndex));
    Handle& hand = (*hands)[SolvedField<Real>::workingHandle];
    std::unique_ptr<EigenDecompSolver<Real>> solver;

    buildBoundaryConfigAndLaunch<Real>(
        solverDim, delta, startIsNeumann, endIsNeumann,
        XYZ<Real>(0, 0, 0), XYZ<Real>(0, 0, 0),
        segType, (*hands)[0],
        [&](const auto& boundaryHost) {
            using SegZType = typename std::decay_t<decltype(boundaryHost.z)>::SegmentType;
            if (solverDim.layers <= 1) {
                solver = std::make_unique<EigenDecomp2d<Real>>(boundaryHost.forDevice(), *hands, helmholtzShift);
            } else if (thomas) {
                solver = std::make_unique<EigenDecompThomas<Real, SegZType>>(boundaryHost, *hands, helmholtzShift);
            } else {
                solver = std::make_unique<EigenDecomp3d<Real>>(boundaryHost.forDevice(), *hands, helmholtzShift);
            }
        }
    );
    hands->waitFor({0, 1, 2}, {SolvedField<Real>::workingHandle});

    GpuField<Real> buffer = GpuField<Real>::allocate(storageNx, storageNy, storageNz, GpuIndex(gpuIndex), hand);

    // This function runs once per field (6x total, never in the
    // time-stepping loop), each call spinning up its own fresh multi-stream
    // Handles. hands->waitFor() above only joins this field's own streams
    // against each other -- it does not order this field's temporary GPU
    // work against the next field's, since that work runs on a different
    // Handles entirely. This cudaDeviceSynchronize() drains it before the
    // next field's construction begins, so one field's setup never races
    // the next field's.
    cudaDeviceSynchronize();

    return SolvedField<Real>(std::move(buffer), std::move(solver), std::move(hands));
}

} // namespace

/**
 * @brief Builds every field's eigendecomposition solver and GPU buffer for
 * one simulation.
 * @tparam Real Floating-point type.
 * @param cfg The simulation configuration.
 * @param mesh The mesh providing each axis's spacing.
 * @return The temperature, vx, vy, vz, pressure, and potential solvers,
 * ready to use.
 *
 * @note Every field's solver uses three named roles -- X-role (GridDim's
 * cols), Y-role (rows), and Z-role (layers) -- chosen so that the (row,
 * col, layer) GridDim matches the CFD/Fortran (x, y, z) field layout:
 * physical x <- Y-role/rows, physical y <- Z-role/layers, physical z <-
 * X-role/cols. Delta vectors, boundary Neumann/Dirichlet flags, and
 * Laplacian segment kinds are correspondingly built in (physical z,
 * physical x, physical y) order. Potential's solver roles follow the
 * same permutation, applied to its own physical extent (Nx, Ny1, Nz):
 * X-role (cols) <- physical Z (Hz12, VariableDeltaLapl), Y-role (rows)
 * <- physical X (Hx12, VariableDeltaLapl), Z-role (layers) <- physical Y
 * (HPy, FluxLapl).
 */
template<typename Real>
BuiltSolvers<Real> GpuSolverBuilder::build(const SimulationConfig& cfg, const Mesh<Real>& mesh) {
    int nx = cfg.Nx, ny = cfg.Ny, nz = cfg.Nz;
    int nx1 = nx + 1, ny1 = ny + 1, nz1 = nz + 1;
    ShiftCoefficients<Real> shift = computeShiftCoefficients<Real>(cfg);

    XYZ<std::vector<Real>> deltaForCellCentred(mesh.hostHpz(), mesh.hostHpx(), mesh.hostHpy());
    XYZ<std::vector<Real>> deltaForVx(mesh.hostHpz(), mesh.hostHx12(), mesh.hostHpy());
    XYZ<std::vector<Real>> deltaForVy(mesh.hostHpz(), mesh.hostHpx(), mesh.hostHy12());
    XYZ<std::vector<Real>> deltaForVz(mesh.hostHz12(), mesh.hostHpx(), mesh.hostHpy());

    XYZ<std::vector<Real>> deltaForPotential(mesh.hostHz12(), mesh.hostHx12(), mesh.hostHpy());

    SolvedField<Real> temperature = buildFieldSolver<Real>(
        GridDim(nx1, nz1, ny1), nx1, ny1, nz1, deltaForCellCentred,
        XYZ<bool>(isNeumann(cfg.temperatureBoundaryZ), isNeumann(cfg.temperatureBoundaryX), isNeumann(cfg.temperatureBoundaryY)),
        XYZ<bool>(isNeumann(cfg.temperatureBoundaryZ), isNeumann(cfg.temperatureBoundaryX), isNeumann(cfg.temperatureBoundaryY)),
        XYZ<eigen::LaplOperatorT>(FluxLapl, FluxLapl, FluxLapl),
        cfg.gpuTemperature.useThomasAlgorithm, shift.temperature, cfg.gpuTemperature.device);

    SolvedField<Real> vx = buildFieldSolver<Real>(
        GridDim(nx, nz1, ny1), nx, ny1, nz1, deltaForVx,
        XYZ<bool>(false, false, false), XYZ<bool>(false, false, false),
        XYZ<eigen::LaplOperatorT>(FluxLapl, VariableDeltaLapl, FluxLapl),
        cfg.gpuVx.useThomasAlgorithm, shift.velocity, cfg.gpuVx.device);

    SolvedField<Real> vy = buildFieldSolver<Real>(
        GridDim(nx1, nz1, ny), nx1, ny, nz1, deltaForVy,
        XYZ<bool>(false, false, false), XYZ<bool>(false, false, false),
        XYZ<eigen::LaplOperatorT>(FluxLapl, FluxLapl, VariableDeltaLapl),
        cfg.gpuVy.useThomasAlgorithm, shift.velocity, cfg.gpuVy.device);

    SolvedField<Real> vz = buildFieldSolver<Real>(
        GridDim(nx1, nz, ny1), nx1, ny1, nz, deltaForVz,
        XYZ<bool>(false, false, false), XYZ<bool>(false, false, false),
        XYZ<eigen::LaplOperatorT>(VariableDeltaLapl, FluxLapl, FluxLapl),
        cfg.gpuVz.useThomasAlgorithm, shift.velocity, cfg.gpuVz.device);

    SolvedField<Real> pressure = buildFieldSolver<Real>(
        GridDim(nx1, nz1, ny1), nx1, ny1, nz1, deltaForCellCentred,
        XYZ<bool>(true, true, true), XYZ<bool>(true, true, true),
        XYZ<eigen::LaplOperatorT>(FluxLapl, FluxLapl, FluxLapl),
        cfg.gpuPressure.useThomasAlgorithm, Real(0), cfg.gpuPressure.device);

    SolvedField<Real> potential = buildFieldSolver<Real>(
        GridDim(nx, nz, ny1), nx, ny1, nz, deltaForPotential,
        XYZ<bool>(isNeumann(cfg.potentialBoundaryZ), isNeumann(cfg.potentialBoundaryX), isNeumann(cfg.potentialBoundaryY)),
        XYZ<bool>(isNeumann(cfg.potentialBoundaryZ), isNeumann(cfg.potentialBoundaryX), isNeumann(cfg.potentialBoundaryY)),
        XYZ<eigen::LaplOperatorT>(VariableDeltaLapl, VariableDeltaLapl, FluxLapl),
        cfg.gpuPotential.useThomasAlgorithm, Real(0), cfg.gpuPotential.device);

    return BuiltSolvers<Real>{std::move(temperature), std::move(vx), std::move(vy),
                               std::move(vz), std::move(pressure), std::move(potential)};
}

template BuiltSolvers<double> GpuSolverBuilder::build<double>(const SimulationConfig&, const Mesh<double>&);
template BuiltSolvers<float> GpuSolverBuilder::build<float>(const SimulationConfig&, const Mesh<float>&);
