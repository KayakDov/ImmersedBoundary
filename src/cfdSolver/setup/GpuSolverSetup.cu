/**
 * @file GpuSolverSetup.cu
 * @ingroup cfd_solver
 * @brief Implements GpuSolvers::build().
 *
 * Constructor call sites below were checked directly against the real
 * headers in src.zip (EigenDecomp2d.h/EigenDecomp3d.cuh/EigenDecompThomas.cuh,
 * poisson/BoundaryConfig.cuh, wrapper/LaplOperatorType.h); the three real
 * build errors this file previously produced -- a wrong include path and
 * enumerator names for eigen::LaplOperatorT, and missing includes for the
 * concrete EigenDecomp subclasses (SolvedField.h only forward-declares the
 * abstract EigenDecompSolver base) -- are fixed below. The constructor
 * argument lists themselves were already correct against the real
 * signatures and are unchanged.
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

bool isNeumann(BoundaryKind kind) { return kind == BoundaryKind::Neumann; }

template<typename Real>
SolvedField<Real> buildFieldSolver(
    GridDim dim,
    const XYZ<std::vector<Real>>& delta,
    XYZ<bool> startIsNeumann, XYZ<bool> endIsNeumann,
    XYZ<eigen::LaplOperatorT> segType,
    bool thomas, Real helmholtzShift,
    int gpuIndex
) {
    // Heap-allocated from the start, via std::unique_ptr, rather than a
    // stack Handles later moved into SolvedField: Handles is deliberately
    // neither copyable nor movable, so the pointer -- never the Handles
    // itself -- is what travels into the returned SolvedField.  Handles
    // 0-2 build the eigen data in parallel; handle SolvedField::workingHandle
    // is the one the field solves on.
    auto hands = std::make_unique<Handles>(SolvedField<Real>::numHandles, GpuIndex(gpuIndex));
    Handle& hand = (*hands)[SolvedField<Real>::workingHandle];
    std::unique_ptr<EigenDecompSolver<Real>> solver;

    buildBoundaryConfigAndLaunch<Real>(
        dim, delta, startIsNeumann, endIsNeumann,
        XYZ<Real>(0, 0, 0), XYZ<Real>(0, 0, 0),
        segType, (*hands)[0],
        [&](const auto& boundaryHost) {
            using SegXType = typename std::decay_t<decltype(boundaryHost.x)>::SegmentType;
            if (dim.layers <= 1) {
                solver = std::make_unique<EigenDecomp2d<Real>>(boundaryHost.forDevice(), *hands, helmholtzShift);
            } else if (thomas) {
                solver = std::make_unique<EigenDecompThomas<Real, SegXType>>(boundaryHost, *hands, helmholtzShift);
            } else {
                solver = std::make_unique<EigenDecomp3d<Real>>(boundaryHost.forDevice(), *hands, helmholtzShift);
            }
        }
    );
    hands->waitFor({0, 1, 2}, {SolvedField<Real>::workingHandle});

    GpuField<Real> buffer = GpuField<Real>::allocate((int)dim.cols, (int)dim.rows, (int)dim.layers, GpuIndex(gpuIndex), hand);
    return SolvedField<Real>(std::move(buffer), std::move(solver), std::move(hands));
}

} // namespace

template<typename Real>
BuiltSolvers<Real> GpuSolverBuilder::build(const SimulationConfig& cfg, const Mesh<Real>& mesh) {
    int nx = cfg.Nx, ny = cfg.Ny, nz = cfg.Nz;
    int nx1 = nx + 1, ny1 = ny + 1, nz1 = nz + 1;
    ShiftCoefficients<Real> shift = computeShiftCoefficients<Real>(cfg);

    XYZ<std::vector<Real>> deltaForCellCentred(mesh.hostHpx(), mesh.hostHpy(), mesh.hostHpz());
    XYZ<std::vector<Real>> deltaForVx(mesh.hostHx12(), mesh.hostHpy(), mesh.hostHpz());
    XYZ<std::vector<Real>> deltaForVy(mesh.hostHpx(), mesh.hostHy12(), mesh.hostHpz());
    XYZ<std::vector<Real>> deltaForVz(mesh.hostHpx(), mesh.hostHpy(), mesh.hostHz12());
    XYZ<std::vector<Real>> deltaForPotential(mesh.hostHx12(), mesh.hostHpy(), mesh.hostHz12());

    SolvedField<Real> temperature = buildFieldSolver<Real>(
        GridDim(ny1, nx1, nz1), deltaForCellCentred,
        XYZ<bool>(isNeumann(cfg.temperatureBoundaryX), isNeumann(cfg.temperatureBoundaryY), isNeumann(cfg.temperatureBoundaryZ)),
        XYZ<bool>(isNeumann(cfg.temperatureBoundaryX), isNeumann(cfg.temperatureBoundaryY), isNeumann(cfg.temperatureBoundaryZ)),
        XYZ<eigen::LaplOperatorT>(FluxLapl, FluxLapl, FluxLapl),
        cfg.gpuTemperature.useThomasAlgorithm, shift.temperature, cfg.gpuTemperature.device);

    SolvedField<Real> vx = buildFieldSolver<Real>(
        GridDim(ny1, nx, nz1), deltaForVx,
        XYZ<bool>(false, false, false), XYZ<bool>(false, false, false),
        XYZ<eigen::LaplOperatorT>(VariableDeltaLapl, FluxLapl, FluxLapl),
        cfg.gpuVx.useThomasAlgorithm, shift.velocity, cfg.gpuVx.device);

    SolvedField<Real> vy = buildFieldSolver<Real>(
        GridDim(ny, nx1, nz1), deltaForVy,
        XYZ<bool>(false, false, false), XYZ<bool>(false, false, false),
        XYZ<eigen::LaplOperatorT>(FluxLapl, VariableDeltaLapl, FluxLapl),
        cfg.gpuVy.useThomasAlgorithm, shift.velocity, cfg.gpuVy.device);

    SolvedField<Real> vz = buildFieldSolver<Real>(
        GridDim(ny1, nx1, nz), deltaForVz,
        XYZ<bool>(false, false, false), XYZ<bool>(false, false, false),
        XYZ<eigen::LaplOperatorT>(FluxLapl, FluxLapl, VariableDeltaLapl),
        cfg.gpuVz.useThomasAlgorithm, shift.velocity, cfg.gpuVz.device);

    SolvedField<Real> pressure = buildFieldSolver<Real>(
        GridDim(ny1, nx1, nz1), deltaForCellCentred,
        XYZ<bool>(true, true, true), XYZ<bool>(true, true, true),
        XYZ<eigen::LaplOperatorT>(FluxLapl, FluxLapl, FluxLapl),
        cfg.gpuPressure.useThomasAlgorithm, Real(0), cfg.gpuPressure.device);

    SolvedField<Real> potential = buildFieldSolver<Real>(
        GridDim(ny1, nx, nz), deltaForPotential,
        XYZ<bool>(isNeumann(cfg.potentialBoundaryX), isNeumann(cfg.potentialBoundaryY), isNeumann(cfg.potentialBoundaryZ)),
        XYZ<bool>(isNeumann(cfg.potentialBoundaryX), isNeumann(cfg.potentialBoundaryY), isNeumann(cfg.potentialBoundaryZ)),
        XYZ<eigen::LaplOperatorT>(VariableDeltaLapl, FluxLapl, VariableDeltaLapl),
        cfg.gpuPotential.useThomasAlgorithm, Real(0), cfg.gpuPotential.device);

    return BuiltSolvers<Real>{std::move(temperature), std::move(vx), std::move(vy),
                               std::move(vz), std::move(pressure), std::move(potential)};
}

template BuiltSolvers<double> GpuSolverBuilder::build<double>(const SimulationConfig&, const Mesh<double>&);
template BuiltSolvers<float> GpuSolverBuilder::build<float>(const SimulationConfig&, const Mesh<float>&);
