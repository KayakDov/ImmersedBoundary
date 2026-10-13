/**
 * @file CfdSimulation.cu
 * @ingroup cfd_solver
 * @brief Implements CfdSimulation.
 *
 * @warning Not compile-verified: see GpuSolverSetup.cu's file-level
 * warning for why. This file additionally depends on every other
 * not-compile-verified file in this project (GpuSolverSetup, SolvedField,
 * TimeDependentField, Mesh, CheckpointIO) -- it is the integration point
 * for all of them, and deserves the most scrutiny of anything here once a
 * real nvcc environment is available.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "CfdSimulation.h"
#include "physics/ElementwiseOps.cuh"

namespace {

/**
 * @brief A safe, fixed-size 3D CUDA launch configuration (8x8x8 = 512
 * threads per block, well within every CUDA device's 1024-thread-per-
 * block limit) sized to cover a field of the given extents.
 */
struct LaunchConfig3D {
    dim3 blocks, threads;
    explicit LaunchConfig3D(int nx, int ny, int nz)
        : blocks((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8), threads(8, 8, 8) {}
};

template<typename Real>
Real gravitationalScale(const SimulationConfig& cfg) { return Real(1) / std::sqrt((Real)cfg.grashof); }

template<typename Real>
Real temperatureDiffusivityScale(const SimulationConfig& cfg) {
    Real g = gravitationalScale<Real>(cfg);
    return (Real)cfg.prandtl / g;
}

template<typename Real>
Real buoyancyCoefficient(const SimulationConfig& cfg) {
    Real g = gravitationalScale<Real>(cfg);
    return (Real)cfg.grashof * g * g;
}

template<typename Real>
Real electromagneticCoefficient(const SimulationConfig& cfg) {
    return gravitationalScale<Real>(cfg) * (Real)(cfg.hartmann * cfg.widthRatio) * (Real)(cfg.hartmann * cfg.widthRatio);
}

/**
 * @brief Enables peer-to-peer device memory access between every pair of
 * distinct GPUs a configuration actually uses.
 *
 * Required for correctness, not just performance: a kernel launched on
 * one field's device commonly reads another field's memory as a plain
 * parameter (temperatureAdvectionKernel, launched on Temperature's
 * device, reads VMx/VMy/VMz, which may live on a different device under
 * this project's own multi-GPU split). Without peer access enabled, that
 * is not merely slow -- it is an illegal memory access.
 * Device 0 is always included, even if no field is configured there: the
 * mesh, Teta table, and every reduction scratch buffer are allocated on
 * device 0 unconditionally, and reduction kernels launched from there
 * read field memory the same way any cross-field kernel does.
 *
 * @param cfg The simulation configuration.
 */
void enablePeerAccessForConfiguredDevices(const SimulationConfig& cfg) {
    int devices[7] = {
        0, cfg.gpuTemperature.device, cfg.gpuVx.device, cfg.gpuVy.device,
        cfg.gpuVz.device, cfg.gpuPressure.device, cfg.gpuPotential.device
    };
    for (int a = 0; a < 7; ++a) {
        for (int b = 0; b < 7; ++b) {
            if (devices[a] == devices[b]) continue;
            int canAccess = 0;
            CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&canAccess, devices[a], devices[b]));
            if (!canAccess)
                throw std::runtime_error("CfdSimulation: GPU " + std::to_string(devices[a]) +
                    " cannot access GPU " + std::to_string(devices[b]) +
                    " as a peer -- this configuration's field-to-device assignment is not usable on this hardware");
            CHECK_CUDA_ERROR(cudaSetDevice(devices[a]));
            cudaError_t result = cudaDeviceEnablePeerAccess(devices[b], 0);
            if (result != cudaSuccess && result != cudaErrorPeerAccessAlreadyEnabled)
                CHECK_CUDA_ERROR(result);
        }
    }
}

void validateImplementedFeatures(const SimulationConfig& cfg) {
    if (cfg.timeAveragingEnabled)
        throw std::runtime_error("CfdSimulation: time_averaging.enabled is not yet implemented in this port "
            "(see CfdSimulation.h's class comment) -- set it to false to proceed");
    if (cfg.runConservationChecks)
        throw std::runtime_error("CfdSimulation: diagnostics.run_conservation_checks is not yet implemented "
            "in this port (see CfdSimulation.h's class comment) -- set it to false to proceed");
}

/**
 * @brief Computes Teta's direct-formula x-lookup table (see Operators.cuh's
 * temperatureAdvection for why this bypasses the usual boundary-condition
 * machinery). Port of ConvMain_3D_Q2D.f90's `Teta(i,...) = 1 - X12(i)/AspRa`.
 * @tparam Real Floating-point type.
 * @param cfg The simulation configuration.
 * @param mesh The already-generated mesh.
 * @return Teta's value at x index 0 through nx+2 inclusive.
 */
template<typename Real>
std::vector<Real> computeBuoyancyTable(const SimulationConfig& cfg, const Mesh<Real>& mesh) {
    const std::vector<Real>& nodeHalfX = mesh.hostNodeHalfX();
    std::vector<Real> table((size_t)cfg.Nx + 3);
    for (size_t i = 0; i < table.size(); ++i)
        table[i] = Real(1) - nodeHalfX[i] / (Real)cfg.aspectRatio;
    return table;
}

template<typename Real>
AxisBoundaries temperatureBoundaries(const SimulationConfig& cfg) {
    return AxisBoundaries{cfg.temperatureBoundaryX, cfg.temperatureBoundaryY, cfg.temperatureBoundaryZ};
}

template<typename Real>
AxisBoundaries potentialBoundaries(const SimulationConfig& cfg) {
    return AxisBoundaries{cfg.potentialBoundaryX, cfg.potentialBoundaryY, cfg.potentialBoundaryZ};
}

} // namespace

template<typename Real>
typename CfdSimulation<Real>::Ingredients CfdSimulation<Real>::prepare(SimulationConfig cfg) {
    validateImplementedFeatures(cfg);

    // Heap-allocated via unique_ptr, not a stack value: Handle is
    // deliberately neither copyable nor movable (see handle.h), so it is
    // constructed once, here, and only the owning pointer ever moves --
    // into Ingredients below, and from there into CfdSimulation::hand_.
    std::unique_ptr<Handle> hand = std::make_unique<Handle>(GpuIndex(0));
    Mesh<Real> mesh = Mesh<Real>::generate(cfg, GpuIndex(0), *hand);
    enablePeerAccessForConfiguredDevices(cfg);

    GpuField<Real> buoyancyByX = GpuField<Real>::allocate(cfg.Nx + 3, 1, 1, GpuIndex(0), *hand);
    buoyancyByX.uploadFrom(computeBuoyancyTable<Real>(cfg, mesh), *hand);

    GpuField<Real> temperatureOld = GpuField<Real>::allocate(cfg.Nx + 1, cfg.Ny + 1, cfg.Nz + 1, GpuIndex(cfg.gpuTemperature.device), *hand);
    GpuField<Real> temperatureCurrent = GpuField<Real>::allocate(cfg.Nx + 1, cfg.Ny + 1, cfg.Nz + 1, GpuIndex(cfg.gpuTemperature.device), *hand);
    GpuField<Real> vxOld = GpuField<Real>::allocate(cfg.Nx, cfg.Ny + 1, cfg.Nz + 1, GpuIndex(cfg.gpuVx.device), *hand);
    GpuField<Real> vxCurrent = GpuField<Real>::allocate(cfg.Nx, cfg.Ny + 1, cfg.Nz + 1, GpuIndex(cfg.gpuVx.device), *hand);
    GpuField<Real> vyOld = GpuField<Real>::allocate(cfg.Nx + 1, cfg.Ny, cfg.Nz + 1, GpuIndex(cfg.gpuVy.device), *hand);
    GpuField<Real> vyCurrent = GpuField<Real>::allocate(cfg.Nx + 1, cfg.Ny, cfg.Nz + 1, GpuIndex(cfg.gpuVy.device), *hand);
    GpuField<Real> vzOld = GpuField<Real>::allocate(cfg.Nx + 1, cfg.Ny + 1, cfg.Nz, GpuIndex(cfg.gpuVz.device), *hand);
    GpuField<Real> vzCurrent = GpuField<Real>::allocate(cfg.Nx + 1, cfg.Ny + 1, cfg.Nz, GpuIndex(cfg.gpuVz.device), *hand);

    BuiltSolvers<Real> solvers = GpuSolverBuilder::build<Real>(cfg, mesh);

    GpuField<Real> pressure = GpuField<Real>::allocate(cfg.Nx + 1, cfg.Ny + 1, cfg.Nz + 1, GpuIndex(cfg.gpuPressure.device), *hand);
    GpuField<Real> rhsXScratch = GpuField<Real>::allocate(cfg.Nx, cfg.Ny + 1, cfg.Nz + 1, GpuIndex(cfg.gpuVx.device), *hand);
    GpuField<Real> rhsYScratch = GpuField<Real>::allocate(cfg.Nx + 1, cfg.Ny, cfg.Nz + 1, GpuIndex(cfg.gpuVy.device), *hand);
    GpuField<Real> rhsZScratch = GpuField<Real>::allocate(cfg.Nx + 1, cfg.Ny + 1, cfg.Nz, GpuIndex(cfg.gpuVz.device), *hand);
    GpuField<Real> reductionScratch = GpuField<Real>::allocate(1, 1, 1, GpuIndex(0), *hand);
    GpuField<unsigned long long> maxReductionScratch = GpuField<unsigned long long>::allocate(1, 1, 1, GpuIndex(0), *hand);
    GpuField<Real> referenceValueScratch = GpuField<Real>::allocate(1, 1, 1, GpuIndex(cfg.gpuPotential.device), *hand);

    return Ingredients{
        std::move(cfg), std::move(hand), std::move(mesh), std::move(buoyancyByX),
        std::move(temperatureOld), std::move(temperatureCurrent),
        std::move(vxOld), std::move(vxCurrent),
        std::move(vyOld), std::move(vyCurrent),
        std::move(vzOld), std::move(vzCurrent),
        std::move(solvers),
        std::move(pressure),
        std::move(rhsXScratch), std::move(rhsYScratch), std::move(rhsZScratch),
        std::move(reductionScratch), std::move(maxReductionScratch), std::move(referenceValueScratch)
    };
}

template<typename Real>
CfdSimulation<Real>::CfdSimulation(SimulationConfig cfg) : CfdSimulation(prepare(std::move(cfg))) {}

template<typename Real>
CfdSimulation<Real>::CfdSimulation(Ingredients ingredients)
    : cfg_(std::move(ingredients.cfg)),
      hand_(std::move(ingredients.hand)),
      mesh_(std::move(ingredients.mesh)),
      buoyancyByX_(std::move(ingredients.buoyancyByX)),
      temperature_(std::move(ingredients.temperatureOld), std::move(ingredients.temperatureCurrent), std::move(ingredients.solvers.temperature)),
      vx_(std::move(ingredients.vxOld), std::move(ingredients.vxCurrent), std::move(ingredients.solvers.vx)),
      vy_(std::move(ingredients.vyOld), std::move(ingredients.vyCurrent), std::move(ingredients.solvers.vy)),
      vz_(std::move(ingredients.vzOld), std::move(ingredients.vzCurrent), std::move(ingredients.solvers.vz)),
      potential_(std::move(ingredients.solvers.potential)),
      pressure_(std::move(ingredients.pressure)),
      pressureCorrection_(std::move(ingredients.solvers.pressure)),
      rhsXScratch_(std::move(ingredients.rhsXScratch)),
      rhsYScratch_(std::move(ingredients.rhsYScratch)),
      rhsZScratch_(std::move(ingredients.rhsZScratch)),
      reductionScratch_(std::move(ingredients.reductionScratch)),
      maxReductionScratch_(std::move(ingredients.maxReductionScratch)),
      referenceValueScratch_(std::move(ingredients.referenceValueScratch)) {}

template<typename Real>
void CfdSimulation<Real>::initialize() {
    if (cfg_.restartFile.has_value()) {
        elapsedTime_ = CheckpointIO::read(*cfg_.restartFile, cfg_.Nx, cfg_.Ny, cfg_.Nz,
            temperature_, vx_, vy_, vz_, pressure_, *hand_);
        return;
    }
    initializeFreshStart();
}

template<typename Real>
void CfdSimulation<Real>::initializeFreshStart() {
    std::vector<Real> smallTemperature(temperature_.oldView().size(), Real(1e-6));
    std::vector<Real> smallVx(vx_.oldView().size(), Real(10e-6));
    std::vector<Real> smallVy(vy_.oldView().size(), Real(1e-6));
    std::vector<Real> smallVz(vz_.oldView().size(), Real(1e-6));
    std::vector<Real> smallPressure((size_t)(cfg_.Nx + 1) * (cfg_.Ny + 1) * (cfg_.Nz + 1), Real(1e-6));

    temperature_.uploadOld(smallTemperature, *hand_);
    temperature_.uploadCurrent(smallTemperature, *hand_);
    vx_.uploadOld(smallVx, *hand_);
    vx_.uploadCurrent(smallVx, *hand_);
    vy_.uploadOld(smallVy, *hand_);
    vy_.uploadCurrent(smallVy, *hand_);
    vz_.uploadOld(smallVz, *hand_);
    vz_.uploadCurrent(smallVz, *hand_);
    pressure_.uploadFrom(smallPressure, *hand_);

    elapsedTime_ = Real(0);
}

template<typename Real>
void CfdSimulation<Real>::buildAndLaunchTemperature() {
    Handle& hand = temperature_.solved().handle();
    FieldView<Real> rhs = temperature_.solved().view();
    AxisBoundaries bc = temperatureBoundaries<Real>(cfg_);
    LaunchConfig3D cfg3d(rhs.nx(), rhs.ny(), rhs.nz());

    temperature_.solved().zero(hand);
    temperatureAdvectionKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(
        rhs, temperature_.currentView(), buoyancyByX_.view().data(),
        vx_.currentView(), vy_.currentView(), vz_.currentView(), mesh_.deviceSpacing(), bc);

    Real dtTemp = cfg_.temperatureStationary ? Real(0) : Real(1);
    Real ht = Real(2) * (Real)cfg_.timeStep;
    subtractTimeDerivativeKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(
        rhs, temperature_.currentView(), temperature_.oldView(), dtTemp / ht);

    scaleKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(rhs, temperatureDiffusivityScale<Real>(cfg_));

    temperature_.solved().solve();
}

template<typename Real>
void CfdSimulation<Real>::buildAndLaunchVy() {
    Handle& hand = vy_.solved().handle();
    FieldView<Real> rhs = vy_.solved().view();
    LaunchConfig3D cfg3d(rhs.nx(), rhs.ny(), rhs.nz());

    pressureGradientYKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(rhs, pressure_.view(), mesh_.deviceSpacing());
    velocityYAdvectionKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(
        rhs, vx_.currentView(), vy_.currentView(), vz_.currentView(), mesh_.deviceSpacing());

    Real ht = Real(2) * (Real)cfg_.timeStep;
    subtractTimeDerivativeKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(
        rhs, vy_.currentView(), vy_.oldView(), Real(1) / ht);

    Real inverseDGr = Real(1) / gravitationalScale<Real>(cfg_);
    scaleKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(rhs, inverseDGr);

    vy_.solved().solve();
}

template<typename Real>
void CfdSimulation<Real>::launchPotential() {
    Handle& hand = potential_.handle();
    FieldView<Real> rhs = potential_.view();
    LaunchConfig3D cfg3d(rhs.nx(), rhs.ny(), rhs.nz());
    potentialRightHandSideKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(
        rhs, vx_.currentView(), vz_.currentView(), mesh_.deviceSpacing());
    potential_.solve();
}

template<typename Real>
void CfdSimulation<Real>::buildVxVzStencilTerms() {
    Handle& handX = vx_.solved().handle();
    FieldView<Real> rhsX = rhsXScratch_.view();
    LaunchConfig3D cfgX(rhsX.nx(), rhsX.ny(), rhsX.nz());
    pressureGradientXKernel<Real><<<cfgX.blocks, cfgX.threads, 0, handX>>>(rhsX, pressure_.view(), mesh_.deviceSpacing());
    velocityXAdvectionKernel<Real><<<cfgX.blocks, cfgX.threads, 0, handX>>>(
        rhsX, vx_.currentView(), vy_.currentView(), vz_.currentView(), mesh_.deviceSpacing());
    Real ht = Real(2) * (Real)cfg_.timeStep;
    subtractTimeDerivativeKernel<Real><<<cfgX.blocks, cfgX.threads, 0, handX>>>(
        rhsX, vx_.currentView(), vx_.oldView(), Real(1) / ht);

    Handle& handZ = vz_.solved().handle();
    FieldView<Real> rhsZ = rhsZScratch_.view();
    LaunchConfig3D cfgZ(rhsZ.nx(), rhsZ.ny(), rhsZ.nz());
    pressureGradientZKernel<Real><<<cfgZ.blocks, cfgZ.threads, 0, handZ>>>(rhsZ, pressure_.view(), mesh_.deviceSpacing());
    velocityZAdvectionKernel<Real><<<cfgZ.blocks, cfgZ.threads, 0, handZ>>>(
        rhsZ, vx_.currentView(), vy_.currentView(), vz_.currentView(), mesh_.deviceSpacing());
    subtractTimeDerivativeKernel<Real><<<cfgZ.blocks, cfgZ.threads, 0, handZ>>>(
        rhsZ, vz_.currentView(), vz_.oldView(), Real(1) / ht);
}

template<typename Real>
Real CfdSimulation<Real>::synchTemperatureAndComputeResidual() {
    temperature_.solved().synch();

    FieldView<Real> newValue = temperature_.solved().view();
    FieldView<Real> oldValue = temperature_.currentView();
    LaunchConfig3D cfg3d(newValue.nx(), newValue.ny(), newValue.nz());

    maxReductionScratch_.zero(*hand_);
    maxRelativeDifferenceKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, *hand_>>>(
        newValue, oldValue, maxReductionScratch_.view().data());
    unsigned long long bits = maxReductionScratch_.downloadScalar(*hand_);
    Real result;
    static_assert(sizeof(Real) <= sizeof(unsigned long long), "Real must fit in the max-reduction accumulator");
    std::memcpy(&result, &bits, sizeof(Real));
    return result;
}

template<typename Real>
void CfdSimulation<Real>::applyBuoyancyToRhsZ() {
    Handle& hand = vz_.solved().handle();
    FieldView<Real> rhsZ = rhsZScratch_.view();
    FieldView<Real> newTemperature = temperature_.solved().view();
    LaunchConfig3D cfg3d(rhsZ.nx(), rhsZ.ny(), rhsZ.nz());

    buoyancyForceZKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(
        rhsZ, newTemperature, buoyancyByX_.view().data(), buoyancyCoefficient<Real>(cfg_));
}

template<typename Real>
void CfdSimulation<Real>::finishPotentialAndApplyForce() {
    potential_.synch();

    FieldView<Real> potentialView = potential_.view();
    AxisBoundaries potentialBc = potentialBoundaries<Real>(cfg_);
    Handle& potentialHand = potential_.handle();
    LaunchConfig3D potentialCfg(potentialView.nx(), potentialView.ny(), potentialView.nz());

    CHECK_CUDA_ERROR(cudaMemcpyAsync(referenceValueScratch_.view().data(), potentialView.data(),
        sizeof(Real), cudaMemcpyDeviceToDevice, potentialHand));
    subtractReferenceValueKernel<Real><<<potentialCfg.blocks, potentialCfg.threads, 0, potentialHand>>>(
        potentialView, referenceValueScratch_.view().data());

    Real coefficient = electromagneticCoefficient<Real>(cfg_);

    Handle& handX = vx_.solved().handle();
    FieldView<Real> rhsX = rhsXScratch_.view();
    LaunchConfig3D cfgX(rhsX.nx(), rhsX.ny(), rhsX.nz());
    applyForceXKernel<Real><<<cfgX.blocks, cfgX.threads, 0, handX>>>(
        rhsX, vx_.currentView(), potentialView, potentialBc, mesh_.deviceSpacing(), coefficient);

    Handle& handZ = vz_.solved().handle();
    FieldView<Real> rhsZ = rhsZScratch_.view();
    LaunchConfig3D cfgZ(rhsZ.nx(), rhsZ.ny(), rhsZ.nz());
    applyForceZKernel<Real><<<cfgZ.blocks, cfgZ.threads, 0, handZ>>>(
        rhsZ, vz_.currentView(), potentialView, potentialBc, mesh_.deviceSpacing(), coefficient);
}

template<typename Real>
void CfdSimulation<Real>::launchVxAndVz() {
    Real inverseDGr = Real(1) / gravitationalScale<Real>(cfg_);

    Handle& handX = vx_.solved().handle();
    FieldView<Real> rhsX = rhsXScratch_.view();
    LaunchConfig3D cfgX(rhsX.nx(), rhsX.ny(), rhsX.nz());
    scaleKernel<Real><<<cfgX.blocks, cfgX.threads, 0, handX>>>(rhsX, inverseDGr);
    FieldView<Real> solveTargetX = vx_.solved().view();
    addIntoKernel<Real><<<cfgX.blocks, cfgX.threads, 0, handX>>>(solveTargetX, rhsX);
    vx_.solved().solve();

    Handle& handZ = vz_.solved().handle();
    FieldView<Real> rhsZ = rhsZScratch_.view();
    LaunchConfig3D cfgZ(rhsZ.nx(), rhsZ.ny(), rhsZ.nz());
    scaleKernel<Real><<<cfgZ.blocks, cfgZ.threads, 0, handZ>>>(rhsZ, inverseDGr);
    FieldView<Real> solveTargetZ = vz_.solved().view();
    addIntoKernel<Real><<<cfgZ.blocks, cfgZ.threads, 0, handZ>>>(solveTargetZ, rhsZ);
    vz_.solved().solve();
}

template<typename Real>
void CfdSimulation<Real>::synchVelocities() {
    vy_.solved().synch();
    vx_.solved().synch();
    vz_.solved().synch();
}

template<typename Real>
void CfdSimulation<Real>::solvePressureCorrection() {
    Handle& hand = pressureCorrection_.handle();
    FieldView<Real> rhs = pressureCorrection_.view();
    LaunchConfig3D cfg3d(rhs.nx(), rhs.ny(), rhs.nz());

    divergenceKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(
        rhs, vx_.solved().view(), vy_.solved().view(), vz_.solved().view(), mesh_.deviceSpacing());

    Real correctionScale = Real(1.5) / (Real)cfg_.timeStep;
    scaleKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(rhs, correctionScale);

    pressureCorrection_.solve();
    pressureCorrection_.synch();
}

template<typename Real>
void CfdSimulation<Real>::correctVelocitiesAndPressure() {
    Real coefficient = (Real)cfg_.timeStep / Real(1.5);
    FieldView<Real> correction = pressureCorrection_.view();

    Handle& handX = vx_.solved().handle();
    FieldView<Real> gradX = rhsXScratch_.view();
    LaunchConfig3D cfgX(gradX.nx(), gradX.ny(), gradX.nz());
    pressureGradientXKernel<Real><<<cfgX.blocks, cfgX.threads, 0, handX>>>(gradX, correction, mesh_.deviceSpacing());
    FieldView<Real> newVx = vx_.solved().view();
    correctVelocityKernel<Real><<<cfgX.blocks, cfgX.threads, 0, handX>>>(newVx, gradX, coefficient);

    Handle& handY = vy_.solved().handle();
    FieldView<Real> gradY = rhsYScratch_.view();
    LaunchConfig3D cfgY(gradY.nx(), gradY.ny(), gradY.nz());
    pressureGradientYKernel<Real><<<cfgY.blocks, cfgY.threads, 0, handY>>>(gradY, correction, mesh_.deviceSpacing());
    FieldView<Real> newVy = vy_.solved().view();
    correctVelocityKernel<Real><<<cfgY.blocks, cfgY.threads, 0, handY>>>(newVy, gradY, coefficient);

    Handle& handZ = vz_.solved().handle();
    FieldView<Real> gradZ = rhsZScratch_.view();
    LaunchConfig3D cfgZ(gradZ.nx(), gradZ.ny(), gradZ.nz());
    pressureGradientZKernel<Real><<<cfgZ.blocks, cfgZ.threads, 0, handZ>>>(gradZ, correction, mesh_.deviceSpacing());
    FieldView<Real> newVz = vz_.solved().view();
    correctVelocityKernel<Real><<<cfgZ.blocks, cfgZ.threads, 0, handZ>>>(newVz, gradZ, coefficient);

    Handle& handP = pressureCorrection_.handle();
    FieldView<Real> pressureView = pressure_.view();
    LaunchConfig3D cfgP(pressureView.nx(), pressureView.ny(), pressureView.nz());
    addIntoKernel<Real><<<cfgP.blocks, cfgP.threads, 0, handP>>>(pressureView, correction);
}

template<typename Real>
void CfdSimulation<Real>::residuals(Real& rnsX, Real& rnsY, Real& rnsZ, Real& rdp) {
    auto maxRelativeDifference = [&](FieldView<Real> newValue, FieldView<Real> oldValue, Handle& hand) -> Real {
        LaunchConfig3D cfg3d(newValue.nx(), newValue.ny(), newValue.nz());
        maxReductionScratch_.zero(hand);
        maxRelativeDifferenceKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(
            newValue, oldValue, maxReductionScratch_.view().data());
        unsigned long long bits = maxReductionScratch_.downloadScalar(hand);
        Real result;
        std::memcpy(&result, &bits, sizeof(Real));
        return result;
    };

    rnsX = maxRelativeDifference(vx_.solved().view(), vx_.currentView(), vx_.solved().handle());
    rnsY = maxRelativeDifference(vy_.solved().view(), vy_.currentView(), vy_.solved().handle());
    rnsZ = maxRelativeDifference(vz_.solved().view(), vz_.currentView(), vz_.solved().handle());

    FieldView<Real> correction = pressureCorrection_.view();
    Handle& hand = pressureCorrection_.handle();
    LaunchConfig3D cfg3d(correction.nx(), correction.ny(), correction.nz());
    maxReductionScratch_.zero(hand);
    maxAbsoluteValueKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(correction, maxReductionScratch_.view().data());
    unsigned long long bits = maxReductionScratch_.downloadScalar(hand);
    std::memcpy(&rdp, &bits, sizeof(Real));
}

template<typename Real>
void CfdSimulation<Real>::shiftAllTimeLevels() {
    temperature_.shiftTimeLevel(temperature_.solved().handle());
    vx_.shiftTimeLevel(vx_.solved().handle());
    vy_.shiftTimeLevel(vy_.solved().handle());
    vz_.shiftTimeLevel(vz_.solved().handle());
}

template<typename Real>
void CfdSimulation<Real>::runOneTimeStep(Real& rnsX, Real& rnsY, Real& rnsZ, Real& rTmpr, Real& rdp) {
    buildAndLaunchTemperature();
    buildAndLaunchVy();
    launchPotential();
    buildVxVzStencilTerms();

    rTmpr = synchTemperatureAndComputeResidual();
    applyBuoyancyToRhsZ();
    finishPotentialAndApplyForce();

    launchVxAndVz();
    synchVelocities();

    solvePressureCorrection();
    correctVelocitiesAndPressure();
    residuals(rnsX, rnsY, rnsZ, rdp);

    shiftAllTimeLevels();
}

template<typename Real>
void CfdSimulation<Real>::run() {
    Real targetTime = elapsedTime_ + (Real)cfg_.timeInterval;
    Real timeStep = (Real)cfg_.timeStep;

    while (true) {
        ++iteration_;
        elapsedTime_ += timeStep;

        Real rnsX, rnsY, rnsZ, rTmpr, rdp;
        runOneTimeStep(rnsX, rnsY, rnsZ, rTmpr, rdp);

        bool shouldPrint = (iteration_ % cfg_.printEvery) == 0;
        if (shouldPrint) {
            printIterationReport(elapsedTime_, rnsX, rnsY, rnsZ, rTmpr, rdp);
            CheckpointIO::write("checkpoint.cfdc", cfg_.Nx, cfg_.Ny, cfg_.Nz,
                temperature_, vx_, vy_, vz_, pressure_, elapsedTime_, *hand_);
        }

        Real worstResidual = std::max(std::max(rnsX, rnsY), std::max(rnsZ, rTmpr));
        if (worstResidual < (Real)cfg_.epsConvergence) break;
        if (!(elapsedTime_ < targetTime) || iteration_ >= cfg_.maxIterations) break;
    }

    CheckpointIO::write("checkpoint.cfdc", cfg_.Nx, cfg_.Ny, cfg_.Nz,
        temperature_, vx_, vy_, vz_, pressure_, elapsedTime_, *hand_);
    printFinalReport();
}

template<typename Real>
void CfdSimulation<Real>::printIterationReport(Real timeNow, Real rnsX, Real rnsY, Real rnsZ, Real rTmpr, Real rdp) const {
    std::printf("\n  Istp=%6d   Time Step=%11.4g   Time=%15.8g   End Time=%15.8g\n",
        iteration_, (double)cfg_.timeStep, (double)timeNow, (double)(elapsedTime_ + (Real)cfg_.timeInterval));
    std::printf("  RNSx=%11.4g  RNSy=%11.4g  RNSz=%11.4g  RTmpr=%11.4g  RDP=%11.4g\n",
        (double)rnsX, (double)rnsY, (double)rnsZ, (double)rTmpr, (double)rdp);
}

template<typename Real>
void CfdSimulation<Real>::printFinalReport() const {
    Handle hand(GpuIndex(0));

    auto sumReduction = [&](auto launchKernel) -> Real {
        GpuField<Real> scratch = GpuField<Real>::allocate(1, 1, 1, GpuIndex(0), hand);
        launchKernel(scratch);
        return scratch.downloadScalar(hand);
    };

    int nx1 = cfg_.Nx + 1, ny1 = cfg_.Ny + 1, nz1 = cfg_.Nz + 1;
    Spacing<Real> spacing = mesh_.deviceSpacing();
    AxisBoundaries temperatureBc = temperatureBoundaries<Real>(cfg_);

    LaunchConfig3D volumeCfg(nx1, ny1, nz1);
    Real kineticEnergySum = sumReduction([&](GpuField<Real>& scratch) {
        kineticEnergyKernel<Real><<<volumeCfg.blocks, volumeCfg.threads, 0, hand>>>(
            vx_.currentView(), vy_.currentView(), vz_.currentView(), spacing, nx1, ny1, nz1, scratch.view().data());
    });

    LaunchConfig3D wallCfg(1, ny1, nz1);
    Real nusseltSum = sumReduction([&](GpuField<Real>& scratch) {
        nusseltKernel<Real><<<wallCfg.blocks, wallCfg.threads, 0, hand>>>(
            temperature_.currentView(), temperatureBc, spacing, ny1, nz1, scratch.view().data());
    });

    int midplaneY = (cfg_.Ny + 2) / 2;
    LaunchConfig3D lineCfg(nz1, 1, 1);
    Real nusseltMiddleSum = sumReduction([&](GpuField<Real>& scratch) {
        nusseltMiddleKernel<Real><<<lineCfg.blocks, lineCfg.threads, 0, hand>>>(
            temperature_.currentView(), temperatureBc, spacing, midplaneY, nz1, scratch.view().data());
    });

    Real nusselt = Diagnostics::nusseltFromSum(nusseltSum, (Real)cfg_.aspectRatio, (Real)cfg_.widthRatio);
    Real nusseltMiddle = Diagnostics::nusseltMiddleFromSum(nusseltMiddleSum, (Real)cfg_.aspectRatio);

    std::printf("           Nusselt number  = %.15g\n", (double)nusselt);
    std::printf("  Midplane Nusselt number  = %.15g\n", (double)nusseltMiddle);
    std::printf("            kinetic energy = %.15g\n", (double)kineticEnergySum);
}

template class CfdSimulation<double>;
template class CfdSimulation<float>;
