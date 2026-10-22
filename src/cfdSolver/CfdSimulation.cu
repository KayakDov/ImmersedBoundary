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

    /**
     * @brief Bytes of dynamic shared memory a block-level reduction
     * (Diagnostics.cuh's blockSumInto) needs for a launch using this
     * config: one Real per thread in the block.
     *
     * blockSumInto's shared buffer is declared `extern __shared__`, which
     * means its size is NOT fixed by that declaration -- it is whatever
     * byte count the launch's third <<<>>> argument supplies. Passing 0
     * there (as every reduction-kernel launch in this file used to)
     * allocates a zero-byte buffer, so every thread's `shared[tid] = value`
     * write is out of bounds. Any kernel that calls blockSumInto must pass
     * this, not 0, as its launch's shared-memory-size argument.
     */
    template<typename Real>
    size_t reductionSharedBytes() const {
        return (size_t)threads.x * threads.y * threads.z * sizeof(Real);
    }
};

/**
 * @brief TEMPORARY DIAGNOSTIC: prints one field's raw value at two specific
 * cells, from a single device thread. Used to check for an x<->z axis
 * transpose in Potential's curl RHS -- Potential is the only field where
 * x and z are both VariableDeltaLapl AND (in this config) both length 100,
 * so a swap wouldn't change the RHS's shape or its Sum(), only which value
 * lands at which cell. A Sum() match (already confirmed) can't catch that;
 * comparing two asymmetric points directly can.
 */
// label: 0 = "curl RHS" (potential's RHS build), 1 = "Vx", 2 = "Vz" --
// an int flag instead of a const char* because a host string literal's
// pointer is not valid to dereference from device code.
template<typename Real>
__global__ void printTwoCellsKernel(FieldView<Real> field, int iter, int label, int i0, int j0, int k0, int i1, int j1, int k1) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
        threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const char* name = (label == 1) ? "Vx" : (label == 2) ? "Vz" :
                            (label == 5) ? "Vx(post-solve,pre-correction)" : (label == 6) ? "Vz(post-solve,pre-correction)" :
                            (label == 7) ? "RHSy/DGr(pre-solve)" :
                            (label == 8) ? "Vy(post-solve)" : (label == 10) ? "pressure RHS(pre-solve)" :
                            (label == 11) ? "divergence(pre-scale)" : "curl RHS";
        printf("  [iter=%d] EM: %s(%d,%d,%d)=%.15g\n", iter, name, i0, j0, k0, (double)field.at(i0, j0, k0));
        printf("  [iter=%d] EM: %s(%d,%d,%d)=%.15g\n", iter, name, i1, j1, k1, (double)field.at(i1, j1, k1));
    }
}

// ---- TEMPORARY DIAGNOSTIC: Sum(|field|) over the whole array, single
// threaded (this only ever runs once, guarded to iteration_==1, so a slow
// serial loop over ~1e6 cells is fine). Unlike Sum() (used everywhere
// else in this investigation), Sum(|.|) can't hide a real discrepancy
// behind positive/negative cancellation -- every term is non-negative.
// The last three rounds of point-sampling (two cells of curl RHS, then
// four cells each of Vx/Vz) all landed on values so close to the
// quiescent-start noise floor (1e-5 to 1e-9) that disagreement there is
// ambiguous: real bug, or just rounding noise on a near-zero quantity?
// This is the robust tiebreaker -- if Sum(|Vx|)/Sum(|Vz|) themselves
// disagree by anything more than a rounding-level amount, the velocity
// fields really are wrong (broadly, not just at a few sampled cells) by
// the end of iteration 1, before Potential's solve is even back in the
// picture.
template<typename Real>
__global__ void printSumAbsKernel(FieldView<Real> field, int iter, int label) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
        threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        double s = 0;
        for (int i = 1; i <= field.nx(); ++i)
          for (int j = 1; j <= field.ny(); ++j)
            for (int k = 1; k <= field.nz(); ++k)
              s += fabs((double)field.at(i, j, k));
        const char* name = (label == 1) ? "Vx" : (label == 2) ? "Vz" :
                            (label == 3) ? "RHSx/DGr(pre-solve)" : (label == 4) ? "RHSz/DGr(pre-solve)" :
                            (label == 5) ? "Vx(post-solve,pre-correction)" : (label == 6) ? "Vz(post-solve,pre-correction)" :
                            (label == 7) ? "RHSy/DGr(pre-solve)" : (label == 8) ? "Vy(post-solve)" :
                            (label == 9) ? "Dprs(pressure-correction,post-solve)" :
                            (label == 10) ? "pressure RHS(pre-solve)" :
                            (label == 11) ? "divergence(pre-scale)" : "field";
        printf("  [iter=%d] EM: Sum|%s|=%.15g\n", iter, name, s);
    }
}

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

    // ---- TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): Vy/RHSy have
    // never actually been sampled in this whole investigation -- every
    // print so far only covered Vx/Vz/Potential. Vy feeds directly into
    // divergenceKernel (Dprs's RHS), so a bug here would corrupt the
    // pressure correction -- and hence BOTH Vx and Vz's final values --
    // even though Vx/Vz's own RHS-build-and-solve pipeline, checked in
    // isolation, is already proven correct to 9-10 significant figures.
    if (iteration_ == 1) {
        printSumAbsKernel<Real><<<1, 1, 0, hand>>>(rhs, iteration_, 7);
        // ---- NEW (Hartmann=1000 investigation): Vy(post-solve) breaks the
        // x<->z square-duct symmetry (Vy(x,y,z)=Vy(z,y,x)) that Fortran's
        // own output confirms holds to 13-14 digits -- by 4-7x, at the same
        // four points below, while Sum still matches. RHSy here has only
        // ever been Sum-checked, never pointwise. If these two already
        // disagree with each other, the bug is upstream of the solve (this
        // RHS build). If they agree here and only label 8 breaks, the bug
        // is inside vy_.solved().solve() itself.
        printTwoCellsKernel<Real><<<1, 1, 0, hand>>>(rhs, iteration_, 7, 10, 50, 50, 90, 50, 50);
        printTwoCellsKernel<Real><<<1, 1, 0, hand>>>(rhs, iteration_, 7, 50, 50, 10, 50, 50, 90);
    }

    vy_.solved().solve();
}

template<typename Real>
void CfdSimulation<Real>::launchPotential() {
    Handle& hand = potential_.handle();
    FieldView<Real> rhs = potential_.view();
    LaunchConfig3D cfg3d(rhs.nx(), rhs.ny(), rhs.nz());
    potentialRightHandSideKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(
        rhs, vx_.currentView(), vz_.currentView(), mesh_.deviceSpacing());

    // ---- TEMPORARY DIAGNOSTIC: two asymmetric single-cell values, to check
    // for an x<->z transpose that Sum() can't reveal (see printTwoCellsKernel's
    // doc comment above). Points chosen well inside Potential's valid range
    // (i,k in [1,Nx]=[1,100], j in [1,Ny1]=[1,101] for this config) and
    // deliberately swapped with each other.
    if (iteration_ <= 13) {
        printTwoCellsKernel<Real><<<1, 1, 0, hand>>>(rhs, iteration_, 0, 10, 5, 90, 90, 5, 10);
    }

    // ---- TEMPORARY DIAGNOSTIC (same Hartmann=1000 investigation as the one
    // in finishPotentialAndApplyForce): sample the raw curl RHS right
    // before the Poisson solve overwrites it in place. Mirrors a new
    // `Write (*,*) ' EM: curl RHS=', Sum(FDRHP(1:Nx,1:Ny1,1:Nz))` line
    // added to Get_Potential_Launch (Fortran) right before
    // `Potential_pinned = FDRHP(1:Nx,1:Ny1,1:Nz)`.
    //
    // Why this is the next question: iteration 1's EM: Potential/RHSx/RHSz
    // matched Fortran to 6-7 digits, but iteration 2's SOLVED Potential
    // doesn't (-6118.11 here vs Fortran's 692.10 -- wrong sign and almost
    // 9x the magnitude). Every formula feeding this RHS (potentialRightHandSide's
    // DVx_dz/DVz_dx, the ghost-cell reads) was checked character-for-character
    // against Get_Potential_Launch and matches. If this print agrees with
    // Fortran's new curl-RHS line (expected, given that audit) while the
    // solved Potential doesn't, the bug is inside the implicit solve
    // itself for this field specifically -- and Potential is the only field
    // whose elliptic solve uses VariableDeltaLapl on BOTH its x- and
    // z-axes at once (Vx only on x, Vz only on z), so that combination
    // (GpuSolverSetup.cu's buildFieldSolver call for `potential`) is where
    // to look next.
    if (iteration_ <= 13) {
        hand.synch();
        std::vector<Real> curlRhsHost = potential_.buffer().downloadTo(hand);
        Real sumCurlRhs = 0;
        for (Real v : curlRhsHost) sumCurlRhs += v;
        std::printf("  [iter=%d] EM: curl RHS=%.15g\n", iteration_, (double)sumCurlRhs);
    }

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

    // ---- TEMPORARY DIAGNOSTIC (remove once the Hartmann=1000 divergence ----
    // is found): mirrors the commented-out `Write (*,*) ' EM: Potential=',
    // Sum(Potential)` / `' EM: RHSx='` / `' EM: RHSz='` lines in
    // EM_forcing.f90's Get_Potential_Finish/EM_force, so iteration 1's sums
    // can be diffed directly against the Fortran reference run.
    if (iteration_ <= 13) {
        potentialHand.synch();
        handX.synch();
        handZ.synch();
        std::vector<Real> potentialHost = potential_.buffer().downloadTo(potentialHand);
        std::vector<Real> rhsXHost = rhsXScratch_.downloadTo(handX);
        std::vector<Real> rhsZHost = rhsZScratch_.downloadTo(handZ);
        Real sumPotential = 0, sumRhsX = 0, sumRhsZ = 0;
        for (Real v : potentialHost) sumPotential += v;
        for (Real v : rhsXHost) sumRhsX += v;
        for (Real v : rhsZHost) sumRhsZ += v;
        std::printf("  [iter=%d] EM: Potential=%.15g\n", iteration_, (double)sumPotential);
        std::printf("  [iter=%d] EM: RHSx=%.15g\n", iteration_, (double)sumRhsX);
        std::printf("  [iter=%d] EM: RHSz=%.15g\n", iteration_, (double)sumRhsZ);
    }
}

template<typename Real>
void CfdSimulation<Real>::launchVxAndVz() {
    Real inverseDGr = Real(1) / gravitationalScale<Real>(cfg_);

    Handle& handX = vx_.solved().handle();
    FieldView<Real> rhsX = rhsXScratch_.view();
    LaunchConfig3D cfgX(rhsX.nx(), rhsX.ny(), rhsX.nz());
    scaleKernel<Real><<<cfgX.blocks, cfgX.threads, 0, handX>>>(rhsX, inverseDGr);
    // ---- TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): Sum|Vx|/
    // Sum|Vz| after the correction step already disagree with Fortran by
    // 6-22% at iteration 1 -- well above the ~0.2% run-to-run GPU noise
    // floor -- so something is wrong before Potential/EM_force even get
    // a chance to touch Vx/Vz. This samples one step further upstream:
    // rhsX here, right after scaling by 1/DGr, is exactly what gets fed
    // into vx_.solved().solve() -- the same array as Fortran's
    // Vx_pinned = RHSx(...)/DGr, right before its own solve_eigen_decomp_d
    // call. If this already disagrees by a similar margin, the bug is
    // upstream of the solve (in VgrdVx/GradPx/the buoyancy or
    // time-derivative terms that built RHSx). If this matches but Vx
    // comes out wrong anyway, the bug is inside the solve itself --
    // EigenDecompThomas, which Vx/Vz/Vy/Temperature use and Pressure/
    // Potential do NOT, so Potential's earlier clean bill of health for
    // "the library" never actually covered this code path.
    if (iteration_ == 1) {
        printSumAbsKernel<Real><<<1, 1, 0, handX>>>(rhsX, iteration_, 3);
    }
    // FIX: vx_.solved()'s buffer is reused across timesteps and is never
    // otherwise assigned (only accumulated into) before this point, so
    // without zeroing it first, addIntoKernel below adds this step's RHS
    // on top of the PREVIOUS timestep's fully-solved Vx -- not a fresh
    // RHS. That extra term compounds every timestep and was the cause of
    // the unconditional exponential blow-up (see SolvedField::zero()'s
    // doc comment: "for a right-hand-side build that accumulates with
    // +=", which this build does, via addIntoKernel, but previously
    // without the required zero() first).
    vx_.solved().zero(handX);
    FieldView<Real> solveTargetX = vx_.solved().view();
    addIntoKernel<Real><<<cfgX.blocks, cfgX.threads, 0, handX>>>(solveTargetX, rhsX);
    vx_.solved().solve();

    Handle& handZ = vz_.solved().handle();
    FieldView<Real> rhsZ = rhsZScratch_.view();
    LaunchConfig3D cfgZ(rhsZ.nx(), rhsZ.ny(), rhsZ.nz());
    scaleKernel<Real><<<cfgZ.blocks, cfgZ.threads, 0, handZ>>>(rhsZ, inverseDGr);
    if (iteration_ == 1) {
        printSumAbsKernel<Real><<<1, 1, 0, handZ>>>(rhsZ, iteration_, 4);
    }
    // FIX: same reasoning as vx_ above -- vz_.solved()'s buffer must be
    // zeroed before this addInto, or it accumulates onto the previous
    // timestep's solved Vz instead of starting from a fresh RHS.
    vz_.solved().zero(handZ);
    FieldView<Real> solveTargetZ = vz_.solved().view();
    addIntoKernel<Real><<<cfgZ.blocks, cfgZ.threads, 0, handZ>>>(solveTargetZ, rhsZ);
    vz_.solved().solve();
}

template<typename Real>
void CfdSimulation<Real>::synchVelocities() {
    vy_.solved().synch();
    vx_.solved().synch();
    vz_.solved().synch();
    // ---- TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): this is the
    // bisection point between the two remaining suspects. Sum|RHSx/DGr| and
    // Sum|RHSz/DGr| (labels 3/4, sampled right before solve()) now match
    // Fortran to 8-9 significant figures, so the RHS build is exonerated.
    // Sum|Vx|/Sum|Vz| sampled AFTER correctVelocitiesAndPressure() (labels
    // 1/2) still disagree with Fortran by 22%/6%. This samples the exact
    // same vx_.solved()/vz_.solved() buffers at the one point in between:
    // right after the Thomas solve has finished and been synched, but
    // BEFORE solvePressureCorrection()/correctVelocitiesAndPressure() touch
    // them at all. If labels 5/6 already disagree with Fortran by a similar
    // margin, the bug is inside the implicit solve itself (EigenDecompThomas
    // -- used by Vx/Vz/Vy/Temperature, never by Pressure/Potential, so never
    // actually exercised by Potential's earlier clean bill of health). If
    // labels 5/6 MATCH Fortran, the solve is innocent and the bug is in the
    // pressure-correction step that follows (most likely the Pressure
    // field's own solve/RHS, which -- unlike Potential's -- has not yet been
    // directly audited).
    if (iteration_ == 1) {
        printSumAbsKernel<Real><<<1, 1, 0, vx_.solved().handle()>>>(vx_.solved().view(), iteration_, 5);
        printSumAbsKernel<Real><<<1, 1, 0, vz_.solved().handle()>>>(vz_.solved().view(), iteration_, 6);
        printSumAbsKernel<Real><<<1, 1, 0, vy_.solved().handle()>>>(vy_.solved().view(), iteration_, 8);
        // ---- TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): labels
        // 5/6/8's Sum(|.|) above match Fortran to 9-12 digits, but a sum
        // cannot rule out a pointwise difference that cancels in the
        // aggregate yet still lands on the wrong side of divergenceKernel's
        // neighbor-difference stencil immediately downstream of this point.
        // Same four points as the final Vx/Vz sample in
        // correctVelocitiesAndPressure(), now also sampled on Vx/Vy/Vz
        // themselves at this earlier, pre-correction moment -- directly
        // comparable, cell by cell, against Fortran's mirrored prints on
        // VMxNew/VMyNew/VMzNew right after this iteration's synch_d calls.
        printTwoCellsKernel<Real><<<1, 1, 0, vx_.solved().handle()>>>(vx_.solved().view(), iteration_, 5, 10, 50, 50, 90, 50, 50);
        printTwoCellsKernel<Real><<<1, 1, 0, vx_.solved().handle()>>>(vx_.solved().view(), iteration_, 5, 50, 50, 10, 50, 50, 90);
        printTwoCellsKernel<Real><<<1, 1, 0, vz_.solved().handle()>>>(vz_.solved().view(), iteration_, 6, 10, 50, 50, 90, 50, 50);
        printTwoCellsKernel<Real><<<1, 1, 0, vz_.solved().handle()>>>(vz_.solved().view(), iteration_, 6, 50, 50, 10, 50, 50, 90);
        printTwoCellsKernel<Real><<<1, 1, 0, vy_.solved().handle()>>>(vy_.solved().view(), iteration_, 8, 10, 50, 50, 90, 50, 50);
        printTwoCellsKernel<Real><<<1, 1, 0, vy_.solved().handle()>>>(vy_.solved().view(), iteration_, 8, 50, 50, 10, 50, 50, 90);
    }
}

template<typename Real>
void CfdSimulation<Real>::solvePressureCorrection() {
    Handle& hand = pressureCorrection_.handle();
    FieldView<Real> rhs = pressureCorrection_.view();
    LaunchConfig3D cfg3d(rhs.nx(), rhs.ny(), rhs.nz());

    divergenceKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(
        rhs, vx_.solved().view(), vy_.solved().view(), vz_.solved().view(), mesh_.deviceSpacing());

    // ---- TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): label 10
    // below (this RHS, post-scale) disagrees with Fortran by ~23.7% even
    // though every input to it is independently verified correct:
    // divergenceKernel's formula matches FDdiv character-for-character,
    // the mesh-spacing arrays match the real Subroutine Mesh, Ckor/Htime
    // matches Real(1.5)/cfg_.timeStep exactly (Htime is read straight off
    // the Fortran input deck and used nowhere as anything but the raw
    // timestep -- confirmed by direct read, it is NOT secretly 2*dt),
    // BoundaryConditions::readVelocity's unconditional zero exactly
    // matches EVDbounds's unconditional VMxNew/VMyNew/VMzNew blocks (no
    // Dirichlet/Neumann branch on velocity at all -- confirmed
    // line-by-line against the real bounds_3D.f90), and Vx/Vy/Vz's own
    // Sum(|.|) match Fortran to 9-12 digits right before this kernel reads
    // them (labels 5/6/8, synchVelocities()). With every formula and every
    // aggregate input checked, the one thing a Sum(|.|) genuinely cannot
    // rule out is a pointwise difference in Vx/Vy/Vz that cancels in their
    // own aggregate but still lands on the wrong side of a neighbor
    // difference here. This pair of prints (raw divergence, label 11,
    // before the Ckor/Htime-equivalent scale below) is directly comparable,
    // cell by cell, against Fortran's FDRHP right after Call FdDiv.
    if (iteration_ == 1) {
        printSumAbsKernel<Real><<<1, 1, 0, hand>>>(rhs, iteration_, 11);
        printTwoCellsKernel<Real><<<1, 1, 0, hand>>>(rhs, iteration_, 11, 10, 50, 50, 90, 50, 50);
        printTwoCellsKernel<Real><<<1, 1, 0, hand>>>(rhs, iteration_, 11, 50, 50, 10, 50, 50, 90);
    }

    Real correctionScale = Real(1.5) / (Real)cfg_.timeStep;
    scaleKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>(rhs, correctionScale);

    // ---- TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): Dprs
    // post-solve disagrees with Fortran by 31% (label 9), the largest
    // mismatch found yet -- bigger than Vx's 22% or Vz's 6%, and those are
    // explained by it (both read this same Dprs through GradPx/GradPz,
    // already proven correct formulas). Vx/Vy/Vz's RHS AND post-solve
    // values all matched to 9-12 digits, so this is the one piece left
    // unverified. This samples the RHS going into the Poisson solve, same
    // bisection as every other field: if this already disagrees, the bug
    // is upstream (divergenceKernel/the scale factor); if it matches,
    // the bug is inside the solve itself -- EigenDecomp3d, the one solver
    // class never yet empirically validated (used only by Pressure and
    // Potential, both all/mixed-Neumann with shift=0 -- and Potential's
    // own earlier-reported anomaly was never actually pinned on anything
    // else, just reinterpreted as downstream of Vx/Vz).
    if (iteration_ == 1) {
        printSumAbsKernel<Real><<<1, 1, 0, hand>>>(rhs, iteration_, 10);
        // Same four points as label 11 above -- same cells, before vs.
        // after this scale -- directly comparable against Fortran's
        // Pressure_pinned right after FDRHP = FDRHP * Ckor / Htime.
        printTwoCellsKernel<Real><<<1, 1, 0, hand>>>(rhs, iteration_, 10, 10, 50, 50, 90, 50, 50);
        printTwoCellsKernel<Real><<<1, 1, 0, hand>>>(rhs, iteration_, 10, 50, 50, 10, 50, 50, 90);
    }

    pressureCorrection_.solve();
    pressureCorrection_.synch();

    // ---- TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): Dprs has
    // never been sampled either. Pressure's solver construction (GridDim,
    // deltas, all-Neumann BCs, FluxLapl segments, shift=0) matches Fortran's
    // PressureHandle init exactly, and this all-Neumann system is singular
    // (per Fortran's own comment, "the library's singular-mode handling
    // must stay engaged") -- a difference in how that singular mode is
    // handled wouldn't show up as a parameter mismatch, only as a numeric
    // one here.
    if (iteration_ == 1) {
        printSumAbsKernel<Real><<<1, 1, 0, hand>>>(pressureCorrection_.view(), iteration_, 9);
    }
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

    // ---- TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation, next step):
    // iter=2's curl RHS (potentialRightHandSideKernel's output, built from
    // vx_.currentView()/vz_.currentView() -- i.e. THIS iteration's newVx/
    // newVz, post-shift) already matches Fortran's Sum() closely, but the
    // solved Potential that comes out of potential_.solve() does not
    // (wrong sign, ~9x magnitude -- see finishPotentialAndApplyForce's
    // diagnostic). Since Fortran and C++ call the identical CudaBandedLib
    // routine with identical parameters for this solve (already audited:
    // dims, deltas, segment types, Neumann flags, shift=0, thomas=false
    // all match), and Fortran's answer is right, the library itself is a
    // dead end -- the one thing that can still differ is what's actually
    // IN the RHS buffer cell-by-cell. A 2-point sample of that RHS showed
    // real disagreement, but those points turned out to sample a
    // near-zero/decaying region (bad choice, noisy). So: go one step
    // further upstream and sample newVx/newVz THEMSELVES -- the velocities
    // THIS correction step just produced, which are exactly what next
    // iteration's curl-RHS kernel will read via currentView(). Points
    // spread near the x-wall and z-wall (where VariableDeltaLapl/mesh
    // stretching concentrate resolution, i.e. where Hartmann boundary
    // layers would form) rather than clustered near one spot, to avoid
    // repeating the same mistake.
    if (iteration_ == 1) {
        printTwoCellsKernel<Real><<<1, 1, 0, handX>>>(newVx, iteration_, 1, 10, 50, 50, 90, 50, 50);
        printTwoCellsKernel<Real><<<1, 1, 0, handX>>>(newVx, iteration_, 1, 50, 50, 10, 50, 50, 90);
        printTwoCellsKernel<Real><<<1, 1, 0, handZ>>>(newVz, iteration_, 2, 10, 50, 50, 90, 50, 50);
        printTwoCellsKernel<Real><<<1, 1, 0, handZ>>>(newVz, iteration_, 2, 50, 50, 10, 50, 50, 90);
        printSumAbsKernel<Real><<<1, 1, 0, handX>>>(newVx, iteration_, 1);
        printSumAbsKernel<Real><<<1, 1, 0, handZ>>>(newVz, iteration_, 2);
    }
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
        kineticEnergyKernel<Real><<<volumeCfg.blocks, volumeCfg.threads, volumeCfg.reductionSharedBytes<Real>(), hand>>>(
            vx_.currentView(), vy_.currentView(), vz_.currentView(), spacing, nx1, ny1, nz1, scratch.view().data());
    });

    LaunchConfig3D wallCfg(1, ny1, nz1);
    Real nusseltSum = sumReduction([&](GpuField<Real>& scratch) {
        nusseltKernel<Real><<<wallCfg.blocks, wallCfg.threads, wallCfg.reductionSharedBytes<Real>(), hand>>>(
            temperature_.currentView(), temperatureBc, spacing, ny1, nz1, scratch.view().data());
    });

    int midplaneY = (cfg_.Ny + 2) / 2;
    LaunchConfig3D lineCfg(nz1, 1, 1);
    Real nusseltMiddleSum = sumReduction([&](GpuField<Real>& scratch) {
        nusseltMiddleKernel<Real><<<lineCfg.blocks, lineCfg.threads, lineCfg.reductionSharedBytes<Real>(), hand>>>(
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
