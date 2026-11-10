/**
 * @file test_electromagnetic_force_kernel_launch.cu
 * @brief GoogleTest verification of the actual CUDA kernel launches in
 * ElectromagneticForce.cuh (potentialRightHandSideKernel,
 * subtractReferenceValueKernel, applyForceXKernel, applyForceZKernel),
 * reproducing LaunchConfig3D's real fixed 8x8x8 thread-block geometry,
 * against the same trusted per-cell static methods
 * test_electromagnetic_force.cpp already proves correct on the host.
 *
 * Like ElementwiseOps.cuh's kernels (see
 * test_elementwise_ops_kernel_launch.cu), these four each do one bounds-
 * guarded per-cell write, with no shared-memory block-sum reduction -- not
 * the nusseltKernel/nusseltMiddleKernel bug class. This file still
 * exercises their real launch geometry, including a deliberately non-
 * multiple-of-8 extent and (for applyForceX/Z) a Dirichlet/Neumann
 * boundary sweep, rather than only their host-callable static methods: a
 * host-only call can never prove the kernel wrapper's own thread-guard and
 * threadGridIndex() usage are correct.
 *
 * Compiled as .cu, like test_nusselt_kernel_launch.cu, because these
 * kernels are defined inside ElectromagneticForce.cuh's `#ifdef __CUDACC__`
 * block. Running this test needs a real CUDA device.
 */
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/ElectromagneticForce.cuh"
#include "CfdTestHelpers.h"
#include "CfdTestCudaHelpers.cuh"
#include <random>
#include <vector>

#define CFDTEST_CUDA_CHECK(expr) ASSERT_EQ(cudaSuccess, (expr)) << "CUDA call failed: " #expr

namespace {

std::mt19937 rng(20261108);
std::uniform_real_distribution<double> dist(-1.0, 1.0);
double R() { return dist(rng); }

/// @brief Random, positive mesh spacing, matching test_electromagnetic_
/// force.cpp's Sp helper -- a fresh host array plus its device upload.
struct Spacings {
    std::vector<double> hx12, hy12, hz12, hpx, hpy, hpz;
    double *hx12D = nullptr, *hy12D = nullptr, *hz12D = nullptr, *hpxD = nullptr, *hpyD = nullptr, *hpzD = nullptr;

    explicit Spacings(int n) : hx12(n + 2), hy12(n + 2), hz12(n + 2), hpx(n + 2), hpy(n + 2), hpz(n + 2) {
        for (int i = 0; i < n + 2; ++i) {
            hx12[i] = 0.5 + 0.3 * std::sin(0.37 * i + 1.0);
            hy12[i] = 0.5 + 0.3 * std::sin(0.41 * i + 2.0);
            hz12[i] = 0.5 + 0.3 * std::sin(0.29 * i + 3.0);
            hpx[i] = 0.5 + 0.3 * std::sin(0.53 * i + 4.0);
            hpy[i] = 0.5 + 0.3 * std::sin(0.61 * i + 5.0);
            hpz[i] = 0.5 + 0.3 * std::sin(0.71 * i + 6.0);
        }
    }

    Spacing<double> host() const { return Spacing<double>{hx12.data(), hy12.data(), hz12.data(), hpx.data(), hpy.data(), hpz.data()}; }
    Spacing<double> device() const { return Spacing<double>{hx12D, hy12D, hz12D, hpxD, hpyD, hpzD}; }
};

/**
 * @brief Shared fixture: a small, deliberately non-multiple-of-8 grid, so
 * LaunchConfig3D's fixed 8-thread blocks include threads past the real
 * extent on every axis. Upload() tracks every device buffer it hands out
 * so TearDown() frees them all, even if a test fails partway through.
 */
class ElectromagneticForceKernelLaunchTest : public ::testing::Test {
protected:
    int nx = 5, ny1 = 6, nz = 7; /* Nx, Ny1, Nz: this project's own grid-size convention. */
    std::vector<double*> devicePointers_;

    std::vector<double> RandomField(size_t count) {
        std::vector<double> v(count);
        for (auto& x : v) x = R();
        return v;
    }

    double* Upload(const std::vector<double>& hostData) {
        double* devicePtr = cfdtest::ToDevice(hostData);
        if (devicePtr) devicePointers_.push_back(devicePtr);
        return devicePtr;
    }

    void TearDown() override {
        for (double* ptr : devicePointers_) cudaFree(ptr);
    }
};

} // namespace

/// potentialRightHandSideKernel writes into a field shaped like Potential
/// (Nx,Ny1,Nz), reading VMx (Nx,Ny1,Nz1) and VMz (Nx1,Ny1,Nz).
TEST_F(ElectromagneticForceKernelLaunchTest, PotentialRightHandSide) {
    const int nz1 = nz + 1, nx1 = nx + 1;
    Spacings sp(std::max({nx, ny1, nz}) + 2);
    sp.hx12D = Upload(sp.hx12); sp.hy12D = Upload(sp.hy12); sp.hz12D = Upload(sp.hz12);
    sp.hpxD = Upload(sp.hpx); sp.hpyD = Upload(sp.hpy); sp.hpzD = Upload(sp.hpz);

    const auto vxHost = RandomField((size_t)nx * ny1 * nz1);
    const auto vzHost = RandomField((size_t)nx1 * ny1 * nz);
    std::vector<double> rhsRef((size_t)nx * ny1 * nz);

    FieldView<double> vxRefView(const_cast<double*>(vxHost.data()), nx, ny1, nz1);
    FieldView<double> vzRefView(const_cast<double*>(vzHost.data()), nx1, ny1, nz);
    FieldView<double> rhsRefView(rhsRef.data(), nx, ny1, nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k)
        rhsRefView.at(i, j, k) = ElectromagneticForce::potentialRightHandSide(i, j, k, vxRefView, vzRefView, sp.host());

    double* vxDevice = Upload(vxHost);
    double* vzDevice = Upload(vzHost);
    double* rhsDevice = Upload(std::vector<double>((size_t)nx * ny1 * nz, 0.0));
    ASSERT_NE(vxDevice, nullptr);
    ASSERT_NE(vzDevice, nullptr);
    ASSERT_NE(rhsDevice, nullptr);

    FieldView<double> vxDeviceView(vxDevice, nx, ny1, nz1), vzDeviceView(vzDevice, nx1, ny1, nz), rhsDeviceView(rhsDevice, nx, ny1, nz);
    const cfdtest::LaunchGeometry g = cfdtest::LaunchGeometry::For(nx, ny1, nz);
    potentialRightHandSideKernel<double><<<g.blocks, g.threads>>>(rhsDeviceView, vxDeviceView, vzDeviceView, sp.device());
    CFDTEST_CUDA_CHECK(cudaGetLastError());
    CFDTEST_CUDA_CHECK(cudaDeviceSynchronize());

    const auto rhsKernel = cfdtest::FromDevice(rhsDevice, rhsRef.size());
    cfdtest::MismatchCollector mismatches("potentialRightHandSideKernel");
    for (size_t idx = 0; idx < rhsRef.size(); ++idx) mismatches.Check(rhsRef[idx], rhsKernel[idx]);
    mismatches.Finish();
}

/// subtractReferenceValueKernel mutates a full (Nx1,Ny1,Nz1) field in
/// place, reading the reference value from a separate one-element buffer
/// (see its own doc comment on why: every thread would otherwise race on
/// reading the very cell some thread is zeroing).
TEST_F(ElectromagneticForceKernelLaunchTest, SubtractReferenceValue) {
    const int nx1 = nx + 1, nz1 = nz + 1;
    const auto potentialHost = RandomField((size_t)nx1 * ny1 * nz1);
    const double referenceValue = R();

    std::vector<double> potentialRef = potentialHost;
    FieldView<double> refView(potentialRef.data(), nx1, ny1, nz1);
    for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
        ElectromagneticForce::subtractReferenceValue(i, j, k, refView, &referenceValue);

    double* potentialDevice = Upload(potentialHost);
    double* referenceDevice = Upload(std::vector<double>{referenceValue});
    ASSERT_NE(potentialDevice, nullptr);
    ASSERT_NE(referenceDevice, nullptr);
    FieldView<double> potentialDeviceView(potentialDevice, nx1, ny1, nz1);
    const cfdtest::LaunchGeometry g = cfdtest::LaunchGeometry::For(nx1, ny1, nz1);
    subtractReferenceValueKernel<double><<<g.blocks, g.threads>>>(potentialDeviceView, referenceDevice);
    CFDTEST_CUDA_CHECK(cudaGetLastError());
    CFDTEST_CUDA_CHECK(cudaDeviceSynchronize());

    const auto potentialKernel = cfdtest::FromDevice(potentialDevice, potentialRef.size());
    cfdtest::MismatchCollector mismatches("subtractReferenceValueKernel");
    for (size_t idx = 0; idx < potentialRef.size(); ++idx) mismatches.Check(potentialRef[idx], potentialKernel[idx]);
    mismatches.Finish();
}

/// applyForceXKernel mutates RHSx (Nx,Ny1,Nz1), reading VMx (Nx,Ny1,Nz1)
/// and Potential (Nx,Ny1,Nz) across every Dirichlet/Neumann combination.
TEST_F(ElectromagneticForceKernelLaunchTest, ApplyForceX) {
    const int nz1 = nz + 1;
    Spacings sp(std::max({nx, ny1, nz}) + 2);
    sp.hx12D = Upload(sp.hx12); sp.hy12D = Upload(sp.hy12); sp.hz12D = Upload(sp.hz12);
    sp.hpxD = Upload(sp.hpx); sp.hpyD = Upload(sp.hpy); sp.hpzD = Upload(sp.hpz);
    const double coefficient = 1.7;

    const BoundaryKind bcVals[] = {BoundaryKind::Dirichlet, BoundaryKind::Neumann};
    for (auto bx : bcVals) for (auto by : bcVals) for (auto bz : bcVals) {
        const AxisBoundaries bc(bx, by, bz);
        const auto vxHost = RandomField((size_t)nx * ny1 * nz1);
        const auto potentialHost = RandomField((size_t)nx * ny1 * nz);
        const auto rhsHost = RandomField((size_t)nx * ny1 * nz1);

        std::vector<double> rhsRef = rhsHost;
        FieldView<double> rhsRefView(rhsRef.data(), nx, ny1, nz1);
        FieldView<double> vxRefView(const_cast<double*>(vxHost.data()), nx, ny1, nz1);
        FieldView<double> potRefView(const_cast<double*>(potentialHost.data()), nx, ny1, nz);
        for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz1; ++k)
            ElectromagneticForce::applyForceX(i, j, k, rhsRefView, vxRefView, potRefView, bc, sp.host(), coefficient);

        double* rhsDevice = Upload(rhsHost);
        double* vxDevice = Upload(vxHost);
        double* potentialDevice = Upload(potentialHost);
        ASSERT_NE(rhsDevice, nullptr);
        ASSERT_NE(vxDevice, nullptr);
        ASSERT_NE(potentialDevice, nullptr);

        FieldView<double> rhsDeviceView(rhsDevice, nx, ny1, nz1), vxDeviceView(vxDevice, nx, ny1, nz1), potDeviceView(potentialDevice, nx, ny1, nz);
        const cfdtest::LaunchGeometry g = cfdtest::LaunchGeometry::For(nx, ny1, nz1);
        applyForceXKernel<double><<<g.blocks, g.threads>>>(rhsDeviceView, vxDeviceView, potDeviceView, bc, sp.device(), coefficient);
        CFDTEST_CUDA_CHECK(cudaGetLastError());
        CFDTEST_CUDA_CHECK(cudaDeviceSynchronize());

        const auto rhsKernel = cfdtest::FromDevice(rhsDevice, rhsRef.size());
        cfdtest::MismatchCollector mismatches("applyForceXKernel");
        for (size_t idx = 0; idx < rhsRef.size(); ++idx) mismatches.Check(rhsRef[idx], rhsKernel[idx]);
        mismatches.Finish();
    }
}

/// applyForceZKernel mutates RHSz (Nx1,Ny1,Nz), reading VMz (Nx1,Ny1,Nz)
/// and Potential (Nx,Ny1,Nz) across every Dirichlet/Neumann combination.
TEST_F(ElectromagneticForceKernelLaunchTest, ApplyForceZ) {
    const int nx1 = nx + 1;
    Spacings sp(std::max({nx, ny1, nz}) + 2);
    sp.hx12D = Upload(sp.hx12); sp.hy12D = Upload(sp.hy12); sp.hz12D = Upload(sp.hz12);
    sp.hpxD = Upload(sp.hpx); sp.hpyD = Upload(sp.hpy); sp.hpzD = Upload(sp.hpz);
    const double coefficient = 1.7;

    const BoundaryKind bcVals[] = {BoundaryKind::Dirichlet, BoundaryKind::Neumann};
    for (auto bx : bcVals) for (auto by : bcVals) for (auto bz : bcVals) {
        const AxisBoundaries bc(bx, by, bz);
        const auto vzHost = RandomField((size_t)nx1 * ny1 * nz);
        const auto potentialHost = RandomField((size_t)nx * ny1 * nz);
        const auto rhsHost = RandomField((size_t)nx1 * ny1 * nz);

        std::vector<double> rhsRef = rhsHost;
        FieldView<double> rhsRefView(rhsRef.data(), nx1, ny1, nz);
        FieldView<double> vzRefView(const_cast<double*>(vzHost.data()), nx1, ny1, nz);
        FieldView<double> potRefView(const_cast<double*>(potentialHost.data()), nx, ny1, nz);
        for (int i = 1; i <= nx1; ++i) for (int j = 1; j <= ny1; ++j) for (int k = 1; k <= nz; ++k)
            ElectromagneticForce::applyForceZ(i, j, k, rhsRefView, vzRefView, potRefView, bc, sp.host(), coefficient);

        double* rhsDevice = Upload(rhsHost);
        double* vzDevice = Upload(vzHost);
        double* potentialDevice = Upload(potentialHost);
        ASSERT_NE(rhsDevice, nullptr);
        ASSERT_NE(vzDevice, nullptr);
        ASSERT_NE(potentialDevice, nullptr);

        FieldView<double> rhsDeviceView(rhsDevice, nx1, ny1, nz), vzDeviceView(vzDevice, nx1, ny1, nz), potDeviceView(potentialDevice, nx, ny1, nz);
        const cfdtest::LaunchGeometry g = cfdtest::LaunchGeometry::For(nx1, ny1, nz);
        applyForceZKernel<double><<<g.blocks, g.threads>>>(rhsDeviceView, vzDeviceView, potDeviceView, bc, sp.device(), coefficient);
        CFDTEST_CUDA_CHECK(cudaGetLastError());
        CFDTEST_CUDA_CHECK(cudaDeviceSynchronize());

        const auto rhsKernel = cfdtest::FromDevice(rhsDevice, rhsRef.size());
        cfdtest::MismatchCollector mismatches("applyForceZKernel");
        for (size_t idx = 0; idx < rhsRef.size(); ++idx) mismatches.Check(rhsRef[idx], rhsKernel[idx]);
        mismatches.Finish();
    }
}
