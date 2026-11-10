/**
 * @file test_nusselt_kernel_launch.cu
 * @brief GoogleTest verification of the actual CUDA kernel launches
 * (nusseltKernel, nusseltMiddleKernel) in Diagnostics.cuh, reproducing
 * their real production launch geometry -- LaunchConfig3D's fixed 8x8x8
 * thread blocks, including the degenerate-extent case -- rather than
 * calling their per-cell static methods directly on the host.
 *
 * test_diagnostics.cpp already proves Diagnostics::nusseltCell() and
 * Diagnostics::nusseltMiddleCell() compute the right formula; that part of
 * the code was never wrong. The bug fixed in Diagnostics.cuh this session
 * (see nusseltKernel's and nusseltMiddleKernel's own @bug doc comments)
 * lived entirely in the kernel wrapper's thread-guard logic: LaunchConfig3D
 * always launches 8 threads per block on every axis, even an axis whose
 * true extent is 1, so a kernel that doesn't gate on that axis's
 * threadIdx double- (or 8x-, or 64x-) counts every cell. A host-only call
 * to the static per-cell method can never exercise that launch geometry
 * at all, so this file exists specifically to close that gap: it launches
 * the real __global__ kernels, with the real degenerate-axis geometry
 * production code actually uses, and checks the result against the same
 * trusted per-cell formula summed on the host.
 *
 * This file is compiled as .cu (not .cpp) deliberately: nusseltKernel and
 * nusseltMiddleKernel are defined inside Diagnostics.cuh's `#ifdef
 * __CUDACC__` block, so only a translation unit compiled by nvcc can see
 * them at all -- a host-only .cpp (like test_diagnostics.cpp) cannot even
 * name them, let alone launch them. Running this test needs a real CUDA
 * device.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/Diagnostics.cuh"
#include "CfdTestHelpers.h"
#include "CfdTestCudaHelpers.cuh"
#include <random>
#include <vector>

/// Wraps one CUDA runtime call so a non-cudaSuccess result fails the test
/// immediately, instead of continuing on top of a half-initialized buffer.
#define CFDTEST_CUDA_CHECK(expr) ASSERT_EQ(cudaSuccess, (expr)) << "CUDA call failed: " #expr

namespace {

/**
 * @brief The exact launch geometry LaunchConfig3D (CfdSimulation.cu, an
 * anonymous-namespace type not reachable from a separate test file) uses:
 * 8 threads per block on every axis, block count ceil-divided by 8, and
 * one Real's worth of dynamic shared memory per thread in the block for
 * blockSumInto's reduction buffer -- reproduced here against the
 * documented contract (Diagnostics.cuh's own blockSumInto doc comment),
 * not by including CfdSimulation.cu's private implementation detail.
 */
struct KernelLaunchGeometry {
    dim3 blocks, threads;
    size_t sharedBytes;

    /// @param nx Requested extent on the x axis (may be a degenerate 1).
    /// @param ny Requested extent on the y axis (may be a degenerate 1).
    /// @param nz Requested extent on the z axis (may be a degenerate 1).
    template<typename Real>
    static KernelLaunchGeometry For(int nx, int ny, int nz) {
        dim3 threads(8, 8, 8);
        dim3 blocks((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
        size_t sharedBytes = (size_t)threads.x * threads.y * threads.z * sizeof(Real);
        return KernelLaunchGeometry{blocks, threads, sharedBytes};
    }
};

std::mt19937 rng(20261030);
std::uniform_real_distribution<double> dist(-1.0, 1.0);
double R() { return dist(rng); }

/**
 * @brief Shared setup for both kernel-launch tests below: a small,
 * non-multiple-of-8 temperature field and mesh spacing, mirrored on both
 * host (for the trusted per-cell reference sum) and device (for the real
 * kernel launch under test).
 *
 * The non-multiple-of-8 sizes are deliberate -- they force
 * LaunchConfig3D's fixed 8-thread blocks to include threads past the real
 * extent on every axis, which is exactly the condition the fixed bug
 * depended on.
 */
class NusseltKernelLaunchTest : public ::testing::Test {
protected:
    int nx1 = 4, ny1 = 5, nz1 = 7;
    std::vector<double> hx12, hy12, hz12, hpx, hpy, hpz, temperatureHost;
    double* temperatureDevice = nullptr;
    double* hx12D = nullptr; double* hy12D = nullptr; double* hz12D = nullptr;
    double* hpxD = nullptr; double* hpyD = nullptr; double* hpzD = nullptr;
    double* resultDevice = nullptr;
    AxisBoundaries bc{BoundaryKind::Dirichlet, BoundaryKind::Neumann, BoundaryKind::Neumann};

    void SetUp() override {
        hx12.resize(20); hy12.resize(20); hz12.resize(20);
        hpx.resize(20); hpy.resize(20); hpz.resize(20);
        for (int i = 0; i < 20; ++i) {
            hx12[i] = 0.3 + 0.1 * i; hy12[i] = 0.4 + 0.1 * i; hz12[i] = 0.5 + 0.1 * i;
            hpx[i] = 0.6 + 0.1 * i;  hpy[i] = 0.7 + 0.1 * i;  hpz[i] = 0.8 + 0.1 * i;
        }
        temperatureHost.resize((size_t)nx1 * ny1 * nz1);
        for (auto& v : temperatureHost) v = R();

        temperatureDevice = cfdtest::ToDevice(temperatureHost);
        hx12D = cfdtest::ToDevice(hx12); hy12D = cfdtest::ToDevice(hy12); hz12D = cfdtest::ToDevice(hz12);
        hpxD = cfdtest::ToDevice(hpx); hpyD = cfdtest::ToDevice(hpy); hpzD = cfdtest::ToDevice(hpz);
        ASSERT_NE(temperatureDevice, nullptr);
        ASSERT_NE(hx12D, nullptr); ASSERT_NE(hy12D, nullptr); ASSERT_NE(hz12D, nullptr);
        ASSERT_NE(hpxD, nullptr); ASSERT_NE(hpyD, nullptr); ASSERT_NE(hpzD, nullptr);
        CFDTEST_CUDA_CHECK(cudaMalloc(&resultDevice, sizeof(double)));
    }

    void TearDown() override {
        cudaFree(temperatureDevice);
        cudaFree(hx12D); cudaFree(hy12D); cudaFree(hz12D);
        cudaFree(hpxD); cudaFree(hpyD); cudaFree(hpzD);
        cudaFree(resultDevice);
    }

    Spacing<double> hostSpacing() const {
        return Spacing<double>{hx12.data(), hy12.data(), hz12.data(), hpx.data(), hpy.data(), hpz.data()};
    }

    Spacing<double> deviceSpacing() const {
        return Spacing<double>{hx12D, hy12D, hz12D, hpxD, hpyD, hpzD};
    }

    /// Zeroes the device accumulator; blockSumInto's atomicAdd requires
    /// this before every launch.
    void ZeroResult() {
        CFDTEST_CUDA_CHECK(cudaMemset(resultDevice, 0, sizeof(double)));
    }

    /// Uses EXPECT_EQ, not the ASSERT-based CFDTEST_CUDA_CHECK: this
    /// function returns a value, and ASSERT_* expands to a bare `return`
    /// on failure, which cannot appear in a non-void-returning function.
    double DownloadResult() {
        double value = 0.0;
        EXPECT_EQ(cudaSuccess, cudaMemcpy(&value, resultDevice, sizeof(double), cudaMemcpyDeviceToHost))
            << "CUDA call failed: cudaMemcpy(download)";
        return value;
    }
};

} // namespace

/**
 * @test Reproduces nusseltKernel's real production launch -- always
 * LaunchConfig3D(1, ny1, nz1), a degenerate x extent -- and checks its
 * result against Diagnostics::nusseltCell() summed directly on the host
 * (already proven correct by test_diagnostics.cpp). Before this session's
 * fix, this comparison failed by exactly 8x.
 */
TEST_F(NusseltKernelLaunchTest, MatchesHostSummedNusseltCell) {
    FieldView<double> temperatureHostView(temperatureHost.data(), nx1, ny1, nz1);
    double referenceSum = 0.0;
    for (int j = 1; j <= ny1; ++j)
        for (int k = 1; k <= nz1; ++k)
            referenceSum += Diagnostics::nusseltCell(j, k, temperatureHostView, bc, hostSpacing());

    FieldView<double> temperatureDeviceView(temperatureDevice, nx1, ny1, nz1);
    KernelLaunchGeometry geometry = KernelLaunchGeometry::For<double>(1, ny1, nz1);
    ZeroResult();
    nusseltKernel<double><<<geometry.blocks, geometry.threads, geometry.sharedBytes>>>(
        temperatureDeviceView, bc, deviceSpacing(), ny1, nz1, resultDevice);
    CFDTEST_CUDA_CHECK(cudaGetLastError());
    CFDTEST_CUDA_CHECK(cudaDeviceSynchronize());
    double kernelSum = DownloadResult();

    EXPECT_TRUE(cfdtest::IsClose(referenceSum, kernelSum))
        << "reference=" << referenceSum << " kernel=" << kernelSum
        << " ratio=" << (referenceSum != 0.0 ? kernelSum / referenceSum : 0.0)
        << " (8.0 here is exactly the bug this test targets)";
}

/**
 * @test Reproduces nusseltMiddleKernel's real production launch --
 * LaunchConfig3D(nz1, 1, 1), degenerate on BOTH y and z -- against
 * Diagnostics::nusseltMiddleCell() summed on the host. Before this
 * session's fix, this comparison failed by exactly 64x.
 */
TEST_F(NusseltKernelLaunchTest, MiddleMatchesHostSummedNusseltMiddleCell) {
    int ny2 = ny1 + 1;
    int midplaneY = ny2 / 2;

    FieldView<double> temperatureHostView(temperatureHost.data(), nx1, ny1, nz1);
    double referenceSum = 0.0;
    for (int k = 1; k <= nz1; ++k)
        referenceSum += Diagnostics::nusseltMiddleCell(k, midplaneY, temperatureHostView, bc, hostSpacing());

    FieldView<double> temperatureDeviceView(temperatureDevice, nx1, ny1, nz1);
    KernelLaunchGeometry geometry = KernelLaunchGeometry::For<double>(nz1, 1, 1);
    ZeroResult();
    nusseltMiddleKernel<double><<<geometry.blocks, geometry.threads, geometry.sharedBytes>>>(
        temperatureDeviceView, bc, deviceSpacing(), midplaneY, nz1, resultDevice);
    CFDTEST_CUDA_CHECK(cudaGetLastError());
    CFDTEST_CUDA_CHECK(cudaDeviceSynchronize());
    double kernelSum = DownloadResult();

    EXPECT_TRUE(cfdtest::IsClose(referenceSum, kernelSum))
        << "reference=" << referenceSum << " kernel=" << kernelSum
        << " ratio=" << (referenceSum != 0.0 ? kernelSum / referenceSum : 0.0)
        << " (64.0 here is exactly the bug this test targets)";
}
