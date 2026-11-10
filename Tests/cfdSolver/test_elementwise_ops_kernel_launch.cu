/**
 * @file test_elementwise_ops_kernel_launch.cu
 * @brief GoogleTest verification of the actual CUDA kernel launches in
 * ElementwiseOps.cuh (subtractTimeDerivativeKernel, scaleKernel,
 * addIntoKernel, correctVelocityKernel, accumulateScaledKernel),
 * reproducing LaunchConfig3D's real fixed 8x8x8 thread-block geometry,
 * against the same trusted per-cell static methods
 * test_elementwise_ops.cpp already proves correct on the host.
 *
 * Unlike nusseltKernel/nusseltMiddleKernel (see
 * test_nusselt_kernel_launch.cu), these five kernels each do one bounds-
 * guarded per-cell WRITE, with no shared-memory block-sum reduction -- so
 * they are not subject to that same degenerate-axis over-counting bug
 * class. This file still exercises their real launch geometry, including a
 * deliberately non-multiple-of-8 extent, rather than only their host-
 * callable static methods: a host-only call can never prove the kernel
 * wrapper's own thread-guard and threadGridIndex() usage are correct.
 *
 * Compiled as .cu, like test_nusselt_kernel_launch.cu, because these
 * kernels are defined inside ElementwiseOps.cuh's `#ifdef __CUDACC__`
 * block. Running this test needs a real CUDA device.
 */
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "physics/ElementwiseOps.cuh"
#include "CfdTestHelpers.h"
#include "CfdTestCudaHelpers.cuh"
#include <random>
#include <vector>

#define CFDTEST_CUDA_CHECK(expr) ASSERT_EQ(cudaSuccess, (expr)) << "CUDA call failed: " #expr

namespace {

std::mt19937 rng(20261108);
std::uniform_real_distribution<double> dist(-1.0, 1.0);
double R() { return dist(rng); }

/**
 * @brief Shared fixture: a small, deliberately non-multiple-of-8 grid, so
 * LaunchConfig3D's fixed 8-thread blocks include threads past the real
 * extent on every axis -- the real production condition on almost every
 * launch, not a corner case. Upload() tracks every device buffer it hands
 * out so TearDown() frees them all, even if a test fails partway through.
 */
class ElementwiseOpsKernelLaunchTest : public ::testing::Test {
protected:
    int nx = 5, ny = 7, nz = 6;
    std::vector<double*> devicePointers_;

    size_t Count() const { return (size_t)nx * (size_t)ny * (size_t)nz; }
    cfdtest::LaunchGeometry Geometry() const { return cfdtest::LaunchGeometry::For(nx, ny, nz); }

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

TEST_F(ElementwiseOpsKernelLaunchTest, SubtractTimeDerivative) {
    std::vector<double> rhsHost(Count()), currentHost(Count()), oldHost(Count());
    for (auto& v : rhsHost) v = R();
    for (auto& v : currentHost) v = R();
    for (auto& v : oldHost) v = R();
    const double coefficient = 0.1 + std::abs(R());

    std::vector<double> rhsRef = rhsHost;
    FieldView<double> rhsRefView(rhsRef.data(), nx, ny, nz);
    FieldView<double> currentRefView(currentHost.data(), nx, ny, nz), oldRefView(oldHost.data(), nx, ny, nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz; ++k)
        ElementwiseOps::subtractTimeDerivative(i, j, k, rhsRefView, currentRefView, oldRefView, coefficient);

    double* rhsDevice = Upload(rhsHost);
    double* currentDevice = Upload(currentHost);
    double* oldDevice = Upload(oldHost);
    ASSERT_NE(rhsDevice, nullptr);
    ASSERT_NE(currentDevice, nullptr);
    ASSERT_NE(oldDevice, nullptr);

    FieldView<double> rhsDeviceView(rhsDevice, nx, ny, nz), currentDeviceView(currentDevice, nx, ny, nz), oldDeviceView(oldDevice, nx, ny, nz);
    const cfdtest::LaunchGeometry g = Geometry();
    subtractTimeDerivativeKernel<double><<<g.blocks, g.threads>>>(rhsDeviceView, currentDeviceView, oldDeviceView, coefficient);
    CFDTEST_CUDA_CHECK(cudaGetLastError());
    CFDTEST_CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<double> rhsKernel = cfdtest::FromDevice(rhsDevice, Count());
    cfdtest::MismatchCollector mismatches("subtractTimeDerivativeKernel");
    for (size_t idx = 0; idx < rhsRef.size(); ++idx) mismatches.Check(rhsRef[idx], rhsKernel[idx]);
    mismatches.Finish();
}

TEST_F(ElementwiseOpsKernelLaunchTest, Scale) {
    std::vector<double> fieldHost(Count());
    for (auto& v : fieldHost) v = R();
    const double factor = 0.1 + std::abs(R());

    std::vector<double> fieldRef = fieldHost;
    FieldView<double> refView(fieldRef.data(), nx, ny, nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz; ++k)
        ElementwiseOps::scale(i, j, k, refView, factor);

    double* fieldDevice = Upload(fieldHost);
    ASSERT_NE(fieldDevice, nullptr);
    FieldView<double> deviceView(fieldDevice, nx, ny, nz);
    const cfdtest::LaunchGeometry g = Geometry();
    scaleKernel<double><<<g.blocks, g.threads>>>(deviceView, factor);
    CFDTEST_CUDA_CHECK(cudaGetLastError());
    CFDTEST_CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<double> fieldKernel = cfdtest::FromDevice(fieldDevice, Count());
    cfdtest::MismatchCollector mismatches("scaleKernel");
    for (size_t idx = 0; idx < fieldRef.size(); ++idx) mismatches.Check(fieldRef[idx], fieldKernel[idx]);
    mismatches.Finish();
}

TEST_F(ElementwiseOpsKernelLaunchTest, AddInto) {
    std::vector<double> targetHost(Count()), sourceHost(Count());
    for (auto& v : targetHost) v = R();
    for (auto& v : sourceHost) v = R();

    std::vector<double> targetRef = targetHost;
    FieldView<double> refView(targetRef.data(), nx, ny, nz), sourceRefView(sourceHost.data(), nx, ny, nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz; ++k)
        ElementwiseOps::addInto(i, j, k, refView, sourceRefView);

    double* targetDevice = Upload(targetHost);
    double* sourceDevice = Upload(sourceHost);
    ASSERT_NE(targetDevice, nullptr);
    ASSERT_NE(sourceDevice, nullptr);
    FieldView<double> targetDeviceView(targetDevice, nx, ny, nz), sourceDeviceView(sourceDevice, nx, ny, nz);
    const cfdtest::LaunchGeometry g = Geometry();
    addIntoKernel<double><<<g.blocks, g.threads>>>(targetDeviceView, sourceDeviceView);
    CFDTEST_CUDA_CHECK(cudaGetLastError());
    CFDTEST_CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<double> targetKernel = cfdtest::FromDevice(targetDevice, Count());
    cfdtest::MismatchCollector mismatches("addIntoKernel");
    for (size_t idx = 0; idx < targetRef.size(); ++idx) mismatches.Check(targetRef[idx], targetKernel[idx]);
    mismatches.Finish();
}

TEST_F(ElementwiseOpsKernelLaunchTest, CorrectVelocity) {
    std::vector<double> velocityHost(Count()), gradientHost(Count());
    for (auto& v : velocityHost) v = R();
    for (auto& v : gradientHost) v = R();
    const double coefficient = 0.1 + std::abs(R());

    std::vector<double> velocityRef = velocityHost;
    FieldView<double> refView(velocityRef.data(), nx, ny, nz), gradientRefView(gradientHost.data(), nx, ny, nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz; ++k)
        ElementwiseOps::correctVelocity(i, j, k, refView, gradientRefView, coefficient);

    double* velocityDevice = Upload(velocityHost);
    double* gradientDevice = Upload(gradientHost);
    ASSERT_NE(velocityDevice, nullptr);
    ASSERT_NE(gradientDevice, nullptr);
    FieldView<double> velocityDeviceView(velocityDevice, nx, ny, nz), gradientDeviceView(gradientDevice, nx, ny, nz);
    const cfdtest::LaunchGeometry g = Geometry();
    correctVelocityKernel<double><<<g.blocks, g.threads>>>(velocityDeviceView, gradientDeviceView, coefficient);
    CFDTEST_CUDA_CHECK(cudaGetLastError());
    CFDTEST_CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<double> velocityKernel = cfdtest::FromDevice(velocityDevice, Count());
    cfdtest::MismatchCollector mismatches("correctVelocityKernel");
    for (size_t idx = 0; idx < velocityRef.size(); ++idx) mismatches.Check(velocityRef[idx], velocityKernel[idx]);
    mismatches.Finish();
}

TEST_F(ElementwiseOpsKernelLaunchTest, AccumulateScaled) {
    std::vector<double> targetHost(Count()), sourceHost(Count());
    for (auto& v : targetHost) v = R();
    for (auto& v : sourceHost) v = R();
    const double factor = 0.1 + std::abs(R());

    std::vector<double> targetRef = targetHost;
    FieldView<double> refView(targetRef.data(), nx, ny, nz), sourceRefView(sourceHost.data(), nx, ny, nz);
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz; ++k)
        ElementwiseOps::accumulateScaled(i, j, k, refView, sourceRefView, factor);

    double* targetDevice = Upload(targetHost);
    double* sourceDevice = Upload(sourceHost);
    ASSERT_NE(targetDevice, nullptr);
    ASSERT_NE(sourceDevice, nullptr);
    FieldView<double> targetDeviceView(targetDevice, nx, ny, nz), sourceDeviceView(sourceDevice, nx, ny, nz);
    const cfdtest::LaunchGeometry g = Geometry();
    accumulateScaledKernel<double><<<g.blocks, g.threads>>>(targetDeviceView, sourceDeviceView, factor);
    CFDTEST_CUDA_CHECK(cudaGetLastError());
    CFDTEST_CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<double> targetKernel = cfdtest::FromDevice(targetDevice, Count());
    cfdtest::MismatchCollector mismatches("accumulateScaledKernel");
    for (size_t idx = 0; idx < targetRef.size(); ++idx) mismatches.Check(targetRef[idx], targetKernel[idx]);
    mismatches.Finish();
}
