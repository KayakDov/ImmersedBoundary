/**
 * @file CfdTestCudaHelpers.cuh
 * @brief Shared CUDA-device helpers for the kernel-launch GoogleTest files
 * (test_nusselt_kernel_launch.cu, test_elementwise_ops_kernel_launch.cu,
 * test_electromagnetic_force_kernel_launch.cu).
 *
 * Kept separate from CfdTestHelpers.h, which every host-only .cpp test also
 * includes and which deliberately never pulls in <cuda_runtime.h> or needs
 * nvcc to parse -- folding these CUDA-only helpers in there would force a
 * CUDA toolchain onto every host-only test file for no reason.
 */
#ifndef CFDSOLVER_TESTS_CFDTESTCUDAHELPERS_CUH
#define CFDSOLVER_TESTS_CFDTESTCUDAHELPERS_CUH

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace cfdtest {

/// @brief Allocates device memory and copies `hostData` into it, or returns
/// nullptr on failure.
inline double* ToDevice(const std::vector<double>& hostData) {
    double* devicePtr = nullptr;
    if (cudaMalloc(&devicePtr, hostData.size() * sizeof(double)) != cudaSuccess) return nullptr;
    if (cudaMemcpy(devicePtr, hostData.data(), hostData.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    return devicePtr;
}

/// @brief Downloads `count` elements from a device buffer into a new host vector.
inline std::vector<double> FromDevice(double* devicePtr, size_t count) {
    std::vector<double> hostData(count);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(hostData.data(), devicePtr, count * sizeof(double), cudaMemcpyDeviceToHost))
        << "CUDA call failed: cudaMemcpy(download)";
    return hostData;
}

/// @brief LaunchConfig3D's real geometry for a non-reduction, full-volume
/// kernel: 8 threads per block on every axis, block count ceil-divided by
/// 8. No shared memory -- for a kernel with a shared-memory reduction
/// buffer to size (e.g. nusseltKernel), see test_nusselt_kernel_launch.cu's
/// own KernelLaunchGeometry instead.
struct LaunchGeometry {
    dim3 blocks, threads;
    static LaunchGeometry For(int nx, int ny, int nz) {
        dim3 threads(8, 8, 8);
        dim3 blocks((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
        return LaunchGeometry{blocks, threads};
    }
};

} // namespace cfdtest

#endif //CFDSOLVER_TESTS_CFDTESTCUDAHELPERS_CUH
