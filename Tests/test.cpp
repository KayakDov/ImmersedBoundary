#include <gtest/gtest.h>
#include <cuda_runtime.h>

TEST(CudaSanity, DeviceExists) {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    ASSERT_EQ(error, cudaSuccess) << "No CUDA driver found!";
    EXPECT_GT(deviceCount, 0);
}

TEST(LibrarySanity, LinkingWorks) {
    EXPECT_TRUE(true);
}