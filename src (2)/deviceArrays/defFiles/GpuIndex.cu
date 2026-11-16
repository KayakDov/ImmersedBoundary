/**
 * @file GpuIndex.cu
 * @brief Implementation of GpuIndex::ensureDevice().
 */

#include "../headers/Support/GpuIndex.cuh"
#include "deviceArrays/headers/handle.h" // for CHECK_CUDA_ERROR only -- no type dependency on Handle

void GpuIndex::switchDevice() const {
    thread_local int cached_device = -1;
    if (cached_device != index_) {
        CHECK_CUDA_ERROR(cudaSetDevice(index_));
        cached_device = index_;
    }
}
