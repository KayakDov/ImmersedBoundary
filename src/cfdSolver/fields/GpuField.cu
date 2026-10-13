/**
 * @file GpuField.cu
 * @ingroup cfd_solver
 * @brief Implements GpuField.
 *
 * @warning Not compile-verified: written against the confirmed real
 * signatures of GpuPointer, Handle, and GpuIndex, but there is no CUDA
 * toolchain available in the environment this was written in.
 */

#include "GpuField.h"

template<typename Real>
GpuField<Real> GpuField<Real>::allocate(int nx, int ny, int nz, GpuIndex gpuIndex, Handle& hand) {
    hand.ensureDevice();
    size_t bytes = (size_t)nx * (size_t)ny * (size_t)nz * sizeof(Real);
    Real* raw = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&raw, bytes));
    CHECK_CUDA_ERROR(cudaMemsetAsync(raw, 0, bytes, hand));
    return GpuField(GpuPointer<Real>(raw, gpuIndex), nx, ny, nz);
}

template<typename Real>
void GpuField<Real>::checkSize(size_t providedSize) const {
    if (providedSize != size())
        throw std::invalid_argument("GpuField: host data size does not match field size");
}

template<typename Real>
void GpuField<Real>::uploadFrom(const std::vector<Real>& hostData, Handle& hand) {
    checkSize(hostData.size());
    hand.ensureDevice();
    CHECK_CUDA_ERROR(cudaMemcpyAsync(buffer_.get(), hostData.data(), size() * sizeof(Real), cudaMemcpyHostToDevice, hand));
    hand.synch();
}

template<typename Real>
void GpuField<Real>::zero(Handle& hand) {
    hand.ensureDevice();
    CHECK_CUDA_ERROR(cudaMemsetAsync(buffer_.get(), 0, size() * sizeof(Real), hand));
}

template<typename Real>
std::vector<Real> GpuField<Real>::downloadTo(Handle& hand) const {
    hand.ensureDevice();
    std::vector<Real> hostData(size());
    CHECK_CUDA_ERROR(cudaMemcpyAsync(hostData.data(), buffer_.get(), size() * sizeof(Real), cudaMemcpyDeviceToHost, hand));
    hand.synch();
    return hostData;
}

template class GpuField<double>;
template class GpuField<float>;
template class GpuField<unsigned long long>; ///< Used for Diagnostics' max-reduction scratch (see CfdSimulation.h).
