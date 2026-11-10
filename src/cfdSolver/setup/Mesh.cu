/**
 * @file Mesh.cu
 * @ingroup cfd_solver
 * @brief Implements Mesh::generate().
 *
 * Compile-verified with nvcc; not run, since no physical GPU is available
 * in the environment this was checked in. The per-axis math
 * (computeAxisSpacing(), declared in Mesh.h so test_mesh.cpp can call it
 * directly without a GPU) is plain host arithmetic with no CUDA-specific
 * calls, and is additionally exercised by that direct test.
 *
 * @warning generate()'s own GPU-upload plumbing compiles, but nvcc emits
 * warnings #20011-D/#20014-D on the XYZ<std::vector<Real>> values built
 * below: XYZ's constructor (math/XYZ.cuh) is unconditionally __host__
 * __device__, and the std::vector copy constructor it calls for this T
 * has no device-side counterpart. Believed harmless in practice -- these
 * particular XYZ<std::vector<Real>> values are host-side bookkeeping,
 * returned from this host function, and never passed into kernel/device
 * code -- but XYZ itself is shared far beyond this file, so this is
 * flagged for a look rather than changed here.
 */

#include <cmath>
#include "Mesh.h"

namespace {

template<typename Real>
GpuField<Real> uploadSpacing(const std::vector<Real>& host, GpuIndex gpuIndex, Handle& hand) {
    GpuField<Real> field = GpuField<Real>::allocate((int)host.size(), 1, 1, gpuIndex, hand);
    field.uploadFrom(host, hand);
    return field;
}

} // namespace

template<typename Real>
Mesh<Real> Mesh<Real>::generate(const SimulationConfig& cfg, GpuIndex gpuIndex, Handle& hand) {
    Real stretch = (Real)cfg.meshStretchingFactor;
    AxisSpacing<Real> x = computeAxisSpacing<Real>(cfg.Nx, (Real)cfg.aspectRatio, stretch);
    AxisSpacing<Real> y = computeAxisSpacing<Real>(cfg.Ny, (Real)cfg.widthRatio, stretch);
    AxisSpacing<Real> z = computeAxisSpacing<Real>(cfg.Nz, Real(1), stretch);

    GpuField<Real> deviceHx12 = uploadSpacing(x.hFull, gpuIndex, hand);
    GpuField<Real> deviceHy12 = uploadSpacing(y.hFull, gpuIndex, hand);
    GpuField<Real> deviceHz12 = uploadSpacing(z.hFull, gpuIndex, hand);
    GpuField<Real> deviceHpx = uploadSpacing(x.hHalf, gpuIndex, hand);
    GpuField<Real> deviceHpy = uploadSpacing(y.hHalf, gpuIndex, hand);
    GpuField<Real> deviceHpz = uploadSpacing(z.hHalf, gpuIndex, hand);

    return Mesh<Real>(
        XYZ<std::vector<Real>>(x.hFull, y.hFull, z.hFull),
        XYZ<std::vector<Real>>(x.hHalf, y.hHalf, z.hHalf),
        XYZ<std::vector<Real>>(x.nodeHalf, y.nodeHalf, z.nodeHalf),
        XYZ<std::vector<Real>>(x.node, y.node, z.node),
        std::move(deviceHx12), std::move(deviceHy12), std::move(deviceHz12),
        std::move(deviceHpx), std::move(deviceHpy), std::move(deviceHpz));
}

template<typename Real>
Spacing<Real> Mesh<Real>::deviceSpacing() const {
    return Spacing<Real>{
        deviceHx12_.pointer().get(), deviceHy12_.pointer().get(), deviceHz12_.pointer().get(),
        deviceHpx_.pointer().get(), deviceHpy_.pointer().get(), deviceHpz_.pointer().get()
    };
}

template class Mesh<double>;
template class Mesh<float>;
