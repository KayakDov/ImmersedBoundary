/**
 * @file Mesh.cu
 * @ingroup cfd_solver
 * @brief Implements Mesh::generate().
 *
 * @warning Not compile-verified: see GpuSolverSetup.cu's file-level
 * warning for why. The per-axis math itself (computeAxisSpacing) is plain
 * host arithmetic with no CUDA-specific calls, and was checked by hand
 * against MeshStretch.f90 line by line.
 */

#include <cmath>
#include "Mesh.h"

namespace {

/**
 * @brief One axis's cell spacing (hFull) and control-volume spacing
 * (hHalf), computed by the same algorithm for every axis.
 * @tparam Real Floating-point type.
 */
template<typename Real>
struct AxisSpacing {
    std::vector<Real> hFull;     ///< Size n+1, index 0..n.
    std::vector<Real> hHalf;     ///< Size n+2, index 0..n+1.
    std::vector<Real> nodeHalf;  ///< Size n+3, index 0..n+2 -- the node-half (cell-centre) positions hHalf was differenced from.
};

/**
 * @brief Generates one axis's tanh-stretched node positions, cell
 * spacing, and control-volume spacing. Direct port of MeshStretch.f90's
 * per-axis block (identical for X, Y, and Z up to the scale factor).
 * @tparam Real Floating-point type.
 * @param n The axis's Nx/Ny/Nz (the solved-field interior count, not n1).
 * @param scale Physical extent to scale the [0,1] stretched grid to
 * (aspectRatio for x, widthRatio for y, 1 for z).
 * @param stretchFactor MeshStretch.f90's `s`.
 * @return This axis's spacing arrays.
 */
template<typename Real>
AxisSpacing<Real> computeAxisSpacing(int n, Real scale, Real stretchFactor) {
    int n1 = n + 1;
    int n2 = n1 + 1;
    std::vector<Real> node(n1 + 1);
    std::vector<Real> nodeHalf(n2 + 1);

    Real h = Real(1) / Real(n1);
    node[0] = Real(0);
    for (int i = 1; i <= n1; ++i) node[i] = node[i - 1] + h;
    Real tanhHalfStretch = std::tanh(Real(0.5) * stretchFactor);
    for (int i = 0; i <= n1; ++i)
        node[i] = scale * (Real(0.5) + Real(0.5) * std::tanh(stretchFactor * (node[i] - Real(0.5))) / tanhHalfStretch);

    nodeHalf[0] = Real(0);
    nodeHalf[n2] = scale;
    for (int i = 1; i <= n1; ++i) nodeHalf[i] = Real(0.5) * (node[i] + node[i - 1]);

    AxisSpacing<Real> result;
    result.hFull.resize(n + 1);
    for (int i = 0; i <= n; ++i) result.hFull[i] = node[i + 1] - node[i];

    result.hHalf.resize(n1 + 1);
    for (int i = 0; i <= n1; ++i) result.hHalf[i] = nodeHalf[i + 1] - nodeHalf[i];

    result.nodeHalf = std::move(nodeHalf);
    return result;
}

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
        x.hFull, y.hFull, z.hFull, x.hHalf, y.hHalf, z.hHalf, x.nodeHalf,
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
