/**
 * @file Mesh.h
 * @ingroup cfd_solver
 * @brief Mesh: tanh-stretched grid generation.
 */

#ifndef CFDSOLVER_MESH_H
#define CFDSOLVER_MESH_H

#include <vector>
#include "../fields/GpuField.h"
#include "../physics/Operators.cuh"
#include "../SimulationConfig.h"

/**
 * @ingroup cfd_solver
 * @brief The six grid-spacing arrays (two per axis), generated once from a
 * SimulationConfig and never changed again.
 *
 * Direct port of MeshStretch.f90. For each axis, N+2 node positions are
 * generated on a uniform [0,1] grid, remapped through a hyperbolic
 * tangent to cluster them near the two walls, then scaled by that axis's
 * physical extent (aspectRatio for x, widthRatio for y, 1 for z). Cell
 * spacing (hx12/hy12/hz12) comes from consecutive node positions; control-
 * volume spacing (hpx/hpy/hpz) comes from consecutive cell-centre
 * (half-node) positions.
 *
 * Host arrays are retained after upload (needed once more, at startup, to
 * construct each field's eigendecomposition solver); device arrays are
 * what every timestep's kernels read. Host memory is never touched again
 * once construction finishes.
 *
 * @tparam Real Floating-point type.
 */
template<typename Real>
class Mesh {
public:
    /**
     * @brief Generates and uploads the mesh for one simulation.
     * @param cfg The simulation configuration (grid extents, aspect/width
     * ratio, stretching factor).
     * @param gpuIndex Which device to upload the spacing arrays to.
     * @param hand Handle to perform the upload on.
     * @return The generated mesh.
     */
    static Mesh generate(const SimulationConfig& cfg, GpuIndex gpuIndex, Handle& hand);

    /// @return Device-resident spacing, for kernel launches.
    Spacing<Real> deviceSpacing() const;

    /// @return Host cell spacing on x, index 0..Nx.
    const std::vector<Real>& hostHx12() const { return hostHx12_; }

    /// @return Host cell spacing on y, index 0..Ny.
    const std::vector<Real>& hostHy12() const { return hostHy12_; }

    /// @return Host cell spacing on z, index 0..Nz.
    const std::vector<Real>& hostHz12() const { return hostHz12_; }

    /// @return Host control-volume spacing on x, index 0..Nx+1.
    const std::vector<Real>& hostHpx() const { return hostHpx_; }

    /// @return Host control-volume spacing on y, index 0..Ny+1.
    const std::vector<Real>& hostHpy() const { return hostHpy_; }

    /// @return Host control-volume spacing on z, index 0..Nz+1.
    const std::vector<Real>& hostHpz() const { return hostHpz_; }

    /// @return Host node-half (cell-centre) positions on x, index 0..Nx+2,
    /// needed only for Teta's initialization and the History point-probe
    /// index lookup -- see CfdSimulation.cu.
    const std::vector<Real>& hostNodeHalfX() const { return hostNodeHalfX_; }

private:
    Mesh(std::vector<Real> hostHx12, std::vector<Real> hostHy12, std::vector<Real> hostHz12,
         std::vector<Real> hostHpx, std::vector<Real> hostHpy, std::vector<Real> hostHpz,
         std::vector<Real> hostNodeHalfX,
         GpuField<Real> deviceHx12, GpuField<Real> deviceHy12, GpuField<Real> deviceHz12,
         GpuField<Real> deviceHpx, GpuField<Real> deviceHpy, GpuField<Real> deviceHpz)
        : hostHx12_(std::move(hostHx12)), hostHy12_(std::move(hostHy12)), hostHz12_(std::move(hostHz12)),
          hostHpx_(std::move(hostHpx)), hostHpy_(std::move(hostHpy)), hostHpz_(std::move(hostHpz)),
          hostNodeHalfX_(std::move(hostNodeHalfX)),
          deviceHx12_(std::move(deviceHx12)), deviceHy12_(std::move(deviceHy12)), deviceHz12_(std::move(deviceHz12)),
          deviceHpx_(std::move(deviceHpx)), deviceHpy_(std::move(deviceHpy)), deviceHpz_(std::move(deviceHpz)) {}

    std::vector<Real> hostHx12_, hostHy12_, hostHz12_;
    std::vector<Real> hostHpx_, hostHpy_, hostHpz_;
    std::vector<Real> hostNodeHalfX_;
    GpuField<Real> deviceHx12_, deviceHy12_, deviceHz12_;
    GpuField<Real> deviceHpx_, deviceHpy_, deviceHpz_;
};

#endif //CFDSOLVER_MESH_H
