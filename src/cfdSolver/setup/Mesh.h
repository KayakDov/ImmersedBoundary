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
#include "math/XYZ.cuh"

/**
 * @ingroup cfd_solver
 * @brief The grid-spacing arrays, generated once from a SimulationConfig
 * and never changed again.
 *
 * Direct port of MeshStretch.f90. For each axis, N+2 node positions are
 * generated on a uniform [0,1] grid, remapped through a hyperbolic
 * tangent to cluster them near the two walls, then scaled by that axis's
 * physical extent (aspectRatio for x, widthRatio for y, 1 for z). Cell
 * spacing (hx12/hy12/hz12) comes from consecutive node positions; control-
 * volume spacing (hpx/hpy/hpz) comes from consecutive cell-centre
 * (half-node) positions.
 *
 * The four host-side array groups are each stored as one
 * XYZ<std::vector<Real>> (.x/.y/.z = that axis's array); the public
 * accessors below are unchanged and still return one axis's array by
 * name, so no caller outside this class is affected. Device arrays stay
 * as six separate GpuField<Real> members, not XYZ-wrapped: see
 * deviceSpacing()'s Spacing<Real> construction, which already keeps
 * axis-labeled pointer names for the same cross-checking-against-Fortran
 * reason Operators.cuh's own Spacing<Real> formulas do.
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
    const std::vector<Real>& hostHx12() const { return hostH12_.x; }

    /// @return Host cell spacing on y, index 0..Ny.
    const std::vector<Real>& hostHy12() const { return hostH12_.y; }

    /// @return Host cell spacing on z, index 0..Nz.
    const std::vector<Real>& hostHz12() const { return hostH12_.z; }

    /// @return Host control-volume spacing on x, index 0..Nx+1.
    const std::vector<Real>& hostHpx() const { return hostHp_.x; }

    /// @return Host control-volume spacing on y, index 0..Ny+1.
    const std::vector<Real>& hostHpy() const { return hostHp_.y; }

    /// @return Host control-volume spacing on z, index 0..Nz+1.
    const std::vector<Real>& hostHpz() const { return hostHp_.z; }

    /// @return Host node-half (cell-centre) positions on x, index 0..Nx+2,
    /// needed for Teta's initialization, the History point-probe index
    /// lookup (see CfdSimulation.cu), and Average_flow's Psi_Yaverage.dat/
    /// Tmpr_Yaverage.dat "R" column (see setup/AverageFlow.cu).
    const std::vector<Real>& hostNodeHalfX() const { return hostNodeHalf_.x; }

    /// @return Host node-half (cell-centre) positions on y, index 0..Ny+2 --
    /// Fortran's Y12. Needed by CfdSimulation's time-averaging output
    /// (Tmpr_av.dat/Vy_av.dat/Amplitudes.dat's Y column -- see
    /// MeshStretch.f90's Y12 and Solution_time_Q2D.f90's Point_Write
    /// calls). computeAxisSpacing() already computed this for every axis;
    /// only x's and z's were kept until this field was added.
    const std::vector<Real>& hostNodeHalfY() const { return hostNodeHalf_.y; }

    /// @return Host node-half (cell-centre) positions on z, index 0..Nz+2 --
    /// the z-axis counterpart of hostNodeHalfX(), needed only for
    /// Average_flow's two output files' "Z" column (see
    /// setup/AverageFlow.cu). computeAxisSpacing() already computed this
    /// for every axis; generate() previously kept only x's.
    const std::vector<Real>& hostNodeHalfZ() const { return hostNodeHalf_.z; }

    /// @return Host plain (cell-corner) node positions on x, index 0..Nx+1 --
    /// Fortran's plain X array (MeshStretch.f90), as opposed to
    /// hostNodeHalfX()'s cell-centre X12. Needed by CfdSimulation's
    /// time-averaging output for Vx_av.dat's X column (Vx lives on the
    /// x-face, not the x-centre -- see Solution_time_Q2D.f90's
    /// `Point_Write(..., VMxNew, X, Y12, Z12, ...)`). computeAxisSpacing()
    /// already computed this (as its local `node`) for every axis; it was
    /// discarded after use until this field was added.
    const std::vector<Real>& hostX() const { return hostNode_.x; }

    /// @return Host plain (cell-corner) node positions on y, index 0..Ny+1 --
    /// Fortran's plain Y array. Needed for Vy_av.dat's Y column.
    const std::vector<Real>& hostY() const { return hostNode_.y; }

    /// @return Host plain (cell-corner) node positions on z, index 0..Nz+1 --
    /// Fortran's plain Z array. Needed for Vz_av.dat's Z column.
    const std::vector<Real>& hostZ() const { return hostNode_.z; }

private:
    Mesh(XYZ<std::vector<Real>> hostH12, XYZ<std::vector<Real>> hostHp,
         XYZ<std::vector<Real>> hostNodeHalf, XYZ<std::vector<Real>> hostNode,
         GpuField<Real> deviceHx12, GpuField<Real> deviceHy12, GpuField<Real> deviceHz12,
         GpuField<Real> deviceHpx, GpuField<Real> deviceHpy, GpuField<Real> deviceHpz)
        : hostH12_(std::move(hostH12)), hostHp_(std::move(hostHp)),
          hostNodeHalf_(std::move(hostNodeHalf)), hostNode_(std::move(hostNode)),
          deviceHx12_(std::move(deviceHx12)), deviceHy12_(std::move(deviceHy12)), deviceHz12_(std::move(deviceHz12)),
          deviceHpx_(std::move(deviceHpx)), deviceHpy_(std::move(deviceHpy)), deviceHpz_(std::move(deviceHpz)) {}

    XYZ<std::vector<Real>> hostH12_;
    XYZ<std::vector<Real>> hostHp_;
    XYZ<std::vector<Real>> hostNodeHalf_;
    XYZ<std::vector<Real>> hostNode_;
    GpuField<Real> deviceHx12_, deviceHy12_, deviceHz12_;
    GpuField<Real> deviceHpx_, deviceHpy_, deviceHpz_;
};

#endif //CFDSOLVER_MESH_H
