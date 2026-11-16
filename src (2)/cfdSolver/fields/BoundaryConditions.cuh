/**
 * @file BoundaryConditions.cuh
 * @ingroup cfd_solver
 * @brief Boundary-conditional field access, replacing Fortran's ghost-cell
 * padding.
 *
 * Every field in the original Fortran solver (Tmpr, VMx, VMy, VMz) carried
 * one layer of ghost cells on every axis, refreshed once per timestep by
 * EVDbounds (bounds_3D.f90). Those ghost cells never held independent data:
 * they always held either a literal 0 (Dirichlet) or a copy of the nearest
 * real cell (Neumann mirror). BoundaryConditions::readTemperature() and
 * BoundaryConditions::readVelocity() replace the ghost layer entirely:
 * fields are allocated at their true interior size only, and any stencil
 * that would have read a ghost cell calls one of these instead.
 *
 * This substitution is not assumed correct; it is a direct, mechanical
 * transcription of the equivalent functions in
 * verify_padding_equivalence.cpp, which proved -- 205,071 cell-by-cell
 * comparisons, 0 mismatches, every live EVD_BCx/y/z combination, 27
 * grid-size combinations -- that it produces bit-identical results to
 * Fortran's ghost-fill approach.
 */

#ifndef CFDSOLVER_BOUNDARYCONDITIONS_CUH
#define CFDSOLVER_BOUNDARYCONDITIONS_CUH

#include <cstddef>
#include "math/XYZ.cuh"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#elif !defined(__host__)
/* math/XYZ.cuh above already pulls in <cuda_runtime.h> unconditionally,
 * which on some CUDA installations defines these even outside
 * __CUDACC__ -- guard on __host__ already being defined so this
 * fallback doesn't redefine them and warn. */
#define __host__
#define __device__
#define __forceinline__ inline
#define __global__
#endif

/**
 * @ingroup cfd_solver
 * @brief A Dirichlet (fixed value) or Neumann (fixed derivative) boundary
 * condition for one axis of one field.
 */
enum class BoundaryKind { Dirichlet, Neumann };

/**
 * @ingroup cfd_solver
 * @brief Per-axis boundary conditions for a field whose boundary treatment
 * is configurable (Temperature, Potential).
 *
 * An XYZ<BoundaryKind> rather than its own hand-rolled struct: every call
 * site already only ever constructs this via a 3-argument {x,y,z}-style
 * initializer or copies one -- both of which XYZ<T>'s own 3-argument
 * constructor and implicit copy constructor already support identically
 * -- so the alias is a drop-in replacement. readTemperature() below also
 * uses XYZ<T>::operator[] to index this by axis number.
 */
using AxisBoundaries = XYZ<BoundaryKind>;

/**
 * @ingroup cfd_solver
 * @brief An unpadded, 1-based view over one field's device-resident real
 * interior data.
 *
 * Real-data extents differ per field, derived directly from
 * bounds_3D.f90's EVDbounds and every stencil's own loop bounds:
 *   - Temperature (Tmpr/TmpNew/TmpOld/Teta): (1:Nx1,1:Ny1,1:Nz1).
 *   - VMx: (1:Nx,1:Ny1,1:Nz1). VMy: (1:Nx1,1:Ny,1:Nz1). VMz: (1:Nx1,1:Ny1,1:Nz).
 *
 * A FieldView never owns memory. It is always a view over a
 * GpuPointer-backed allocation owned by a GpuField, constructed via
 * fromGpuPointer() -- the same bytes are simultaneously viewable as a
 * SimpleArray for handing to EigenDecompSolver::solve(), so a field's RHS
 * build, solve, and solution read are three operations on one buffer, with
 * no copy between them.
 *
 * @tparam Real Floating-point type (float or double).
 */
template<typename Real>
class FieldView {
public:
    /**
     * @brief Constructs a view over existing device memory.
     * @param data Pointer to the first real-interior element.
     * @param nx Real-data extent on the x axis.
     * @param ny Real-data extent on the y axis.
     * @param nz Real-data extent on the z axis.
     */
    __host__ __device__ __forceinline__
    FieldView(Real* data, int nx, int ny, int nz) : data_(data), nx_(nx), ny_(ny), nz_(nz) {}

    /**
     * @brief Constructs a view over a pointer-wrapper's memory with no
     * copy.
     *
     * Templated on the pointer-wrapper type, rather than naming
     * GpuPointer<Real> directly, so this header stays host-compilable
     * without pulling in GpuPointer.h's own CUDA-toolkit dependency chain.
     * Under nvcc this is always called with a real GpuPointer<Real>&.
     *
     * @tparam PointerLike Any type exposing `Real* get() const`.
     * @param pointer The owning pointer-wrapper.
     * @param nx Real-data extent on the x axis.
     * @param ny Real-data extent on the y axis.
     * @param nz Real-data extent on the z axis.
     * @return A FieldView over the same memory.
     */
    template<typename PointerLike>
    __host__ static FieldView fromGpuPointer(const PointerLike& pointer, int nx, int ny, int nz) {
        return FieldView(pointer.get(), nx, ny, nz);
    }

    /// @return The real-data extent on the x axis.
    __host__ __device__ __forceinline__ int nx() const { return nx_; }

    /// @return The real-data extent on the y axis.
    __host__ __device__ __forceinline__ int ny() const { return ny_; }

    /// @return The real-data extent on the z axis.
    __host__ __device__ __forceinline__ int nz() const { return nz_; }

    /// @return Total element count (nx * ny * nz).
    __host__ __device__ __forceinline__ size_t size() const {
        return (size_t)nx_ * (size_t)ny_ * (size_t)nz_;
    }

    /**
     * @brief Mutable access to one real-interior cell.
     * @param i 1-based x index, must be in [1, nx()].
     * @param j 1-based y index, must be in [1, ny()].
     * @param k 1-based z index, must be in [1, nz()].
     * @return Reference to that cell.
     */
    __host__ __device__ __forceinline__ Real& at(int i, int j, int k) {
        return data_[index(i, j, k)];
    }

    /**
     * @brief Read-only access to one real-interior cell.
     * @param i 1-based x index, must be in [1, nx()].
     * @param j 1-based y index, must be in [1, ny()].
     * @param k 1-based z index, must be in [1, nz()].
     * @return That cell's value.
     */
    __host__ __device__ __forceinline__ Real at(int i, int j, int k) const {
        return data_[index(i, j, k)];
    }

    /// @return The raw underlying pointer.
    __host__ __device__ __forceinline__ Real* data() const { return data_; }

private:
    __host__ __device__ __forceinline__ size_t index(int i, int j, int k) const {
        return (size_t)(i - 1) + (size_t)nx_ * ((size_t)(j - 1) + (size_t)ny_ * (size_t)(k - 1));
    }

    Real* data_;
    int nx_, ny_, nz_;
};

/**
 * @ingroup cfd_solver
 * @brief Stateless boundary-condition resolution for Temperature-like and
 * velocity-like fields.
 *
 * Every member is a static, host+device-callable function so it can be
 * called identically from a host unit test (g++) or from a CUDA kernel
 * (nvcc) -- the code exercised by the test harness is exactly the code
 * that runs on the GPU.
 */
class BoundaryConditions {
public:
    /**
     * @brief Reads a Temperature-like field (Tmpr/TmpNew/TmpOld/Teta) at
     * any index, resolving a Dirichlet-zero or Neumann-mirror boundary
     * value when the index falls outside the field's real range.
     *
     * Equivalent to reading a ghost cell EVDbounds would have filled; see
     * this file's header comment for the proof of that equivalence.
     *
     * @tparam Real Floating-point type.
     * @param field The field, unpadded.
     * @param i 1-based x index; may be 0 or nx()+1.
     * @param j 1-based y index; may be 0 or ny()+1.
     * @param k 1-based z index; may be 0 or nz()+1.
     * @param bc This field's per-axis boundary condition.
     * @return The resolved value.
     *
     * @note Not __forceinline__, unlike every other function in this file:
     * this function and boundaryValue() call each other (a true corner
     * re-clamps one axis per recursive step, up to 3 levels, always
     * terminating), and nvcc's host pass maps __forceinline__ to GCC's
     * always_inline, which GCC refuses on any cycle in the static call
     * graph regardless of actual depth -- "inlining failed ... function
     * not inlinable". Plain (optional) inlining has no such restriction
     * and changes no return value, only whether a boundary-cell read costs
     * one real call instead of being flattened -- interior cells (the vast
     * majority of any grid) never reach this function at all.
     */
    template<typename Real>
    __host__ __device__
    static Real readTemperature(const FieldView<Real>& field, int i, int j, int k, AxisBoundaries bc) {
        int idx[3] = {i, j, k};
        int extent[3] = {field.nx(), field.ny(), field.nz()};
        for (int axis = 0; axis < 3; ++axis) {
            if (idx[axis] < 1 || idx[axis] > extent[axis]) {
                int mirror[3] = {i, j, k};
                mirror[axis] = (idx[axis] < 1) ? 1 : extent[axis];
                return boundaryValue(bc[(size_t)axis], field, mirror[0], mirror[1], mirror[2], bc);
            }
        }
        return field.at(i, j, k);
    }

    /**
     * @brief Reads a velocity component (VMx/VMy/VMz, current or *New) at
     * any index, returning 0 for any index outside the field's own real
     * range.
     *
     * Velocity is always Dirichlet-zero (no-slip), on every axis,
     * unconditionally -- confirmed from bounds_3D.f90's VMxNew/VMyNew/
     * VMzNew blocks, which have no boundary-kind branch at all, unlike
     * Temperature's.
     *
     * @tparam Real Floating-point type.
     * @param field The velocity component, unpadded.
     * @param i 1-based x index.
     * @param j 1-based y index.
     * @param k 1-based z index.
     * @return The resolved value.
     */
    template<typename Real>
    __host__ __device__ __forceinline__
    static Real readVelocity(const FieldView<Real>& field, int i, int j, int k) {
        if (i < 1 || i > field.nx() || j < 1 || j > field.ny() || k < 1 || k > field.nz())
            return Real(0);
        return field.at(i, j, k);
    }

private:
    /**
     * @brief Resolves one axis's boundary value: 0 for Dirichlet, or the
     * nearest real cell for Neumann.
     * @tparam Real Floating-point type.
     * @param kind The boundary kind for the axis that was out of range.
     * @param field The field being read.
     * @param mirrorI 1-based x index of the nearest real cell.
     * @param mirrorJ 1-based y index of the nearest real cell.
     * @param mirrorK 1-based z index of the nearest real cell.
     * @param bc This field's full per-axis boundary condition, needed in
     * case the mirrored read is itself at a corner.
     * @return The resolved boundary value.
     *
     * @note Not __forceinline__ -- see readTemperature()'s @note; the two
     * are mutually recursive, which is incompatible with nvcc's host-side
     * always_inline regardless of this recursion's bounded depth.
     */
    template<typename Real>
    __host__ __device__
    static Real boundaryValue(BoundaryKind kind, const FieldView<Real>& field,
                               int mirrorI, int mirrorJ, int mirrorK, AxisBoundaries bc) {
        if (kind == BoundaryKind::Dirichlet) return Real(0);
        return readTemperature(field, mirrorI, mirrorJ, mirrorK, bc);
    }
};

#endif //CFDSOLVER_BOUNDARYCONDITIONS_CUH
