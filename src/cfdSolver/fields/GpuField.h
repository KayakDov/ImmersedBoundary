/**
 * @file GpuField.h
 * @ingroup cfd_solver
 * @brief GpuField: a single device-resident array with host upload/download
 * at its boundary only.
 */

#ifndef CFDSOLVER_GPUFIELD_H
#define CFDSOLVER_GPUFIELD_H

#include <vector>
#include <stdexcept>
#include "BoundaryConditions.cuh"
#include "deviceArrays/headers/Support/GpuPointer.h"
#include "deviceArrays/headers/handle.h"

/**
 * @ingroup cfd_solver
 * @brief A device-resident 3D field with no attached solver.
 *
 * Used for fields that are never themselves the subject of an
 * eigendecomposition solve: Teta (static, set once), Prs/Dprs (the
 * pressure correction's own RHS gradient target), and the shared
 * right-hand-side scratch buffers the stencil operators accumulate into.
 *
 * Every GpuField owns exactly one GpuPointer allocation. Host contact
 * happens only in uploadFrom() and downloadTo() -- never inside a timestep.
 *
 * @tparam Real Floating-point type.
 */
template<typename Real>
class GpuField {
public:
    /**
     * @brief Allocates a new, zero-initialized field.
     * @param nx Real-data extent on the x axis.
     * @param ny Real-data extent on the y axis.
     * @param nz Real-data extent on the z axis.
     * @param gpuIndex Which device the field lives on.
     * @param hand A handle on gpuIndex, used for the zero-fill.
     * @return The allocated field.
     */
    static GpuField allocate(int nx, int ny, int nz, GpuIndex gpuIndex, Handle& hand);

    /**
     * @brief Constructs a view over this field for kernel use.
     * @return A FieldView over this field's device memory.
     */
    FieldView<Real> view() const { return FieldView<Real>::fromGpuPointer(buffer_, nx_, ny_, nz_); }

    /**
     * @brief Copies host data onto this field's device memory.
     * @param hostData Row-major host data, i fastest, matching FieldView's
     * own indexing; size must equal size().
     * @param hand Handle to perform the transfer on.
     */
    void uploadFrom(const std::vector<Real>& hostData, Handle& hand);

    /**
     * @brief Copies this field's device memory to host.
     * @param hand Handle to perform the transfer on.
     * @return Host data in the same layout uploadFrom() expects.
     */
    std::vector<Real> downloadTo(Handle& hand) const;

    /**
     * @brief Sets every element to zero, asynchronously. Matches the
     * Fortran pattern of zeroing a scratch buffer (`FDRHP = 0.D0`) before
     * a kernel that accumulates into it with +=.
     * @param hand Handle to perform the fill on.
     */
    void zero(Handle& hand);

    /// @return Total element count.
    size_t size() const { return (size_t)nx_ * (size_t)ny_ * (size_t)nz_; }

    /**
     * @brief Downloads a size-1 field's single element. Convenience for
     * reading back a reduction result.
     * @param hand Handle to perform the transfer on.
     * @return The one element's value.
     */
    Real downloadScalar(Handle& hand) const { return downloadTo(hand)[0]; }

    /// @return Real-data extent on the x axis.
    int nx() const { return nx_; }

    /// @return Real-data extent on the y axis.
    int ny() const { return ny_; }

    /// @return Real-data extent on the z axis.
    int nz() const { return nz_; }

    /// @return The device this field lives on.
    GpuIndex gpuIndex() const { return buffer_.gpuIndex(); }

    /**
     * @brief The underlying pointer, for classes (SolvedField) that need
     * to view the same memory as a different type (SimpleArray) for the
     * eigendecomposition solver.
     * @return This field's owning GpuPointer.
     */
    const GpuPointer<Real>& pointer() const { return buffer_; }

private:
    GpuField(GpuPointer<Real> buffer, int nx, int ny, int nz) : buffer_(buffer), nx_(nx), ny_(ny), nz_(nz) {}

    void checkSize(size_t providedSize) const;

    GpuPointer<Real> buffer_;
    int nx_, ny_, nz_;
};

#endif //CFDSOLVER_GPUFIELD_H
