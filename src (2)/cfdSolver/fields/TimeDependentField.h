/**
 * @file TimeDependentField.h
 * @ingroup cfd_solver
 * @brief TimeDependentField: the Old/Current/New triad shared by
 * Temperature and each velocity component.
 */

#ifndef CFDSOLVER_TIMEDEPENDENTFIELD_H
#define CFDSOLVER_TIMEDEPENDENTFIELD_H

#include "SolvedField.h"

/**
 * @ingroup cfd_solver
 * @brief A field with three simultaneously-live time levels: the value one
 * step back (old), the value at the start of the current step (current),
 * and the value the current step's solve produces (new, held directly in
 * a SolvedField's own buffer).
 *
 * Port of the VMxOld/VMx/VMxNew (and TmpOld/Tmpr/TmpNew) triads in
 * modfv_3D_Q2D.f90's Variables module, and of the fused Old<-Current,
 * Current<-New copy loop at the end of time_step_Q2D.f90.
 *
 * Deliberately uses explicit device-to-device copies in shiftTimeLevel()
 * rather than a pointer-rotation scheme: a rotation would avoid the
 * copies entirely, but adds bookkeeping (which buffer currently plays
 * which role) that is easy to get subtly wrong and hard to verify without
 * a CUDA toolchain to compile and run against. A device-to-device memcpy
 * never crosses the PCIe bus either way, so the simpler design was chosen
 * deliberately, not by oversight -- see this project's own record of
 * exactly this kind of indexing mistake being worth avoiding when it
 * cannot be compiled and checked.
 *
 * @tparam Real Floating-point type.
 */
template<typename Real>
class TimeDependentField {
public:
    /**
     * @brief Constructs a TimeDependentField.
     * @param old The value one timestep back.
     * @param current The value at the start of the current timestep.
     * @param solved Owns the current timestep's solve, whose buffer holds
     * the new value once solved and synched.
     */
    TimeDependentField(GpuField<Real> old, GpuField<Real> current, SolvedField<Real> solved)
        : old_(std::move(old)), current_(std::move(current)), solved_(std::move(solved)) {}

    /// @return The value one timestep back.
    FieldView<Real> oldView() const { return old_.view(); }

    /// @return The value at the start of the current timestep.
    FieldView<Real> currentView() const { return current_.view(); }

    /// @return The solver for this field's current-timestep right-hand
    /// side; its view() is the right-hand-side build target before
    /// solve(), and the new value after solve() and synch().
    SolvedField<Real>& solved() { return solved_; }

    /**
     * @brief Advances this field by one timestep: old becomes the value
     * current held, and current becomes the value the solve just produced.
     *
     * Must be called only after solved().solve() and solved().synch()
     * have both completed for this timestep.
     * @param hand Handle to perform the copies on.
     */
    void shiftTimeLevel(Handle& hand);

    /**
     * @brief Downloads the old-timestep value to host, for checkpointing.
     * @param hand Handle to perform the transfer on.
     * @return Host data in FieldView-compatible layout.
     */
    std::vector<Real> downloadOld(Handle& hand) const { return old_.downloadTo(hand); }

    /**
     * @brief Downloads the current-timestep value to host, for
     * checkpointing.
     * @param hand Handle to perform the transfer on.
     * @return Host data in FieldView-compatible layout.
     */
    std::vector<Real> downloadCurrent(Handle& hand) const { return current_.downloadTo(hand); }

    /**
     * @brief Uploads a checkpointed old-timestep value, overwriting the
     * current one.
     * @param hostData Host data in FieldView-compatible layout.
     * @param hand Handle to perform the transfer on.
     */
    void uploadOld(const std::vector<Real>& hostData, Handle& hand) { old_.uploadFrom(hostData, hand); }

    /**
     * @brief Uploads a checkpointed current-timestep value, overwriting
     * the current one.
     * @param hostData Host data in FieldView-compatible layout.
     * @param hand Handle to perform the transfer on.
     */
    void uploadCurrent(const std::vector<Real>& hostData, Handle& hand) { current_.uploadFrom(hostData, hand); }

private:
    GpuField<Real> old_;
    GpuField<Real> current_;
    SolvedField<Real> solved_;
};

#endif //CFDSOLVER_TIMEDEPENDENTFIELD_H
