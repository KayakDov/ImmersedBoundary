/**
 * @file SolvedField.h
 * @ingroup cfd_solver
 * @brief SolvedField: a field whose right-hand side is solved by
 * eigendecomposition, entirely on the GPU.
 */

#ifndef CFDSOLVER_SOLVEDFIELD_H
#define CFDSOLVER_SOLVEDFIELD_H

#include <memory>
#include "GpuField.h"
#include "solvers/EigenDecomp/EigenDecompSolver.h"
#include "deviceArrays/headers/SimpleArray.h"

/**
 * @ingroup cfd_solver
 * @brief One of the six eigendecomposition-solved fields (Temperature, Vx,
 * Vy, Vz, Pressure, Potential).
 *
 * Deliberately does not use EigenDecompForFortran. That class exists to
 * bridge a CPU-resident Fortran caller to the GPU, and both its solve()
 * and synch() move data across the PCIe bus on every call -- the right
 * design for a CPU caller, the wrong one here, where the stencil kernels
 * in FiniteDifferenceOperators already write their right-hand side
 * directly into device memory.
 *
 * SolvedField instead owns one GpuField and views it two ways: as a
 * FieldView for the stencil kernels (rhsView(), read before solve() and
 * written again after synch() for the next timestep), and as a
 * SimpleArray for EigenDecompSolver::solve() (asSimpleArray()) -- the same
 * bytes, no copy between the two.
 *
 * Owns its Handle through a std::unique_ptr rather than by value. Handle
 * is deliberately neither copyable nor movable (see handle.h's class
 * comment) -- every caller in the codebase takes a `Handle&`/`Handle*`,
 * and a SolvedField is no exception to that pattern even though it is
 * itself a long-lived owner: the Handle is heap-allocated once, in
 * GpuSolverBuilder::build()'s buildFieldSolver() (via
 * std::make_unique<Handle>(...)), and only the unique_ptr -- never the
 * Handle itself -- is ever moved.
 *
 * @tparam Real Floating-point type.
 */
template<typename Real>
class SolvedField {
public:
    /**
     * @brief Constructs a SolvedField over an existing buffer, solver, and
     * heap-owned handle.
     * @param buffer The field's data, shared between kernel access and the
     * solver.
     * @param solver The eigendecomposition solver for this field's
     * Laplacian/Helmholtz operator.
     * @param hand The handle this field's solve runs on, already
     * heap-allocated by the caller (e.g. via std::make_unique<Handle>(...))
     * so that only the owning pointer, never the non-movable Handle
     * itself, needs to move.
     */
    SolvedField(GpuField<Real> buffer, std::unique_ptr<EigenDecompSolver<Real>> solver, std::unique_ptr<Handle> hand)
        : buffer_(std::move(buffer)), solver_(std::move(solver)), hand_(std::move(hand)) {}

    /**
     * @brief The view the stencil kernels read and write -- the field's
     * right-hand side before solve(), and its solution after synch().
     * @return A FieldView over this field's single buffer.
     */
    FieldView<Real> view() const { return buffer_.view(); }

    /**
     * @brief Launches the solve asynchronously on this field's handle.
     *
     * The right-hand side must already be written into view() before
     * calling this. Returns immediately; call synch() before reading the
     * solution back out of view().
     */
    void solve();

    /**
     * @brief Blocks until this field's launched solve has completed.
     */
    void synch();

    /// @return This field's handle, for building cross-field event
    /// dependencies in the surrounding timestep.
    Handle& handle() { return *hand_; }

    /// @return This field's real-data extents and device, via its buffer.
    const GpuField<Real>& buffer() const { return buffer_; }

    /**
     * @brief Zeros this field's buffer, for a right-hand-side build that
     * accumulates with += (matching Fortran's `FDRHP = 0.D0` before such
     * a kernel).
     * @param hand Handle to perform the fill on.
     */
    void zero(Handle& hand) { buffer_.zero(hand); }

private:
    SimpleArray<Real> asSimpleArray() const;

    GpuField<Real> buffer_;
    std::unique_ptr<EigenDecompSolver<Real>> solver_;
    std::unique_ptr<Handle> hand_;
};

#endif //CFDSOLVER_SOLVEDFIELD_H
