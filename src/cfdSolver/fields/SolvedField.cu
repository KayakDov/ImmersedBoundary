/**
 * @file SolvedField.cu
 * @ingroup cfd_solver
 * @brief Implements SolvedField.
 *
 * @warning Not compile-verified: see GpuSolverSetup.cu's file-level
 * warning for why.
 */

#include "SolvedField.h"

template<typename Real>
SimpleArray<Real> SolvedField<Real>::asSimpleArray() const {
    return SimpleArray<Real>(buffer_.size(), buffer_.pointer());
}

template<typename Real>
void SolvedField<Real>::solve() {
    Handle& hand = handle();
    hand.ensureDevice();
    // Stencil writes, this snapshot, and the solve share a stream. Keep
    // buffer_ stable for callers while supplying GEMM with disjoint input.
    CHECK_CUDA_ERROR(cudaMemcpyAsync(rightHandSide_.pointer().get(), buffer_.pointer().get(),
        buffer_.size() * sizeof(Real), cudaMemcpyDeviceToDevice, hand));
    SimpleArray<Real> rightHandSide(rightHandSide_.size(), rightHandSide_.pointer());
    SimpleArray<Real> solution = asSimpleArray();
    solver_->solve(solution, rightHandSide, hand);
}

template<typename Real>
void SolvedField<Real>::synch() {
    handle().synch();
}

template class SolvedField<double>;
template class SolvedField<float>;
