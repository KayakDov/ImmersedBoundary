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
    SimpleArray<Real> rightHandSide = asSimpleArray();
    SimpleArray<Real> solution = asSimpleArray();
    solver_->solve(solution, rightHandSide, *hand_);
}

template<typename Real>
void SolvedField<Real>::synch() {
    hand_->synch();
}

template class SolvedField<double>;
template class SolvedField<float>;
