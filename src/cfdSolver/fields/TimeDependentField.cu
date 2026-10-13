/**
 * @file TimeDependentField.cu
 * @ingroup cfd_solver
 * @brief Implements TimeDependentField::shiftTimeLevel().
 *
 * @warning Not compile-verified: see GpuSolverSetup.cu's file-level
 * warning for why.
 */

#include "TimeDependentField.h"

template<typename Real>
void TimeDependentField<Real>::shiftTimeLevel(Handle& hand) {
    hand.ensureDevice();
    size_t bytes = old_.size() * sizeof(Real);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(old_.pointer().get(), current_.pointer().get(), bytes, cudaMemcpyDeviceToDevice, hand));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(current_.pointer().get(), solved_.buffer().pointer().get(), bytes, cudaMemcpyDeviceToDevice, hand));
    hand.synch();
}

template class TimeDependentField<double>;
template class TimeDependentField<float>;
