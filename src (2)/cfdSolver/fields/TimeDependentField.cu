/**
 * @file TimeDependentField.cu
 * @ingroup cfd_solver
 * @brief Implements TimeDependentField::shiftTimeLevel().
 *
 * Compile-verified with nvcc (zero warnings); not run, since no physical
 * GPU is available in the environment this was checked in. shiftTimeLevel()
 * is two device-to-device cudaMemcpyAsync calls plus a synch -- no
 * device-independent remainder to extract and test the way Mesh's
 * computeAxisSpacing() or CheckpointIO's writeBuffers()/readBuffers() were.
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
