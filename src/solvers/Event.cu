/**
* @file Event.cu
 * @brief Implementation of the Event class for CUDA event management.
 */

#include "solvers/Event.h"

Event::Event() : event(nullptr) {
    cudaEvent_t tmp;
    CHECK_CUDA_ERROR (cudaEventCreateWithFlags(&tmp, cudaEventDisableTiming));

    event.reset(tmp);
}

void Event::record(const Handle& h) const {
    if (!event) throw std::runtime_error("Attempted to record on a null CUDA event");
    CHECK_CUDA_ERROR(cudaEventRecord(event.get(), h));
}

void Event::hold(const Handle& h) const {
    if (!event) throw std::runtime_error("Attempted to wait on a null CUDA event");

    CHECK_CUDA_ERROR(cudaStreamWaitEvent(h, event.get(), 0));
}

