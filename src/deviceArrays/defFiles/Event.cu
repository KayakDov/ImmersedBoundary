/**
* @file Event.cu
 * @brief Implementation of the Event class for CUDA event management.
 */

#include "solvers/Event.h"

Event::Event() : event(nullptr) {
    cudaEvent_t tmp;
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&tmp, cudaEventDisableTiming));
    event.reset(tmp);

    // Record whichever device this landed on so record() can detect a
    // mismatch later -- see the class-level comment in Event.h.
    int gpuIndexInt;
    CHECK_CUDA_ERROR(cudaGetDevice(&gpuIndexInt));
    gpuIndex_.index_ = gpuIndexInt;
}

void Event::record(const Handle& h) const {
    if (!event) throw std::runtime_error("Attempted to record on a null CUDA event");

    if (h.device() != gpuIndex_.index()) {
        // This event lives on a different device than the stream we're
        // being asked to record on. cudaEventRecord() requires them to
        // match, and there's no CUDA call that rebinds an existing
        // cudaEvent_t to a different device -- so the only option is to
        // destroy this one and create a fresh one on the new device.
        //
        // Cost warning: this is a real cudaEventDestroy + cudaEventCreate
        // pair, not just updating a field. Calling record() on the same
        // Event from alternating devices will pay this cost every time.

        // Make the OLD device current before destroying -- undocumented
        // either way whether cudaEventDestroy cares, so this costs nothing
        // extra in the common (non-mismatched) case and removes any doubt
        // in the mismatched case.
        gpuIndex_.switchDevice();
        event.reset();

        GpuIndex(h.device()).switchDevice();
        cudaEvent_t tmp;
        CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&tmp, cudaEventDisableTiming));
        event.reset(tmp);
        gpuIndex_.index_ = h.device();
    }

    CHECK_CUDA_ERROR(cudaEventRecord(event.get(), h));
}

void Event::hold(const Handle& h) const {
    if (!event) throw std::runtime_error("Attempted to wait on a null CUDA event");

    // No device handling needed here -- cudaStreamWaitEvent() is explicitly
    // safe across devices. See the class-level comment in Event.h.
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(h, event.get(), 0));
}
