/**
 * @file Event.cu
 * @brief Implementation of the Event class for CUDA event management.
 */

#include "Event.h"

Event::Event() : event(nullptr), valid(false) {
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA event");
    }
    valid = true;
}

Event::~Event() {
    if (valid) {
        cudaEventDestroy(event);
        valid = false;
    }
}

void Event::renew() {
    if (valid) {
        cudaEventDestroy(event);
    }
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
        throw std::runtime_error("Failed to recreate CUDA event");
    }
    valid = true;
}

void Event::record(const Handle& h) {
    if (!valid) throw std::runtime_error("Attempted to record on invalid event");
    if (cudaEventRecord(event, h) != cudaSuccess) {
        throw std::runtime_error("Failed to record CUDA event");
    }
}

void Event::wait(const Handle& h) const {
    if (!valid) throw std::runtime_error("Attempted to wait on invalid event");
    if (cudaStreamWaitEvent(h, event, 0) != cudaSuccess) {
        throw std::runtime_error("Failed to make stream wait on CUDA event");
    }
}

void Event::renewAndRecord(const Handle& h) {
    renew();
    record(h);
}

Event::Event(Event&& other) noexcept : event(other.event), valid(other.valid) {
    other.event = nullptr;
    other.valid = false;
}

Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) {
        if (valid) {
            cudaEventDestroy(event);
        }
        event = other.event;
        valid = other.valid;
        other.event = nullptr;
        other.valid = false;
    }
    return *this;
}
