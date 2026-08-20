/**
 * @file Event.h
 * @brief RAII wrapper for CUDA event management using smart pointers.
 * @ingroup solvers
 */

#ifndef EVENT_H
#define EVENT_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <memory>
#include "../deviceArrays/headers/GpuArray.h"
#include "../deviceArrays/headers/handle.h" // for ensure_device_impl

/**
 * @class Event
 * @brief A robust RAII wrapper around a CUDA event using std::unique_ptr.
 *
 * This class manages a CUDA event (`cudaEvent_t`) through a smart pointer
 * with a custom deleter. It supports:
 *  - creation
 *  - destruction (automatic via RAII)
 *  - recording on a CUDA stream
 *  - making a stream wait on the event
 *
 * The event is always either:
 *  - owned exclusively by this class, or
 *  - null (if moved-from)
 *
 * No destructor is needed, and no `valid` flag is used.
 *
 * @par Multi-GPU behavior
 * A `cudaEvent_t` is permanently bound to whichever device is current when
 * it is created -- there is no CUDA call that rebinds an existing event to a
 * different device. `cudaEventRecord()` enforces this: it fails outright if
 * the event and the stream it's being recorded on belong to different
 * devices. `cudaStreamWaitEvent()`, by contrast, is explicitly documented as
 * safe across devices -- it's the intended mechanism for making one GPU's
 * stream wait on another GPU's event.
 *
 * Because of that asymmetry, this class handles the two operations
 * differently:
 *  - hold() needs no special handling. It works across devices by design.
 *  - record() tracks which device its underlying event was created on. If
 *    asked to record against a Handle on a *different* device, it destroys
 *    the current event and transparently creates a new one on the new
 *    device before recording -- there being no other way to make an event
 *    usable on a device it wasn't created on.
 *
 * @warning That transparent recreation is a real cost, not just bookkeeping
 * -- an event destroy plus an event create, not a flag flip. An Event that
 * gets record()'d alternately from two different GPUs (e.g. in a loop that
 * round-robins devices) will pay this cost on *every single call*, since
 * each call undoes the device switch the previous call made. If you need to
 * record events on more than one device, keep one Event per device rather
 * than sharing a single Event across them.
 */
class Event {
public:

    /**
     * @brief Constructs a new CUDA event on whatever device is currently
     * active.
     *
     * The event is created with `cudaEventDisableTiming` for minimal overhead.
     * The device it lands on is recorded internally; see the class-level
     * comment for what happens if record() is later called against a
     * different device.
     *
     * @throws std::runtime_error if CUDA fails to create the event.
     */
    Event();

    /**
     * @brief Records the event on the CUDA stream associated with the given Handle.
     *
     * If `h` belongs to a different device than the one this event currently
     * lives on, the existing event is destroyed and a new one is
     * transparently created on `h`'s device before recording -- see the
     * class-level @par Multi-GPU behavior section for why this is necessary
     * and what it costs.
     *
     * @param h A Handle object containing a CUDA stream.
     * @throws std::runtime_error if the event is null (moved-from) or if
     *         the underlying CUDA calls fail.
     */
    void record(const Handle& h) const;

    /**
     * @brief Makes the stream in the given Handle wait until this event completes.
     *
     * Unlike record(), this is safe to call with a Handle on a different
     * device than the one this event was created on -- cudaStreamWaitEvent()
     * is explicitly documented as supporting cross-device synchronization,
     * so no device tracking or recreation happens here.
     *
     * @param h A Handle object containing a CUDA stream.
     * @throws std::runtime_error if the event is null or if `cudaStreamWaitEvent` fails.
     */
    void hold(const Handle& h) const;

    ~Event() { gpuIndex_.switchDevice(); }
private:

    /**
     * @struct EventDeleter
     * @brief Custom deleter that destroys the CUDA event when the unique_ptr resets.
     */
    struct EventDeleter {
        void operator()(cudaEvent_t e) const noexcept {

            if (e) CHECK_CUDA_ERROR(cudaEventDestroy(e));
        }
    };

    /// @brief Smart pointer owning the CUDA event.
    using EventPtr = std::unique_ptr<std::remove_pointer<cudaEvent_t>::type, EventDeleter>;

    // Both mutable: record() is logically const from the caller's point of
    // view (it doesn't change what Event conceptually represents), but may
    // need to destroy and recreate the underlying event internally to
    // service a cross-device call. See record()'s implementation.
    mutable EventPtr event; ///< The owned CUDA event (or null if moved-from).
    mutable GpuIndex gpuIndex_; ///< Device this event currently lives on.
};

#endif // EVENT_H
