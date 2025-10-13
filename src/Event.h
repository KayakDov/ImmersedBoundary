/**
 * @file Event.h
 * @brief Wrapper class for CUDA event management.
 */

#ifndef EVENT_H
#define EVENT_H

#include <cuda_runtime.h>
#include <stdexcept>
#include "deviceArrays/headers/deviceArrays.h"
/**
 * @class Event
 * @brief A RAII wrapper for CUDA events, supporting creation, destruction,
 *        renewal, recording, and stream synchronization.
 */
class Event {
public:
    /**
     * @brief Constructor. Creates a CUDA event.
     */
    Event();

    /**
     * @brief Destructor. Destroys the CUDA event if it exists.
     */
    ~Event();

    /**
     * @brief Renew the event by destroying and recreating it.
     */
    void renew();

    /**
     * @brief Record this event on the stream managed by a Handle.
     * @param h Reference to a Handle whose stream is used for recording.
     */
    void record(const Handle& h);

    /**
     * @brief Make the stream in the given Handle wait until this event is complete.
     * @param h Reference to a Handle whose stream will wait on this event.
     */
    void wait(const Handle& h) const;

    /**
     * @brief the even in case it has been used in the past, and records.
     * @param h Reference to a Handle whose stream is used for recording.
     */
    void renewAndRecord(const Handle& h);

    /// Deleted copy constructor and copy assignment (events are not copyable).
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    /// Allow move semantics.
    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;

private:
    cudaEvent_t event; ///< The underlying CUDA event.
    bool valid;        ///< Tracks whether the event is currently valid.
};

#endif // EVENT_H
