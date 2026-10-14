/**
 * @file Handles.h
 * @brief A fixed group of Handles, and the waiting between them.
 */

#ifndef HANDLES_H
#define HANDLES_H

#include "deviceArrays/headers/handle.h"
#include "solvers/Event.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Owns a fixed number of Handles, each on its own stream, and
 *        orders them against each other with events.
 *
 * Work on separate handles runs concurrently. When one handle's work
 * needs another's results, the waiting methods enqueue a dependency on the
 * device: the patient's stream stalls until everything the slow poke's
 * stream had been given at the time of the call has finished. Nothing
 * here blocks the host. Only synch() does.
 *
 * Every waiting method means "as of this call". The slow poke's event is
 * recorded at the moment of the call and captures what has been enqueued
 * on it up to then; work enqueued on the slow poke afterwards is not
 * waited for, and work enqueued on the patient afterwards runs after the
 * wait. A call in which a handle is both a slow poke and a patient waits
 * on the state before the call, not on the waits of the call itself, and a
 * handle never waits for itself.
 *
 * Handles are held by pointer, so a Handles object can be moved and the
 * references get() returned stay valid. It cannot be copied.
 */
class Handles {
public:
    /**
     * @param count    Number of handles, at least one.
     * @param gpuIndex Device the handles are created on.
     *
     * @throws std::invalid_argument if count is zero.
     */
    explicit Handles(size_t count, GpuIndex gpuIndex = GpuIndex(0))
        : events_(count == 0 ? nullptr : new Event[count]) {
        if (count == 0)
            throw std::invalid_argument("Handles needs at least one handle.");

        handles_.reserve(count);
        for (size_t index = 0; index < count; ++index)
            handles_.push_back(std::make_unique<Handle>(gpuIndex));
    }

    /**
     * @return The number of handles.
     */
    [[nodiscard]] size_t size() const {
        return handles_.size();
    }

    /**
     * @brief One of the handles.
     *
     * Const on the group, not on the handle: a Handle is a stream and
     * enqueueing work on it is the point.
     *
     * @throws std::out_of_range if index is not less than size().
     */
    Handle& get(size_t index) const {
        check(index);
        return *handles_[index];
    }

    /**
     * @brief Makes one handle wait for every other handle.
     *
     * @param patientIndex The handle that waits.
     */
    void waitForAll(size_t patientIndex) const {
        check(patientIndex);
        for (size_t slowPoke = 0; slowPoke < size(); ++slowPoke)
            if (slowPoke != patientIndex)
                waitFor(slowPoke, patientIndex);
    }

    /**
     * @brief Makes every other handle wait for one handle.
     *
     * @param slowPokeIndex The handle the others wait for.
     */
    void allWaitFor(size_t slowPokeIndex) const {
        check(slowPokeIndex);
        events_[slowPokeIndex].record(*handles_[slowPokeIndex]);
        for (size_t patient = 0; patient < size(); ++patient)
            if (patient != slowPokeIndex)
                events_[slowPokeIndex].hold(*handles_[patient]);
    }

    /**
     * @brief Makes one handle wait for one other.
     *
     * @param slowPokeIndex The handle waited for.
     * @param patientIndex  The handle that waits. Waiting for itself does
     *                      nothing.
     */
    void waitFor(size_t slowPokeIndex, size_t patientIndex) const {
        check(slowPokeIndex);
        check(patientIndex);
        if (slowPokeIndex == patientIndex)
            return;

        events_[slowPokeIndex].record(*handles_[slowPokeIndex]);
        events_[slowPokeIndex].hold(*handles_[patientIndex]);
    }

    /**
     * @brief Makes a group of handles wait for a group of handles: every
     *        patient waits for every slow poke.
     *
     * Every slow poke is recorded before any patient waits, so a handle in
     * both groups is waited for as it stood before this call.
     *
     * @param slowPokeIndices The handles waited for.
     * @param patientIndices  The handles that wait.
     */
    void waitFor(const std::vector<size_t>& slowPokeIndices, const std::vector<size_t>& patientIndices) const {
        for (size_t slowPoke : slowPokeIndices)
            check(slowPoke);
        for (size_t patient : patientIndices)
            check(patient);

        for (size_t slowPoke : slowPokeIndices)
            events_[slowPoke].record(*handles_[slowPoke]);

        for (size_t slowPoke : slowPokeIndices)
            for (size_t patient : patientIndices)
                if (patient != slowPoke)
                    events_[slowPoke].hold(*handles_[patient]);
    }

    /**
     * @brief Blocks the host until every handle has finished everything
     *        enqueued on it.
     */
    void synch() const {
        for (const auto& handle : handles_)
            handle->synch();
    }

private:
    std::vector<std::unique_ptr<Handle>> handles_;
    std::unique_ptr<Event[]> events_;

    /**
     * @throws std::out_of_range if index is not less than size().
     */
    void check(size_t index) const {
        if (index >= handles_.size())
            throw std::out_of_range("Handles index " + std::to_string(index) + " is out of range for " + std::to_string(handles_.size()) + " handles.");
    }
};

#endif // HANDLES_H
