/**
 * @file Handles.h
 * @brief A fixed group of Handles, and the waiting between them.
 */

#ifndef HANDLES_H
#define HANDLES_H

#include "deviceArrays/headers/handle.h"
#include "deviceArrays/headers/Support/Event.h"

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
 * Like a single Handle, a Handles is never copied or moved: it is passed
 * by reference or by pointer, so the references operator[] returns stay valid
 * for as long as the object lives.
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

        marks_.resize(count);
        handles_.reserve(count);
        for (size_t index = 0; index < count; ++index)
            handles_.push_back(std::make_unique<Handle>(gpuIndex));
    }

    Handles(const Handles&) = delete;
    Handles& operator=(const Handles&) = delete;
    Handles(Handles&&) = delete;
    Handles& operator=(Handles&&) = delete;

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
    Handle& operator[](size_t index) const {
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
     * @brief Marks the point the slow poke has reached, so a handle can wait for it later.
     *
     * Unlike waitFor, which waits for the slow poke as of the call, a mark is taken now and
     * waited for at some later line of code (waitForMark).  Work enqueued on the slow poke after
     * the mark is not waited for.
     *
     * Each (slow poke, slot) pair holds one mark: marking the same pair again replaces the earlier
     * mark, as waitForMark waits for the most recent one.  Use different slots for marks that
     * must be outstanding at the same time.  Any number of handles may wait for one mark, and one
     * mark may be waited for repeatedly.  The mark's event is created the first time its slot is
     * used and reused after that, so marking in a loop costs one event record and no allocation.
     * Marks are separate from the events waitFor uses, so waitFor never disturbs a mark.
     *
     * @param slowPokeIndex The handle whose progress is marked.
     * @param slot          Which of the slow poke's marks to set; 0 unless several are needed at once.
     */
    void markCompletion(size_t slowPokeIndex, size_t slot = 0) const {
        check(slowPokeIndex);
        auto& slots = marks_[slowPokeIndex];
        while (slots.size() <= slot)
            slots.push_back(std::make_unique<Event>());
        slots[slot]->record(*handles_[slowPokeIndex]);
    }

    /**
     * @brief Makes a handle wait for the most recent mark of a slow poke.
     *
     * If the slot has never been marked there is nothing to wait for and this does nothing, as with a
     * CUDA event that has not been recorded.  Waiting for yourself does nothing.
     *
     * @param slowPokeIndex The handle whose mark is waited for.
     * @param patientIndex  The handle that waits.
     * @param slot          The mark's slot, as passed to markCompletion.
     */
    void waitForMark(size_t slowPokeIndex, size_t patientIndex, size_t slot = 0) const {
        check(slowPokeIndex);
        check(patientIndex);
        const auto& slots = marks_[slowPokeIndex];
        if (slowPokeIndex == patientIndex || slot >= slots.size())
            return;
        slots[slot]->hold(*handles_[patientIndex]);
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
    /// marks_[handle][slot]; grown on first use of a slot, never shrunk.
    mutable std::vector<std::vector<std::unique_ptr<Event>>> marks_;

    /**
     * @throws std::out_of_range if index is not less than size().
     */
    void check(size_t index) const {
        if (index >= handles_.size())
            throw std::out_of_range("Handles index " + std::to_string(index) + " is out of range for " + std::to_string(handles_.size()) + " handles.");
    }
};

#endif // HANDLES_H
