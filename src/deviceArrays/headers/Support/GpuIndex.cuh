/**
 * @file GpuIndex.h
 * @brief A small value type identifying which physical GPU something belongs to.
 * @ingroup device_arrays
 */

#ifndef GPU_INDEX_H
#define GPU_INDEX_H

/**
 * @class GpuIndex
 * @brief Identifies a physical GPU -- same numbering as `nvidia-smi -L` /
 * `CUDA_VISIBLE_DEVICES`.
 *
 * Any class whose resources live on a specific device -- Handle, and
 * (pending a separate, larger retrofit) GpuArray and everything derived
 * from it -- can expose one via a device()/operator GpuIndex() accessor.
 * That lets code holding more than one such object check they agree on a
 * device before doing anything cross-device-sensitive, e.g. before an
 * operation that takes two Mats and assumes they live on the same GPU.
 *
 * GpuIndex is a pure value: constructing one, reading its raw index(), or
 * comparing two, never touches the CUDA driver or changes what device is
 * current. Only ensureDevice() does that, and it's meant to be called
 * immediately before an actual CUDA API call -- not as a side effect of
 * merely asking "what device is this."
 */
class GpuIndex {

public:
    size_t index_;

    GpuIndex(size_t index = 0) : index_(index) {}

    /** The raw device index this refers to. */
    size_t index() const { return index_; }

    /**
     * @brief Makes this the currently active device for this host thread.
     *
     * Cheap and idempotent if this device is already current -- routes
     * through the same thread_local cache as everything else that calls
     * cudaSetDevice(), so calling this repeatedly with the same GpuIndex
     * costs one integer comparison after the first call.
     *
     * This is the only method on GpuIndex with a side effect. Everything
     * else -- construction, index(), comparison -- is a pure query.
     * Nothing calls this automatically; it's meant to be invoked explicitly
     * at the point something is about to make a real CUDA call (see
     * Handle's conversion operators for the intended usage).
     */
    void switchDevice() const;

    friend bool operator==(const GpuIndex& a, const GpuIndex& b) { return a.index_ == b.index_; }
    friend bool operator!=(const GpuIndex& a, const GpuIndex& b) { return !(a == b); }
};

#endif // GPU_INDEX_H
