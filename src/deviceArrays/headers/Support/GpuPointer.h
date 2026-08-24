/**
 * @file GpuPointer.h
 * @brief A shared_ptr<T>-backed device pointer that carries its own GpuIndex,
 * plus the deleter and convenience helpers for creating one.
 * @ingroup device_arrays
 */

#ifndef GPU_POINTER_H
#define GPU_POINTER_H

#include <memory>
#include "GpuIndex.cuh"
#include "deviceArrays/headers/handle.h" // for CHECK_CUDA_ERROR -- no type dependency on Handle

// ---------------------------------------------------------------------
// Destruction
// ---------------------------------------------------------------------

/**
 * @brief Deleter for GPU memory obtained via cudaMalloc/cudaMallocPitch,
 * shared (stateless) version -- frees on whatever device happens to be
 * current when it runs.
 *
 * NOT used internally by GpuPointer<T> (see makeCudaFreeDeleter below for
 * why) -- kept available for other code that already uses cudaFreeDeleter
 * directly as a bare shared_ptr deleter and doesn't have a GpuIndex to
 * give it.
 *
 * @warning Marked noexcept to match this codebase's existing deleter
 * convention (see CublasDeleter, EventDeleter in handle.h/Event.h) --
 * which means if cudaFree ever actually fails, CHECK_CUDA_ERROR's throw
 * escaping a noexcept function calls std::terminate rather than
 * propagating. Guarded against the one specific, benign failure mode
 * this is actually known to hit in practice: see the comment inline.
 */
struct CudaFreeDeleter {
    template <typename T>
    void operator()(T* p) const noexcept {
        if (!p) return;
        cudaError_t err = cudaFree(p);
        // cudaErrorCudartUnloading ("driver shutting down") happens when
        // this deleter runs during process exit, after the CUDA driver has
        // already begun its own teardown -- e.g. a static object holding
        // GPU memory getting destroyed as part of global static
        // destruction, whose ordering relative to CUDA's own internal
        // cleanup is unspecified. Nothing useful to do about this specific
        // error -- the process is already exiting and the memory is being
        // reclaimed by the driver's own teardown regardless -- and
        // throwing out of a noexcept function calls std::terminate() and
        // aborts the whole process, which is strictly worse. Swallow only
        // this one case; still check everything else normally.
        if (err != cudaSuccess && err != cudaErrorCudartUnloading)
            CHECK_CUDA_ERROR(err);
    }
};

/** Single reusable instance -- usable bare, e.g. shared_ptr<T>(raw, cudaFreeDeleter). */
inline CudaFreeDeleter cudaFreeDeleter;

/**
 * @brief Builds a per-instance deleter that switches to the right device
 * before freeing -- what GpuPointer<T> actually uses internally.
 *
 * cudaFreeDeleter above can't do this itself: it's a single, stateless,
 * shared functor with no way to know which device any given pointer
 * belongs to. This returns a small lambda that captures gpuIndex by value,
 * so each GpuPointer's owned memory gets freed against the correct device
 * regardless of what's currently active when destruction actually happens.
 */
inline auto makeCudaFreeDeleter(GpuIndex gpuIndex) {
    return [gpuIndex](auto* p) noexcept {
        if (!p) return;
        gpuIndex.switchDevice();
        cudaError_t err = cudaFree(p);
        // Same cudaErrorCudartUnloading reasoning as CudaFreeDeleter above.
        if (err != cudaSuccess && err != cudaErrorCudartUnloading)
            CHECK_CUDA_ERROR(err);
    };
}

// ---------------------------------------------------------------------
// The pointer itself
// ---------------------------------------------------------------------

/**
 * @class GpuPointer
 * @brief Like std::shared_ptr<T>, but every instance knows which device its
 * memory lives on, and that device travels automatically with every copy
 * and every windowed view -- see window() below.
 *
 * Deliberately wraps std::shared_ptr<T> rather than reimplementing
 * reference counting: ownership, custom deleters, and the aliasing
 * mechanism a "window" needs are all delegated straight to the standard
 * library rather than hand-rolled.
 */
template <typename T>
class GpuPointer {
    std::shared_ptr<T> ptr_;
    GpuIndex gpuIndex_;

    /// Low-level constructor used only by window() -- takes an
    /// already-aliased shared_ptr (sharing some other GpuPointer's
    /// refcount) directly, paired with the device that aliased memory
    /// actually lives on. Private: not one of the public ways to create a
    /// GpuPointer from scratch, just window()'s implementation.
    GpuPointer(std::shared_ptr<T> aliasedPtr, GpuIndex gpuIndex) : ptr_(std::move(aliasedPtr)), gpuIndex_(gpuIndex) {}

public:
    /** A GpuPointer that owns nothing -- a valid, non-null-GpuIndex, safe
     *  default/sentinel value. Points to nullptr; nothing to free, so no
     *  device-switching or CUDA call happens on either construction or
     *  destruction of this. */
    static const GpuPointer null;

    /**
     * @brief Constructs a GpuPointer.
     * @param raw The raw device pointer (e.g. straight out of cudaMallocPitch).
     * @param gpuIndex Which device raw lives on -- for the ownsMemory=false
     *        case, this is the caller's responsibility to get right; there's
     *        no way to verify it from the pointer value alone (see the
     *        earlier discussion of cudaPointerGetAttributes for why that's
     *        not a reliable check).
     * @param ownsMemory If true (the default), raw is freed via cudaFree
     *        -- on gpuIndex's device, switched to automatically -- when
     *        the last GpuPointer referencing it is destroyed. If false,
     *        raw is treated as externally owned and never freed here.
     */
    GpuPointer(T* raw, GpuIndex gpuIndex, bool ownsMemory = true)
        : ptr_(ownsMemory
              ? std::shared_ptr<T>(raw, makeCudaFreeDeleter(gpuIndex))
              : std::shared_ptr<T>(raw, [](T*){})),
          gpuIndex_(gpuIndex) {}

    /**
     * @brief Constructs a GpuPointer using whichever device is currently active.
     *
     * Intended for exactly one pattern: switch to the target device,
     * allocate (cudaMallocPitch or similar), then wrap the result with
     * this constructor immediately -- no separate GpuIndex needed, since
     * the device is already known to be correct at that exact point (it's
     * the one you just switched to in order to allocate). The one
     * precondition this relies on: nothing may run between the switch/
     * allocation and this call that could switch the active device again.
     * If anything could intervene, use the explicit-GpuIndex constructor
     * instead rather than relying on ambient state here.
     *
     * @param raw The raw device pointer (e.g. straight out of cudaMallocPitch).
     * @param ownsMemory If true (the default), raw is freed via cudaFree.
     */
    GpuPointer(T* raw, bool ownsMemory = true)
        : GpuPointer(raw, []() {
            int current_device;
            CHECK_CUDA_ERROR(cudaGetDevice(&current_device));
            return GpuIndex(current_device);
        }(), ownsMemory) {}

    /**
     * @brief A window into this same memory at a given element offset --
     * e.g. one column of a matrix, or one element offset into a larger
     * buffer.
     *
     * Shares this GpuPointer's reference count (so the underlying memory
     * stays alive as long as either this GpuPointer or the returned
     * window does, exactly like shared_ptr's own aliasing constructor)
     * and always carries the correct device: a window into a buffer
     * necessarily lives on the same device as the buffer itself, so it's
     * inherited from `this` rather than passed in -- there's no separate
     * value it could correctly be, and no way for it to end up mismatched.
     * It also inherits `this`'s deleter (the one makeCudaFreeDeleter built
     * for the original owner), so freeing still happens on the right
     * device no matter how many windows outlive the original GpuPointer.
     *
     * TODO: create an operator overload + for pointer arithmetic.
     *
     * @param offset Element offset from this->get(), not bytes -- ordinary
     *        pointer arithmetic on a T*, matching how offsets are already
     *        expressed everywhere else in this codebase (e.g. col * _ld + row).
     */
    GpuPointer<T> window(size_t offset) const {
        return GpuPointer<T>(std::shared_ptr<T>(ptr_, ptr_.get() + offset), gpuIndex_);
    }

    /** The raw pointer, exactly as shared_ptr<T>::get() -- unaffected by
     *  anything device-related. Everything downstream that already expects
     *  a raw T* (kernel launches, DeviceData2d, etc.) keeps working unchanged. */
    T* get() const { return ptr_.get(); }

    /** Which device this pointer's memory lives on. */
    GpuIndex gpuIndex() const { return gpuIndex_; }

    long use_count() const { return ptr_.use_count(); }

    /** Releases this GpuPointer's reference now, rather than waiting for
     *  it to go out of scope. If this was the last reference, the
     *  device-aware deleter built in the constructor runs immediately --
     *  same correctness guarantee as ordinary destruction, just on demand. */
    void freeMem() { ptr_.reset(); }
};

template <typename T>
inline const GpuPointer<T> GpuPointer<T>::null = GpuPointer<T>(nullptr, GpuIndex(0), false);

#endif // GPU_POINTER_H