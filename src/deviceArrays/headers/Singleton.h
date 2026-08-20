/**
 * @file Singleton.h
 * @brief Defines a GPU-resident scalar wrapper used by vector and matrix operations.
 * @ingroup device_arrays
 *
 * @details
 * These classes form the core GPU container layer. Public containers generally own or view CUDA device memory through shared ownership, while explicit factory functions allocate new storage.
 */

#ifndef BICGSTAB_SINGLETON_H
#define BICGSTAB_SINGLETON_H

#include "deviceArrays/headers/SimpleArray.h"
#include <unordered_map>
#include <memory>

template<typename T> class Mat;
template<typename T> class Tensor;

/**
 * @brief Represents a single-element vector on the GPU.
 * 
 * Inherits from Vec<T> and is designed to provide convenient access to a single scalar
 * while still using the GPU-backed memory model of Vec<T>. Useful for operations that
 * require a scalar in GPU computations, e.g., as alpha/beta in matrix-vector multiplications.
 * 
 * @tparam T Type of the element.
 */
template <typename T>
class Singleton final : public SimpleArray<T> {
private:
    /**
     * @brief Private constructor from shared pointer.
     * @param ptr Shared pointer to device memory holding the single value.
     */
    explicit Singleton(GpuPointer<T> ptr);

    // Grant access to Vec/Mat/Tensor getters that return a Singleton<T>
    friend Vec<T>;
    friend Mat<T>;
    friend Tensor<T>;

public:
    /// Predefined constants for convenience

    using Vec<T>::get;  ///< Inherit Vec<T>::get methods
    using Vec<T>::set;  ///< Inherit Vec<T>::set methods

    /**
     * @brief Create an empty Singleton on the device.
     * @param stream Optional CUDA stream to associate with allocation.
     * @return Singleton<T> instance.
     */
    static Singleton<T> create(Handle& stream);

    /**
     * @brief Create a Singleton initialized to a given value.
     * @param val Value to store in the Singleton.
     * @param stream Optional CUDA stream to associate with allocation.
     * @return Singleton<T> instance.
     */
    static Singleton<T> create(T val, Handle& stream);

    /**
     * @brief Get the value stored in this Singleton.
     * @param stream Optional CUDA stream for device synchronization.
     * @return Value of type T.
     */
    T get(Handle& stream) const;

    /**
     * @brief Set the value of this Singleton.
     * @param val Value to store.
     * @param stream Optional CUDA stream for device synchronization.
     */
    void set(T val, Handle& stream);

    /**
     * @brief Sets this scalar to the product of two quotients.
     *
     * Computes @c (numA/denA) * (numB/denB) on the device and stores the
     * result in this Singleton.
     *
     * @param numA Numerator of the first quotient.
     * @param denA Denominator of the first quotient.
     * @param numB Numerator of the second quotient.
     * @param denB Denominator of the second quotient.
     * @param stream CUDA stream used for the operation.
     */
    void setProductOfQuotients(const Singleton<T>& numA, const Singleton<T>& denA, const Singleton<T>& numB, const Singleton<T>& denB, Handle& stream);


    /**
     * @brief Retrieves an existing target or creates a new target of type Singleton.
     *
     * This method checks if the target `result` is already provided. If `result` is not
     * null, it returns the existing target. Otherwise, it creates a new instance of
     * `Singleton` using the given CUDA stream and assigns it to `out_ptr_unique`.
     * The newly created object is then returned.
     *
     * @param result Pointer to an existing `Singleton` instance, if available. If null,
     * a new instance will be created.
     * @param out_ptr_unique Reference to a `std::unique_ptr` to store the newly created
     * `Singleton` instance if `result` is null.
     * @param stream The CUDA stream used for initializing the `Singleton` instance
     * when creating a new target.
     * @return Pointer to the existing or newly created `Singleton` instance.
     */
    static Singleton<T>* _get_or_create_target(Singleton<T> *result, std::unique_ptr<Singleton<T>> &out_ptr_unique, Handle& stream);

    /**
     * @brief Retrieves an existing target or creates a new one with the specified default value.
     *
     * This method fetches a given target if it exists, or initializes and returns a new
     * instance of the target using the provided default value and handle. If no target exists
     * (`result` is null), a new instance is created using `std::unique_ptr` and returned.
     *
     * @param defaultVal The default value used for initializing the new target, if required.
     * @param hand The handle providing additional context such as a stream for target creation.
     * @param result Pointer to an existing target instance, or null if no instance exists.
     * @param out_ptr_unique A unique pointer that holds ownership of a newly created target
     * if `result` is null.
     * @return A pointer to an existing or newly created target of type `Singleton<T>`.
     */
    static const Singleton<T>* _get_or_create_target(T defaultVal, Handle& hand, const Singleton<T>* result, std::unique_ptr<Singleton<T>>& out_ptr_unique);
};

template <typename T>
class GPUScalar {

    /// One GPUScalar<T> instance per device -- created the first time this
    /// type is requested on a given device, never before. Keyed by
    /// Handle::device(), i.e. the same gpuIndex used everywhere else.
    static std::unordered_map<size_t, std::unique_ptr<GPUScalar<T>>> instances;

public:
    /**
     * Holds the Singleton constants.
     */
    SimpleArray<T> base;

    /**
     * Sets up a set of GPU scalars integers from -2 to 2 inclusive, on
     * whichever device hand is bound to.
     * @param hand The device this instance's constants live on.
     */
    explicit GPUScalar(Handle& hand);

    /**
     * Gets a singleton that holds the desired value, on the same device as
     * hand, without allocating new memory except the first time this type
     * is requested on that particular device.
     * @param i An integer between -2 and 2 inclusive.
     * @param hand Determines which device's cache to use (or create).
     * @return A singleton containing the requested integer, resident on hand's device.
     */
    static const Singleton<T>& get(int32_t i, Handle& hand);

    const Singleton<T> ONE;       ///< Singleton containing 1
    const Singleton<T> ZERO;      ///< Singleton containing 0
    const Singleton<T> TWO;
    const Singleton<T> MINUS_ONE; ///< Singleton containing -1
    const Singleton<T> MINUS_TWO;
};



#endif //BICGSTAB_SINGLETON_H
