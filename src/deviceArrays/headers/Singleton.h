#ifndef BICGSTAB_SINGLETON_H
#define BICGSTAB_SINGLETON_H

#include "Vec.h"

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
class Singleton final : public Vec<T> {
private:
    /**
     * @brief Private constructor from shared pointer.
     * @param ptr Shared pointer to device memory holding the single value.
     */
    explicit Singleton(std::shared_ptr<T> ptr);

    // Grant access to Vec/Mat/Tensor getters that return a Singleton<T>
    friend /*Singleton<T> */Vec<T>;//::get(size_t i);
    friend /*Singleton<T>*/ Mat<T>;//::get(size_t row, size_t col);
    friend /*Singleton<T>*/ Tensor<T>;//::get(size_t row, size_t col, size_t layer);

public:
    /// Predefined constants for convenience
    static const Singleton<T> ONE;       ///< Singleton containing 1
    static const Singleton<T> ZERO;      ///< Singleton containing 0
    static const Singleton<T> MINUS_ONE; ///< Singleton containing -1

    using Vec<T>::get;  ///< Inherit Vec<T>::get methods
    using Vec<T>::set;  ///< Inherit Vec<T>::set methods

    /**
     * @brief Create an empty Singleton on the device.
     * @param stream Optional CUDA stream to associate with allocation.
     * @return Singleton<T> instance.
     */
    static Singleton<T> create(cudaStream_t stream = 0);

    /**
     * @brief Create a Singleton initialized to a given value.
     * @param val Value to store in the Singleton.
     * @param stream Optional CUDA stream to associate with allocation.
     * @return Singleton<T> instance.
     */
    static Singleton<T> create(T val, cudaStream_t stream = 0);

    /**
     * @brief Get the value stored in this Singleton.
     * @param stream Optional CUDA stream for device synchronization.
     * @return Value of type T.
     */
    T get(cudaStream_t stream = nullptr) const;

    /**
     * @brief Set the value of this Singleton.
     * @param val Value to store.
     * @param stream Optional CUDA stream for device synchronization.
     */
    void set(T val, cudaStream_t stream);

    /**
     * Multiplies to fractions, and sets this to the result.
     * @param numA
     * @param denA
     * @param numB
     * @param denB
     * @param stream
     */
    void setProductOfQuotients(const Singleton<T>& numA, const Singleton<T>& denA, const Singleton<T>& numB, const Singleton<T>& denB, cudaStream_t stream);


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
    static Singleton<T>* _get_or_create_target(Singleton<T> *result, std::unique_ptr<Singleton<T>> &out_ptr_unique, cudaStream_t stream);

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

#endif //BICGSTAB_SINGLETON_H
