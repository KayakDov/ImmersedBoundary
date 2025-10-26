#ifndef BICGSTAB_SINGLETON_H
#define BICGSTAB_SINGLETON_H

#include "vec.h"

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
    friend Singleton<T> Vec<T>::get(size_t i);
    friend Singleton<T> Mat<T>::get(size_t row, size_t col);
    friend Singleton<T> Tensor<T>::get(size_t row, size_t col, size_t layer);

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
    void setProductOfQutients(const Singleton<T>& numA, const Singleton<T>& denA, const Singleton<T>& numB, const Singleton<T>& denB, cudaStream_t stream);
};

#endif //BICGSTAB_SINGLETON_H
