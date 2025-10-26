/**
 * @file Vec.h
 * @brief Defines the Vec class for 1D GPU-stored arrays.
 * 
 * This class represents a vector stored in GPU memory and provides methods
 * for element-wise access, arithmetic operations, and multiplication with
 * Mat or Vec objects. It inherits from GpuArray<T>.
 */

#ifndef BICGSTAB_VEC_H
#define BICGSTAB_VEC_H

#include "Mat.h"
#include "Tensor.h"

/**
 * @class Vec
 * @brief Represents a 1D vector stored on GPU.
 * 
 * Vec provides access to vector elements, supports arithmetic operations,
 * and can multiply with matrices or other vectors. It inherits from GpuArray<T>.
 * 
 * @tparam T The type of elements (e.g., float, double, int32_t).
 */
template <typename T>
class Vec : public GpuArray<T> {
    using GpuArray<T>::mult;

private:
    /**
     * @brief Grants access to Mat and Tensor friend methods that produce Vec views.
     */
    friend Vec<T> Mat<T>::row(size_t index);
    friend Vec<T> Mat<T>::col(size_t index);
    friend Vec<T> Tensor<T>::depth(size_t row, size_t col);

protected:
    /**
     * @brief Protected constructor for internal use or friend classes.
     * 
     * @param size Length of the vector.
     * @param _ptr Shared pointer to underlying GPU memory.
     * @param stride Stride for elements (for views/subvectors).
     */
    Vec(size_t size, std::shared_ptr<T> _ptr, size_t stride);

public:
    /**
     * @brief Factory method to create a new vector of given length.
     * 
     * @param length Number of elements.
     * @param stream Optional CUDA stream.
     * @return Vec<T> instance.
     */
    static Vec<T> create(size_t length, cudaStream_t stream = nullptr);

    /**
     * @brief Returns a subvector view of this vector.
     * 
     * @param offset Starting index.
     * @param length Length of the subvector.
     * @param stride Optional stride between elements.
     * @return Vec<T> representing the subvector.
     */
    Vec<T> subVec(size_t offset, size_t length, size_t stride = 1) const;

    /// @copydoc GpuArray::size
    [[nodiscard]] size_t size() const override;

    /// @copydoc GpuArray::bytes
    [[nodiscard]] size_t bytes() const override;

    /// @copydoc GpuArray::set(const T*, cudaStream_t)
    void set(const T* hostData, cudaStream_t stream) override;

    /// @copydoc GpuArray::get(T*, cudaStream_t) const
    void get(T* hostData, cudaStream_t stream) const override;

    /// @copydoc GpuArray::set(const GpuArray<T>&, cudaStream_t)
    void set(const GpuArray<T>& src, cudaStream_t stream) override;

    /// @copydoc GpuArray::get(GpuArray<T>&, cudaStream_t) const
    void get(GpuArray<T>& dst, cudaStream_t stream) const override;

    /// @copydoc GpuArray::set(std::istream&, bool, bool, cudaStream_t)
    void set(std::istream& input_stream, bool isText, bool isColMjr, cudaStream_t stream) override;

    /// @copydoc GpuArray::get(std::ostream&, bool, bool, cudaStream_t) const
    void get(std::ostream& output_stream, bool isText, bool isColMjr, cudaStream_t stream) const override;

    /// @copydoc GpuArray::fill
    void fill(T val, cudaStream_t stream) override;

    /**
     * @brief Returns a single element as Singleton<T>.
     * 
     * @param i Index of the element.
     * @return Singleton<T> representing the element.
     */
    Singleton<T> get(size_t i);

    /**
     * @brief Multiply this vector by a matrix and optionally store result.
     * 
     * @param other Matrix to multiply with.
     * @param result Optional pointer to store result.
     * @param handle Optional GPU handle for operations.
     * @param alpha Optional scalar multiplier.
     * @param beta Optional scalar addition.
     * @param transpose Whether to transpose this vector.
     * @return Vec<T> result of the multiplication.
     */
    void mult(const Mat<T> &other, Vec &result, Handle *handle = nullptr, const Singleton<T> *alpha = nullptr, const
              Singleton<T> *beta = nullptr, bool transpose = false) const;

    /**
     * @brief Dot product with another vector.
     * 
     * @param other Vector to dot with.
     * @param result Output Singleton<T> for result.
     * @param handle Optional GPU handle.
     * @return Dot product value.
     */
    void mult(const Vec<T>& other, Singleton<T> &result, Handle* handle = nullptr) const;

    /**
     * @brief Operator overload for multiplication with a matrix.
     * 
     * @param other Matrix to multiply with.
     * @return Vec<T> result of multiplication.
     */
    Vec<T> operator*(const Mat<T>& other) const;

    /**
     * @brief Operator overload for dot product with another vector.
     * 
     * @param other Vector to dot with.
     * @return Dot product value.
     */
    T operator*(const Vec<T>& other) const;

    /**
     * @brief Adds a scaled vector to this vector.
     * 
     * @param x Vector to add.
     * @param alpha Scaling factor.
     * @param handle Optional GPU handle.
     */
    void add(const Vec<T>& x, const Singleton<T> *alpha, Handle* handle);

    /**
     * @brief Subtracts a scaled vector from this vector.
     * 
     * @param x Vector to subtract.
     * @param alpha Scaling factor.
     * @param handle Optional GPU handle.
     */
    void sub(const Vec<T>& x, const Singleton<T>* alpha, Handle* handle);

    /**
     * @brief Multiply this vector by a scalar.
     * 
     * @param alpha Scalar multiplier.
     * @param handle Optional GPU handle.
     */
    void mult(const Singleton<T>& alpha, Handle* handle = nullptr);

    /**
     * @brief Fill vector with random values.
     * 
     * @param handle Optional GPU handle.
     */
    void fillRandom(Handle* handle = nullptr);

    /**
     * @brief Element-wise exponential/beta power operation.
     * 
     * @param t Singleton<T> exponent base.
     * @param n Singleton<T> power exponent.
     * @param stream CUDA stream to use.
     */
    void EBEPow(const Singleton<T>& t, const Singleton<T>& n, cudaStream_t stream);

    /**
     * @brief Set this vector to a sum of two scaled vectors.
     * 
     * Performs: this = alpha * a + beta * B
     * 
     * @param a First vector.
     * @param b Second vector.
     * @param alpha Scaling factor for a.
     * @param beta Scaling factor for B.
     * @param handle Optional GPU handle.
     */
    void setSum(const Vec<T> &a, const Vec<T> &b, const Singleton<T> &alpha, const Singleton<T> &beta, Handle *handle);
    void setDifference(const Vec<T>& a, const Vec<T>& b, const Singleton<T>& alpha, const Singleton<T>& beta, Handle* handle);
};

#endif //BICGSTAB_VEC_H
