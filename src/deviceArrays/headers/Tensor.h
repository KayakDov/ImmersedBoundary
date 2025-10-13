/**
 * @file Tensor.h
 * @brief Defines the Tensor class for 3D GPU-stored arrays.
 *
 * This class represents a 3-dimensional tensor (rows x cols x layers) and provides
 * methods to access layers, columns, and individual elements. It inherits from Mat<T>
 * and supports GPU storage and operations.
 */

#ifndef BICGSTAB_TENSOR_H
#define BICGSTAB_TENSOR_H

#include "deviceArrays.h"

/**
 * @class Tensor
 * @brief Represents a 3D tensor stored on GPU.
 *
 * Tensor is a final class that inherits from Mat<T> and provides additional
 * access methods for 3D data. It supports row/column/depth access and allows
 * retrieving individual elements as Singleton<T>.
 *
 * @tparam T The data type of tensor elements (e.g., float, double, int32_t).
 */
template <typename T>
class Tensor final : public Mat<T> {
private:
    /**
     * @brief Private constructor for internal use.
     *
     * Constructs a Tensor with the given dimensions and shared pointer to data.
     * Users should use the static create() method instead.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param layers Number of layers (depth).
     * @param ld Leading dimension of the underlying storage.
     * @param _ptr Shared pointer to the underlying GPU memory.
     */
    Tensor(size_t rows, size_t cols, size_t layers, size_t ld, std::shared_ptr<T> _ptr);

public:
    /**
     * @brief Inherit get() methods from Mat<T> to avoid hiding them.
     */
    using Mat<T>::get;

    /**
     * @brief Factory method to create a Tensor of given dimensions.
     *
     * Allocates GPU memory for a tensor of size rows x cols x layers.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param layers Number of layers (depth).
     * @param stream Optional CUDA stream for GPU operations.
     * @return A new Tensor<T> object.
     */
    static Tensor<T> create(size_t rows, size_t cols, size_t layers, cudaStream_t stream);

    /**
     * @brief Returns a specific layer of the tensor as a Mat<T>.
     *
     * @param index Layer index (0-based).
     * @return Mat<T> representing the requested layer.
     */
    Mat<T> layer(size_t index);

    /**
     * @brief Returns a column-depth vector at the given row and column.
     *
     * @param row Row index (0-based).
     * @param col Column index (0-based).
     * @return Vec<T> representing the depth vector at the specified position.
     */
    Vec<T> depth(size_t row, size_t col);

    /**
     * @brief Returns a single element of the tensor as a Singleton<T>.
     *
     * @param row Row index (0-based).
     * @param col Column index (0-based).
     * @param layer Layer index (0-based).
     * @return Singleton<T> representing the single element at the specified location.
     */
    Singleton<T> get(size_t row, size_t col, size_t layer);
};

#endif //BICGSTAB_TENSOR_H
