
#ifndef CUDABANDED_GPUSIMPLEARRAY_H
#define CUDABANDED_GPUSIMPLEARRAY_H

#include "Vec.h"

/**
 * @brief Shared-owner wrapper for a cuSPARSE dense vector descriptor.
 *
 * The pointed-to type is the opaque descriptor object behind
 * @c cusparseDnVecDescr_t. Using a shared pointer allows descriptor lifetime
 * to be tied to C++ object lifetime.
 */
using DnVecDescrPtr = std::shared_ptr<std::remove_pointer<cusparseDnVecDescr_t>::type>;

/**
 * @brief Maps a floating-point C++ type to the corresponding CUDA data type.
 *
 * @tparam T Floating-point scalar type. Intended for @c float or @c double.
 * @return @c CUDA_R_64F when @p T is 8 bytes, otherwise @c CUDA_R_32F.
 */
template <typename T>
inline cudaDataType cuValueType() {
    return sizeof(T) == 8 ? CUDA_R_64F : CUDA_R_32F;;
}

/**
 * @brief Maps an integer C++ type to the corresponding cuSPARSE index type.
 *
 * @tparam T Integer index type.
 * @return @c CUSPARSE_INDEX_64I when @p T is 8 bytes, otherwise
 *         @c CUSPARSE_INDEX_32I.
 */
template<typename T>
inline cusparseIndexType_t cuIndexType() {
    return (sizeof(T) == 8) ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I;
}

/**
 * @brief One-dimensional GPU array with vector operations and cuSPARSE support.
 *
 * @details
 * @c SimpleArray extends @c Vec<T> by adding optional support for a cuSPARSE
 * dense-vector descriptor. It represents a contiguous one-dimensional device
 * allocation or a view into one.
 *
 * The class also provides convenience methods for reinterpreting the same
 * underlying data as lower-dimensional views, such as @c Mat<T> and
 * @c Tensor<T>. These view operations do not allocate new data; they create
 * window/view objects over the existing storage.
 *
 * @tparam T Scalar value type stored on the GPU.
 *
 * @note Subarrays, matrices, and tensors returned from this class are views
 *       into the same underlying memory. Mutating the returned view mutates
 *       the original data.
 */
template <typename T>
class SimpleArray: public Vec<T> {

protected:
    /**
     * @brief Lazily initialized cuSPARSE dense-vector descriptor.
     *
     * The descriptor describes this array as a cuSPARSE dense vector. It may be
     * null until @c initDescr() or @c getDescr() is called, depending on how the
     * object was constructed.
     */
    mutable DnVecDescrPtr dnVecDescr;

public:
    using GpuArray<T>::col;
    using Vec<T>::get;

    /**
     * @brief Constructs a simple array from an existing shared device pointer.
     *
     * @param size Number of scalar elements in the array.
     * @param ptr Shared pointer to the first element of device memory.
     * @param initDescr If true, initializes the cuSPARSE dense-vector
     *                  descriptor during construction.
     */
    SimpleArray(size_t size, std::shared_ptr<T> ptr, bool initDescr = false);

    /**
     * @brief Allocates a new GPU array.
     *
     * @param size Number of scalar elements to allocate.
     * @param stream CUDA stream used for allocation/initialization.
     * @param initDescr If true, initializes the cuSPARSE dense-vector
     *                  descriptor before returning.
     * @return A newly allocated @c SimpleArray<T>.
     */
    static SimpleArray create(size_t size, cudaStream_t stream, bool initDescr = false);

    /**
    * @brief Creates an empty simple array.
    *
    * @return A @c SimpleArray<T> with size zero, leading dimension one through
    *         the vector constructor, and a null device pointer.
    *
    * @note No GPU memory is allocated.
    */
    static SimpleArray empty();


    /**
     * @brief Constructs a simple array view from a vector with leading dimension 1.
     *
     * @param vecWithLD1 Vector whose data should be viewed as a simple
     *                   contiguous array.
     */
    SimpleArray(Vec<T> vecWithLD1);

    /**
     * @brief Creates a view into a contiguous subrange of this array.
     *
     * @param offset Index of the first element in the subarray.
     * @param length Number of elements in the subarray.
     * @return A @c SimpleArray<T> view of the requested subrange.
     *
     * @note The returned object shares memory with this array. No data is copied.
     */
    const SimpleArray<T> subArray(size_t offset, size_t length) const;

    /**
     * @brief Initializes the cuSPARSE dense-vector descriptor for this array.
     *
     * @details
     * This method is logically const because it does not change the represented
     * array data, only the cached descriptor used to pass the array to cuSPARSE.
     */
    void initDescr() const;

    /**
     * @brief Gets or creates the cuSPARSE dense-vector descriptor.
     *
     * @return Raw cuSPARSE dense-vector descriptor pointer.
     */
    cusparseDnVecDescr_t getDescr() const;

    /**
     * @brief Converts this array to a cuSPARSE dense-vector descriptor.
     *
     * @return Raw cuSPARSE dense-vector descriptor pointer.
     *
     * @note This enables passing @c SimpleArray directly to APIs expecting
     *       @c cusparseDnVecDescr_t.
     */
    operator  cusparseDnVecDescr_t() const;

    /**
     * @brief Reinterprets this one-dimensional array as a 3D tensor view.
     *
     * @param rows Number of rows in each layer.
     * @param layers Number of layers in the tensor.
     * @return A @c Tensor<T> view into this array's data.
     *
     * @pre The array size must be divisible by @c rows * @c layers.
     *
     * @note The returned tensor shares memory with this array. No data is copied.
     *       Mutating the returned tensor mutates this array.
     */
    Tensor<T> tensor(size_t rows, size_t layers) const;

    /**
     * @brief Reinterprets this one-dimensional array as a matrix view.
     *
     * @param height Number of rows in the resulting matrix.
     * @return A @c Mat<T> view into this array's data.
     *
     * @pre The array size must be divisible by @c height.
     *
     * @note The returned matrix shares memory with this array. No data is copied.
     *       Mutating the returned matrix mutates this array.
     */
    Mat<T> matrix(size_t height) const;

};

#endif //CUDABANDED_GPUSIMPLEARRAY_H
