
#ifndef CUDABANDED_GPUSIMPLEARRAY_H
#define CUDABANDED_GPUSIMPLEARRAY_H

#include "Vec.h"

using DnVecDescrPtr = std::shared_ptr<std::remove_pointer<cusparseDnVecDescr_t>::type>;

/**
 * float or double?
 * @tparam T
 * @return
 */
template <typename T>
inline cudaDataType cuValueType() {
    return sizeof(T) == 8 ? CUDA_R_64F : CUDA_R_32F;;
}

template<typename T>
inline cusparseIndexType_t cuIndexType() {
    return (sizeof(T) == 8) ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I;
}


template <typename T>
class SimpleArray: public Vec<T> {

protected:
    mutable DnVecDescrPtr dnVecDescr;

public:
    using GpuArray<T>::col;

    SimpleArray(size_t size, std::shared_ptr<T> ptr, bool initDescr = false);

    static SimpleArray create(size_t size, cudaStream_t stream, bool initDescr = false);

    SimpleArray(Vec<T> vecWithLD1);

    const SimpleArray<T> subArray(size_t offset, size_t length) const;

    void initDescr() const;

    /**
     * @brief Gets or creates the cuSPARSE dense vector descriptor.
     * @return Raw descriptor pointer.
     */
    cusparseDnVecDescr_t getDescr() const;

    operator  cusparseDnVecDescr_t() const;

    /**
     * Creates a tensor that is a window into this data.
     * Be sure that size is divisible by height * layers.
     *
     * This method may create a window to chnage a const object, so be sure to set recipiant of this to const. TODO: set this to pass a type of const pointer, but be prepared for a const cascade.
     *
     * @param layers The number of layers in the tensor.
     * @param height The length of each column.
     * @return a tensor that is a window into this data.
     */
    Tensor<T> tensor(size_t height, size_t layers) const;

    /**
     * The data in this vector reorganized as a matrix.
     * @param height The height of the matrix created. This should be a divisor of size.
     * @return A matrix containing the data in this vector.
     */

    Mat<T> matrix(size_t height) const;

};

#endif //CUDABANDED_GPUSIMPLEARRAY_H
