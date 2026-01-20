
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
    return (sizeof(T) == 8) ? CUDA_R_64F : CUDA_R_32F;;
}

template<typename T>
inline cusparseIndexType_t cuIndexType() {
    return (sizeof(T) == 8) ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I;
}


template <typename T>
class SimpleArray: public Vec<T> {

protected:
    SimpleArray(size_t size, std::shared_ptr<T> ptr);
    mutable DnVecDescrPtr dnVecDescr;

public:
    static SimpleArray create(size_t size, cudaStream_t stream);

    SimpleArray(Vec<T> vecWithLD1);

    SimpleArray<T> subAray(size_t offset, size_t length);

    /**
     * @brief Gets or creates the cuSPARSE dense vector descriptor.
     * @return Raw descriptor pointer.
     */
    cusparseDnVecDescr_t getDescr() const;

    operator  cusparseDnVecDescr_t() const;
};

#endif //CUDABANDED_GPUSIMPLEARRAY_H
