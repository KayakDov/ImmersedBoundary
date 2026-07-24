//
// Created by usr on 1/5/26.
//

#include "../headers/SimpleArray.h"
#include "deviceArrays/headers/Tensor.h"
#include <string>

template<typename T>
SimpleArray<T>::SimpleArray(size_t size, std::shared_ptr<T> ptr, bool initDescr): Vec<T>(size, ptr, 1) {
}

template<typename T>
SimpleArray<T> SimpleArray<T>::create(size_t size, cudaStream_t stream, bool initDescr) {
    auto preSimple = Vec<T>::create(size, stream);
    return SimpleArray(size, preSimple.ptr(), initDescr);
}

template<typename T>
SimpleArray<T> SimpleArray<T>::create(std::vector<T> hostData, cudaStream_t stream, bool initDescr) {
    auto deviceData = SimpleArray<T>::create(hostData.size(), stream, initDescr);
    deviceData.set(hostData.data(), stream);
    return deviceData;
}

template<typename T>
SimpleArray<T> SimpleArray<T>::empty() {
    return SimpleArray<T>(0, nonOwningGpuPtr<T>(nullptr));
}

template<typename T>
SimpleArray<T>::SimpleArray(Vec<T> vecWithLD1)
    : SimpleArray<T>(vecWithLD1.size(), vecWithLD1.ptr()) {

    if (vecWithLD1._ld > 1) {
        throw std::invalid_argument(
            "SimpleArray requires a contiguous Vec (leading dimension must be 1). "
            "Received Vec with _ld = " + std::to_string(vecWithLD1._ld)
        );
    }
}

template<typename T>
const SimpleArray<T> SimpleArray<T>::subArray(size_t offset, size_t length) const {
    auto subArray = Vec<T>::subVec(offset, length, 1);
    return {length, subArray.ptr()};
}

template<typename T>
void SimpleArray<T>::initDescr() const{
    cusparseDnVecDescr_t rawDescr;

    const cudaDataType valueType = cuValueType<T>();

    CHECK_SPARSE_ERROR(cusparseCreateDnVec(
        &rawDescr,
        this->size(),        // Vector length
        (void*)this->data(), // Raw device pointer from GpuArray
        valueType
    ));

    dnVecDescr = DnVecDescrPtr(rawDescr, [](const cusparseDnVecDescr_t p) {
        if (p) cusparseDestroyDnVec(p);
    });
}

template<typename T>
cusparseDnVecDescr_t SimpleArray<T>::getDescr() const {
    if (!dnVecDescr) initDescr();
    return dnVecDescr.get();
}

template<typename T>
SimpleArray<T>::operator cusparseDnVecDescr_t() const {
    return getDescr();
}

template<typename T>
Tensor<T> SimpleArray<T>::tensor(size_t rows, size_t layers) const{
    const size_t ld = rows * layers;
    const size_t cols = this->size()/ld;
    return Tensor<T>(rows, cols, layers, ld, this->_ptr);
}

template class SimpleArray<uint32_t>;
template class SimpleArray<int32_t>;
template class SimpleArray<int64_t>;
template class SimpleArray<size_t>;
template class SimpleArray<double>;
template class SimpleArray<float>;
template class SimpleArray<unsigned char>;