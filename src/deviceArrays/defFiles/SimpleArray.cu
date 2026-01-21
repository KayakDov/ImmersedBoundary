//
// Created by usr on 1/5/26.
//

#include "../headers/SimpleArray.h"


template<typename T>
SimpleArray<T>::SimpleArray(size_t size, std::shared_ptr<T> ptr): Vec<T>(size, ptr, 1) {
}

template<typename T>
SimpleArray<T> SimpleArray<T>::create(size_t size, cudaStream_t stream) {
    auto preSimple = Vec<T>::create(size, stream);
    return {size, preSimple.ptr()};
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
SimpleArray<T> SimpleArray<T>::subAray(size_t offset, size_t length) {
    auto subArray = Vec<T>::subVec(offset, length, 1);
    return {length, subArray.ptr()};
}

template<typename T>
cusparseDnVecDescr_t SimpleArray<T>::getDescr() const {
    if (!dnVecDescr) {
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
    return dnVecDescr.get();
}


template<typename T>
SimpleArray<T>::operator cusparseDnVecDescr_t() const {
    return getDescr();
}

template <typename T>
SimpleArray<T> GpuArray<T>::col(const size_t index){
    if (index >= this->_cols) throw std::out_of_range("Out of range");
    return SimpleArray<T>(this->_rows, std::shared_ptr<T>(this->_ptr, this->_ptr.get() + index * this->_ld));
}
template <typename T>
SimpleArray<T> GpuArray<T>::col(const size_t index) const {
    return SimpleArray<T>((const_cast<GpuArray<T>*>(this))->col(index));
}

template<typename T>
SimpleArray<T> Tensor<T>::col(size_t col, size_t layer) {
    return layerRowCol(layer).col(col);
}

template SimpleArray<float>        GpuArray<float>::col(size_t);
template SimpleArray<double>       GpuArray<double>::col(size_t);
template SimpleArray<size_t>       GpuArray<size_t>::col(size_t);
template SimpleArray<int>          GpuArray<int>::col(size_t);
template SimpleArray<unsigned char>GpuArray<unsigned char>::col(size_t);
template SimpleArray<uint32_t>     GpuArray<uint32_t>::col(size_t);

template SimpleArray<float> GpuArray<float>::col(size_t) const;
template SimpleArray<double> GpuArray<double>::col(size_t) const;
template SimpleArray<size_t> GpuArray<size_t>::col(size_t) const; // Maps to 'unsigned long' in the error
template SimpleArray<int> GpuArray<int>::col(size_t) const;
template SimpleArray<unsigned char> GpuArray<unsigned char>::col(size_t) const;
template SimpleArray<uint32_t> GpuArray<uint32_t>::col(size_t) const;


template SimpleArray<double> Tensor<double>::col(size_t col, size_t layer);
template SimpleArray<float> Tensor<float>::col(size_t col, size_t layer);


template class SimpleArray<int>;
template class SimpleArray<uint32_t>;
template class SimpleArray<size_t>;
template class SimpleArray<double>;
template class SimpleArray<float>;