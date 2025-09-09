#include "deviceArrays.h"

template <typename T>
Singleton<T>::Singleton():Vec<T>(1){}

template <typename T>
Singleton<T>::Singleton(const Vec<T>& superVector, int index): Vec<T>(superVector, index, 1){}

template <typename T>
Singleton<T>::Singleton(const Mat<T>& superMatrix, int row, int col):Vec<T>(1,1,1){
    const size_t offset = col * superMatrix.getLD() + row;
    this->_ptr = std::shared_ptr<void>(
        superMatrix.getPtr(),
        static_cast<char*>(superMatrix.getPtr().get()) + offset * sizeof(T)
    );
}

template<typename T>
Singleton<T>::Singleton(T val, Handle* hand) : Singleton<T>() {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(hand, temp_hand_ptr);
    set(val, hand->stream);
}

template <typename T>
T Singleton<T>::get(cudaStream_t stream) const{
    T cpuPointer[1];
    this->Vec<T>::get(cpuPointer, stream);
    return cpuPointer[0];
}
template <typename T>
void Singleton<T>::set(const T val, cudaStream_t stream){
    T cpuPointer[1];
    cpuPointer[0] = val;
    this->Vec<T>::set(cpuPointer, stream);    
}

template <typename T>
const Singleton<T> Singleton<T>::ONE(static_cast<T>(1));

template <typename T>
const Singleton<T> Singleton<T>::ZERO(static_cast<T>(0));

template <typename T>
const Singleton<T> Singleton<T>::MINUS_ONE(static_cast<T>(-1));


template class Singleton<int>;
template class Singleton<float>;
template class Singleton<double>;