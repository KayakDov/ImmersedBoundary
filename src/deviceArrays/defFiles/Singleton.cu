#include "../headers/Singleton.h"

#include <vector>


template<typename T>
Singleton<T>::Singleton(std::shared_ptr<T> ptr):Vec<T>(static_cast<size_t>(1), ptr, static_cast<size_t>(1)) {}

template<typename T>
Singleton<T> Singleton<T>::create(cudaStream_t stream) {
    Vec<T> preSing = Vec<T>::create(static_cast<size_t>(1), stream);
    return preSing.get(0);
}

template<typename T>
Singleton<T> Singleton<T>::create(T val, cudaStream_t stream) {
    Singleton<T> temp = create(stream);
    temp.set(val, stream);
    return temp;
}

template <typename T>
T Singleton<T>::get(cudaStream_t stream) const{
    T cpuPointer[1];
    this->Vec<T>::get(cpuPointer, stream);
    cudaStreamSynchronize(stream);
    return cpuPointer[0];
}
template <typename T>
void Singleton<T>::set(const T val, cudaStream_t stream){
    T cpuPointer[1];
    cpuPointer[0] = val;
    this->Vec<T>::set(cpuPointer, stream);    
}



template <typename T>
__global__ void setProductOfQuotientsKernel(T* result, const T* numA, const T* denA, const T* numB, const T* denB) {
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
        *result = *numA * *numB/(*denA * *denB);
}


template<typename T>
void Singleton<T>::setProductOfQuotients(const Singleton<T> &numA, const Singleton<T> &denA, const Singleton<T> &numB, const Singleton<T> &denB, cudaStream_t stream) {

    constexpr int THREADS_PER_BLOCK = 1;
    int numBlocks = 1;

    setProductOfQuotientsKernel<<<numBlocks, THREADS_PER_BLOCK, 0, stream>>>(
        this->toKernel1d().data, // Destination: 'this' vector
        numA.toKernel1d().data,     // Input 1: 'a' vector
        denA.toKernel1d().data,     // Input 2: 'b' vector
        numB.toKernel1d().data,            // Scalar alpha (passed by value)
        denB.toKernel1d().data             // Scalar beta (passed by value)
    );
}

template <typename T>
Singleton<T>* Singleton<T>::_get_or_create_target(Singleton<T>* result, std::unique_ptr<Singleton<T>>& out_ptr_unique, cudaStream_t stream) {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Singleton<T>>(Singleton<T>::create(stream));
        return out_ptr_unique.get();
    }
}

template <typename T>
const Singleton<T>* Singleton<T>::_get_or_create_target(T defaultVal, Handle& hand, const Singleton<T>* result, std::unique_ptr<Singleton<T>>& out_ptr_unique) {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Singleton<T>>(Singleton<T>::create(defaultVal, hand));
        return out_ptr_unique.get();
    }
}

template<typename T> Consts<T>* Singleton<T>::initConsts;
template<typename T> const Singleton<T>& Singleton<T>::ZERO = Singleton<T>::initConsts->ZERO;
template<typename T> const Singleton<T>& Singleton<T>::ONE = Singleton<T>::initConsts->ONE;
template<typename T> const Singleton<T>& Singleton<T>::TWO = Singleton<T>::initConsts->TWO;
template<typename T> const Singleton<T>& Singleton<T>::MINUS_ONE = Singleton<T>::initConsts->MINUS_ONE;


template<typename T>
Consts<T>::Consts(Handle hand):
    base(SimpleArray<T>::create(4, hand)),
    ZERO(base.get(0)),
    ONE(base.get(1)),
    MINUS_ONE(base.get(2)),
    TWO(base.get(3))
{
    std::vector<T> hostConsts = {
        static_cast<T>(0),
        static_cast<T>(1),
        static_cast<T>(-1),
        static_cast<T>(2)
    };
    base.set(hostConsts.data(), hand);
}


template class Singleton<int32_t>;
template class Singleton<size_t>;
template class Singleton<float>;
template class Singleton<double>;
template class Singleton<unsigned char>;
template class Singleton<uint32_t>;

template class Consts<int32_t>;
template class Consts<size_t>;
template class Consts<float>;
template class Consts<double>;
template class Consts<unsigned char>;
template class Consts<uint32_t>;

