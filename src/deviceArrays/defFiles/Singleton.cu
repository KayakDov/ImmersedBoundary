#include "../headers/Singleton.h"

#include <vector>


template<typename T>
Singleton<T>::Singleton(std::shared_ptr<T> ptr):SimpleArray<T>(1, ptr) {}

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
        *result = (*numA / *denA) * (*numB / *denB);
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

template<typename T>
const Singleton<T> & GPUScalar<T>::get(int32_t i) {
    if (!universal) universal = std::make_unique<GPUScalar<T>>();
    switch (i) {
        case 0: return universal->ZERO;
        case 1: return universal->ONE;
        case 2: return universal->TWO;
        case -1: return universal->MINUS_ONE;
        case -2: return universal->MINUS_TWO;
        default: throw std::out_of_range("");
    }
}

template<typename T>
GPUScalar<T>::GPUScalar(Handle hand):
    base(SimpleArray<T>::create(5, hand)),
    ZERO(base.get(0)),
    ONE(base.get(1)),
    MINUS_ONE(base.get(2)),
    TWO(base.get(3)),
    MINUS_TWO(base.get(4))
{
    std::vector<T> hostConsts = {
        static_cast<T>(0),
        static_cast<T>(1),
        static_cast<T>(-1),
        static_cast<T>(2),
        static_cast<T>(-2)
    };
    base.set(hostConsts.data(), hand);
}

template <typename T> std::unique_ptr<GPUScalar<T>> GPUScalar<T>::universal = nullptr;

template class Singleton<int32_t>;
template class Singleton<size_t>;
template class Singleton<float>;
template class Singleton<double>;
template class Singleton<unsigned char>;
template class Singleton<uint32_t>;
template class Singleton<int64_t>;

template class GPUScalar<int32_t>;
template class GPUScalar<size_t>;
template class GPUScalar<float>;
template class GPUScalar<double>;
template class GPUScalar<unsigned char>;
template class GPUScalar<uint32_t>;
template class GPUScalar<int64_t>;

