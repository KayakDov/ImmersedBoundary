#include "../headers/singleton.h"


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
    return cpuPointer[0];
}
template <typename T>
void Singleton<T>::set(const T val, cudaStream_t stream){
    T cpuPointer[1];
    cpuPointer[0] = val;
    this->Vec<T>::set(cpuPointer, stream);    
}



template <typename T>
__global__ void setProductOfQutientsKernel(T* result, const T* numA, const T* denA, const T* numB, const T* denB) {
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
        *result = *numA * *numB/(*denA * *denB);
}

template<typename T>
void Singleton<T>::setProductOfQutients(const Singleton<T> &numA, const Singleton<T> &denA, const Singleton<T> &numB, const Singleton<T> &denB, cudaStream_t stream) {

    constexpr int THREADS_PER_BLOCK = 1;
    int numBlocks = 1;

    setProductOfQutientsKernel<<<numBlocks, THREADS_PER_BLOCK, 0, stream>>>(
        this->data(), // Destination: 'this' vector
        numA.data(),     // Input 1: 'a' vector
        denA.data(),     // Input 2: 'b' vector
        numB.data(),            // Scalar alpha (passed by value)
        denB.data()             // Scalar beta (passed by value)
    );
}

template <typename T>
const Singleton<T> Singleton<T>::ONE = Singleton<T>::create(static_cast<T>(1));

template <typename T>
const Singleton<T> Singleton<T>::ZERO = Singleton<T>::create(static_cast<T>(0));

template <typename T>
const Singleton<T> Singleton<T>::MINUS_ONE = Singleton<T>::create(static_cast<T>(-1));


template class Singleton<int32_t>;
template class Singleton<size_t>;
template class Singleton<float>;
template class Singleton<double>;
template class Singleton<unsigned char>;
