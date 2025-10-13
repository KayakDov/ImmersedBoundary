
#ifndef BICGSTAB_SINGLETON_H
#define BICGSTAB_SINGLETON_H

#include "vec.h"

template <typename T>
class Singleton final : public Vec<T> {
private:
    friend Singleton<T> Vec<T>::get(size_t i);
    friend Singleton<T> Mat<T>::get(size_t row, size_t col);
    friend Singleton<T> Tensor<T>::get(size_t row, size_t col, size_t layer);

    explicit Singleton(std::shared_ptr<T> ptr);
public:
    static const Singleton<T> ONE, ZERO, MINUS_ONE;

    using Vec<T>::get;
    using Vec<T>::set;

    static Singleton<T> create(cudaStream_t stream = 0);
    static Singleton<T> create(T val,cudaStream_t = 0);

    T get(cudaStream_t stream = nullptr) const;
    void set(T val, cudaStream_t stream);
};


#endif //BICGSTAB_SINGLETON_H