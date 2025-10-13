//
// Created by dov on 10/10/25.
//

#ifndef BICGSTAB_TENSOR_H
#define BICGSTAB_TENSOR_H

#include "deviceArrays.h"

template <typename T>
class Tensor final : public Mat<T> {
private:
    Tensor(size_t rows, size_t cols, size_t layers, size_t ld, std::shared_ptr<T> _ptr);
public:
    static Tensor<T> create(size_t rows, size_t cols, size_t layers, cudaStream_t stream);
    Mat<T> layer(size_t index);
    Vec<T> depth(size_t row, size_t col);
    Singleton<T> get(size_t row, size_t col, size_t layer);
};

#endif //BICGSTAB_TENSOR_H