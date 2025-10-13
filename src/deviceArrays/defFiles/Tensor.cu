#include "../headers/Tensor.h"

template<typename T>
Tensor<T>::Tensor(size_t rows, size_t cols, size_t layers, size_t ld, std::shared_ptr<T> _ptr):GpuArray<T>(rows, cols * layers, ld, _ptr) {

}

template<typename T>
Tensor<T> Tensor<T>::create(size_t rows, size_t cols, size_t layers, cudaStream_t stream) {
    Mat<T> temp = Mat<T>::create(rows, cols * layers);
    return Tensor<T>(rows, cols, layers, temp._ld, temp._ptr);
}


template<typename T>
Mat<T> Tensor<T>::layer(size_t index) {
    return Mat<T>(this->_rows, this->_cols, this->_ld, std::shared_ptr<T>(this->_ptr, this->_ptr.get() + index * this->_ld * this->_cols));
}

template<typename T>
Vec<T> Tensor<T>::depth(size_t row, size_t col) {
    return Vec<T>(
        this->_layers,
        std::shared_ptr<T>(this->_ptr, this->_ptr.get() + col * this->_ld + row),
        this->_ld * this->_cols
        );
}

template<typename T>
Singleton<T> Tensor<T>::get(size_t row, size_t col, size_t layer) {
    return Singleton<T>(
        std::shared_ptr<T>(this->_ptr, this->_ptr.get() + layer * this->_ld * this->_cols + col * this->_ld + row)
        );
}
