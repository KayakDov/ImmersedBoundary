//
// Created by dov on 11/4/25.
//

#include "../headers/DeviceData.cuh"

template<typename T>
DeviceData1d<T>::DeviceData1d(const size_t cols, const size_t ld, T *data): cols(cols), ld(ld), data(data) {
}

template<typename T>
T & DeviceData1d<T>::operator[](size_t idx) {
    return data[idx * ld];
}

template<typename T>
__host__ __device__ DeviceData2d<T>::DeviceData2d(const size_t rows, const size_t cols, const size_t ld, T* data) : DeviceData1d<T>(cols, ld, data), rows(rows) {
}

template<typename T>
__device__ size_t DeviceData2d<T>::flat(const size_t row, const size_t col) const {
    return col * this->ld + row;
}

template<typename T>
__device__ size_t DeviceData2d<T>::flat(const size_t idx) const {
    return (idx / rows) * this->ld + idx % rows;
}

template<typename T>
__device__ size_t DeviceData2d<T>::row(const size_t idx) const {
    return idx % rows;
}

template<typename T>
__device__ size_t DeviceData2d<T>::col(const size_t idx) const {
    return idx / rows;
}


template<typename T>
__host__ __device__ DeviceData1d<T>::operator const T*() const {
    return static_cast<const T*>(this->data);
}
template<typename T>
__host__ __device__ DeviceData1d<T>::operator T*(){
    return this->data;
}

template<typename T>
__device__ __host__ DeviceData3d<T>::DeviceData3d(size_t rows, size_t cols, size_t layers, size_t ld, T *data): DeviceData2d<T>(rows, cols, ld, data), layers(layers) {
}

template<typename T>
T & DeviceData3d<T>::operator[](GridInd3d ind) {
    return this->operator()(ind.layer * (this -> rows * this -> cols) +  ind.row, ind.col);
}

template<typename T>
__device__ T& DeviceData2d<T>::operator[](const size_t idx) {
    return this->data[flat(idx)];
}

template<typename T>
__device__ T & DeviceData2d<T>::operator[](GridInd2d ind) {
    return this->data[flat(ind.row, ind.col)];
}

template<typename T>
__device__ T& DeviceData2d<T>::operator()(const size_t row, const size_t col) {
    return this->data[flat(row, col)];
}

template<typename T>
__device__ bool operator>=(const GridInd2d &ind, const DeviceData2d<T> &dim) {
    return !(ind < dim);
}

template<typename T>
__device__ bool operator<(const GridInd3d &ind, const DeviceData3d<T> &dim) {
    return ind.row < dim.rows && ind.col < dim.cols && ind.layer < dim.layers;
}

template<typename T>
__device__ bool operator>=(const GridInd3d &ind, const DeviceData3d<T> &dim) {
    return !(ind < dim);
}
template<typename T>
__device__ bool operator<(const GridInd2d &ind, const DeviceData2d<T> &dim) {
    return ind.row < dim.rows && ind.col < dim.cols;
}

template class DeviceData1d<float>;
template class DeviceData1d<double>;
template class DeviceData1d<size_t>;
template class DeviceData1d<int32_t>;
template class DeviceData1d<unsigned char>;

template class DeviceData2d<float>;
template class DeviceData2d<double>;
template class DeviceData2d<size_t>;
template class DeviceData2d<int32_t>;
template class DeviceData2d<unsigned char>;

template class DeviceData3d<float>;
template class DeviceData3d<double>;
template class DeviceData3d<size_t>;
template class DeviceData3d<int32_t>;
template class DeviceData3d<unsigned char>;
