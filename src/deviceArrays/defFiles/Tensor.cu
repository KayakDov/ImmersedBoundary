#include "../headers/Tensor.h"

template<typename T>
Tensor<T>::Tensor(size_t rows, size_t cols, size_t layers, size_t ld, std::shared_ptr<T> _ptr) :
    GpuArray<T>(rows, cols, ld, _ptr), _layers(layers), utilityMatrix(subMatrix(0,0,0, rows * layers, cols, ld)) {}


template<typename T>
Mat<T> Tensor<T>::subMatrix(size_t startRow, size_t startCol, size_t startLayer, size_t height, size_t width, size_t ld) const{
    return Mat<T>(
        height,
        width,
        ld,
        std::shared_ptr<T>(this->_ptr, this->_ptr.get() +
            startLayer * this->_rows + startCol * this->_ld + startRow));
}

template<typename T>
Tensor<T> Tensor<T>::create(size_t rows, size_t cols, size_t layers, cudaStream_t stream) {
    Mat<T> temp = Mat<T>::create(rows * layers, cols);
    return Tensor<T>(rows, cols, layers, temp._ld, temp._ptr);
}


template<typename T>
Mat<T> Tensor<T>::layerRowCol(size_t deptIndex) const{
    return subMatrix(0, 0, deptIndex, this->_rows, this->_cols, this->_ld);
}

template<typename T>
Mat<T> Tensor<T>::layerColDepth(size_t colIndex) const {
    return subMatrix(0, colIndex, 0, this->_rows, this->_layers, this->_rows);
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
Vec<T> Tensor<T>::row(size_t row, size_t layer) {
    return layerRowCol(layer).row(row);
}

template<typename T>
Singleton<T> Tensor<T>::get(size_t row, size_t col, size_t layer) {
    return Singleton<T>(
        std::shared_ptr<T>(this->_ptr, this->_ptr.get() + layer * this->_ld * this->_cols + col * this->_ld + row)
        );
}

template<typename T>
size_t Tensor<T>::layerSize() const {
    return this->_rows * this->_cols;
}

template<typename T>
DeviceData3d<T> Tensor<T>::toKernel3d() const{
    return DeviceData3d<T>(this->_rows, this->_cols, this->_layers, this->_ld, this->_ptr.get());
}

template<typename T>
DeviceData3d<T> Tensor<T>::toKernel3d() {
    return DeviceData3d<T>(this->_rows, this->_cols, this->_layers, this->_ld, this->_ptr.get());
}

template<typename T>
Tensor<T>::operator DeviceData3d<T>() {
    return toKernel3d();
}

template<typename T>
Tensor<T>::operator DeviceData3d<T>() const{
    return toKernel3d();
}

template<typename T>
KernelPrep Tensor<T>::kernelPrep() const {
    return KernelPrep(this->_cols, this->_rows, this->_layers);
}

template<typename T>
size_t Tensor<T>::size() const {
    return utilityMatrix.size();
}

template<typename T>
void Tensor<T>::set(const T *hostData, cudaStream_t stream) {
    utilityMatrix.set(hostData, stream);

}

template<typename T>
void Tensor<T>::get(T *hostData, cudaStream_t stream) const {
    utilityMatrix.get(hostData, stream);
}

template<typename T>
void Tensor<T>::set(const GpuArray<T> &src, cudaStream_t stream) {
    utilityMatrix.set(src, stream);
}

template<typename T>
void Tensor<T>::get(GpuArray<T> &dst, cudaStream_t stream) const {
    utilityMatrix.get(dst, stream);
}

template<typename T>
void Tensor<T>::set(std::istream &input_stream, bool isText, bool isColMjr, Handle *hand) {
    utilityMatrix.set(input_stream, isText, isColMjr, hand);
}

template<typename T>
std::ostream &Tensor<T>::get(std::ostream &output_stream, bool isText, bool printColMajor, Handle &hand) const {
    return utilityMatrix.get(output_stream, isText, printColMajor, hand);
}

template class Tensor<float>;
template class Tensor<double>;
template class Tensor<size_t>;
template class Tensor<int32_t>;
template class Tensor<unsigned char>;
template class Tensor<uint32_t>;
