#include "KroneckerTriplet.h"

template<typename T>
void KroneckerTriplet<T>::multRows(const SimpleArray<T> &other, SimpleArray<T> result, bool transposeThis, Handle &hand)  const{
    Mat<T> otherMat = other.matrix(dim.rows * dim.layers), resultMat = result.matrix(dim.rows * dim.layers);
    otherMat.mult(this->x, &resultMat, &hand, false, !transposeThis);
}

template<typename T>
void KroneckerTriplet<T>::multCols(const SimpleArray<T> &other, SimpleArray<T> result, bool transposeThis, Handle &hand)  const{
    Mat<T> otherMat = other.matrix(dim.rows), resultMat = result.matrix(dim.rows);
    this->y.mult(otherMat, &resultMat, &hand, transposeThis, false);
}

template<typename T>
void KroneckerTriplet<T>::multDepths(const SimpleArray<T> &other, SimpleArray<T> result, bool transposeThis, Handle &hand)  const{
    Tensor<T> resultTensor = result.tensor(dim.rows, dim.layers),
            otherTensor = other.tensor(dim.rows, dim.layers);
    auto dst1 = resultTensor.layerColDepth(0);

    size_t stride = dim.layers * dim.rows;

    Mat<T>::batchMult(
        otherTensor.layerColDepth(0), stride,
        this->z, 0,
        dst1, stride,
        false, !transposeThis, hand,
        dim.cols, GPUConst<T>::get(1), GPUConst<T>::get(0)
    );
}

template<typename T>
KroneckerTriplet<T>::KroneckerTriplet(const XYZ<SquareMat<T>> &mat): XYZ<SquareMat<T>>(mat), dim(mat.y._cols, mat.x._cols, mat.z._cols) {
    std::cout << dim << std::endl;
}

template<typename T>
KroneckerTriplet<T>::KroneckerTriplet(const SquareMat<T> x, const SquareMat<T> y, const SquareMat<T> z): XYZ<SquareMat<T>>(x, y, z), dim(y._cols, x._cols, z._cols) {
}

template<typename T>
void KroneckerTriplet<T>::product(Mat<T> &result, Mat<T>& xDimMultZDimBuffer, Handle &hand) const{
    this->x.multKronecker(this->z, xDimMultZDimBuffer, hand);
    xDimMultZDimBuffer.multKronecker(this->y, result, hand);
}

template<typename T>
Mat<T> KroneckerTriplet<T>::product(Handle &hand) const{
    size_t yzRows = this->y._rows * this->z._rows, yzCols = this->y._cols * this->z._cols;
    auto result = Mat<T>::create(this->x._rows * yzRows, this->x._cols * yzCols);
    auto buffer = Mat<T>::create(yzRows, yzCols);
    product(result, buffer, hand);
    return result;
}

template<typename T>
void KroneckerTriplet<T>::mult(const SimpleArray<T>& other, SimpleArray<T>& result, bool transposeThis, Handle &hand)  const{
    auto buffer = SimpleArray<T>::create(result.size(), hand);
    mult(other, result, transposeThis, buffer, hand);
}

template<typename T>
void KroneckerTriplet<T>::mult(const SimpleArray<T>& other, SimpleArray<T>& result, bool transposeThis, const SimpleArray<T>& resultSizeBuffer, Handle &hand) const{
        multCols(other, result, transposeThis, hand);
        multDepths(result, resultSizeBuffer, transposeThis, hand);
        multRows(resultSizeBuffer, result, transposeThis, hand);
}

template<typename T>
void KroneckerTriplet<T>::mult(const Mat<T>& other, Mat<T>& result, bool transposeThis, Handle &hand)  const{
    auto buffer = SimpleArray<T>::create(result._rows, hand);
    mult(other, result, transposeThis, buffer, hand);
}

template<typename T>
void KroneckerTriplet<T>::mult(const Mat<T>& other, Mat<T>& result, bool transposeThis, SimpleArray<T>& resultHeightBuffer, Handle &hand)  const{
    for (size_t colInd = 0; colInd < other._cols; colInd++) {
        SimpleArray<T> resultCol = result.col(colInd);
        mult(other.col(colInd), resultCol, transposeThis, resultHeightBuffer, hand);
    }
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::xOperator(const GridDim& gridDim, const SquareMat<T>& forRows) {
    auto Iy = SquareMat<T>::create(gridDim.rows);
    auto Iz = SquareMat<T>::create(gridDim.layers);
    return {forRows, Iy, Iz};
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::yOperator(const GridDim& gridDim, const SquareMat<T>& forCols) {
    auto Ix = SquareMat<T>::create(gridDim.cols);
    auto Iz = SquareMat<T>::create(gridDim.layers);
    return {Ix, forCols, Iz};
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::zOperator(const GridDim& gridDim, const SquareMat<T>& forLayers) {
    auto Ix = SquareMat<T>::create(gridDim.cols);
    auto Iy = SquareMat<T>::create(gridDim.rows);
    return {Ix, Iy, forLayers};
}

template<typename T>
KroneckerTriplet<T>::KroneckerTriplet(const SquareMat<T> &X, const SquareMat<T> &Y) : KroneckerTriplet<T>(X, Y, GPUConst<T>::get(1).matrix(1).sqSubMat(0,0,1)) {
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::xOperator2d(const GridDim &gridDim, const SquareMat<T> &forRows) {
    auto Iy = SquareMat<T>::create(gridDim.rows);
    return {forRows, Iy};
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::yOperator2d(const GridDim &gridDim, const SquareMat<T> &forCols) {
    auto Ix = SquareMat<T>::create(gridDim.cols);
    return {Ix, forCols};
}

template<typename T>
void KroneckerTriplet<T>::mult2d(const SimpleArray<T>& other, SimpleArray<T>& result, bool transposeThis, const SimpleArray<T>& resultSizeBuffer, Handle &hand) const {
    multCols(other, resultSizeBuffer, transposeThis, hand);
    multRows(resultSizeBuffer, result, transposeThis, hand);
}

template class KroneckerTriplet<float>;
template class KroneckerTriplet<double>;