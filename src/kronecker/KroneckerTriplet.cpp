#include "KroneckerTriplet.h"

#include <filesystem>

#include "solvers/Event.h"

template<typename T>
void KroneckerTriplet<T>::multRows(const SimpleArray<T> &other, SimpleArray<T> result, Handle &hand)  const{
    Mat<T> otherMat = other.matrix(dim.rows * dim.layers), resultMat = result.matrix(dim.rows * dim.layers);
    otherMat.mult(this->x, &resultMat, &hand, false, !transpose.x);
}

template<typename T>
void KroneckerTriplet<T>::multCols(const SimpleArray<T> &other, SimpleArray<T> result, Handle &hand)  const{
    Mat<T> otherMat = other.matrix(dim.rows), resultMat = result.matrix(dim.rows);
    this->y.mult(otherMat, &resultMat, &hand, transpose.y, false);
}

template<typename T>
void KroneckerTriplet<T>::multDepths(const SimpleArray<T> &other, SimpleArray<T> result, Handle &hand)  const{
    Tensor<T> resultTensor = result.tensor(dim.rows, dim.layers),
            otherTensor = other.tensor(dim.rows, dim.layers);
    auto dst1 = resultTensor.layerColDepth(0);

    size_t stride = dim.layers * dim.rows;

    Mat<T>::batchMult(
        otherTensor.layerColDepth(0), stride,
        this->z, 0,
        dst1, stride,
        false, !transpose.z, hand,
        dim.cols, GPUScalar<T>::get(1), GPUScalar<T>::get(0)
    );
}

template<typename T>
KroneckerTriplet<T>::KroneckerTriplet(const XYZ<SquareMat<T>> &mat, const XYZ<bool>& transpose):
    XYZ<SquareMat<T>>(mat),
    dim(mat.y._cols, mat.x._cols, mat.z.size() == 0 ? 1 : mat.z._cols),
    transpose(transpose) {
}

template<typename T>
KroneckerTriplet<T>::KroneckerTriplet(const SquareMat<T> x, const SquareMat<T> y, const SquareMat<T> z, XYZ<bool> transpose):
    XYZ<SquareMat<T>>(x, y, z), dim(y._cols, x._cols, z.size() == 0 ? 1 : z._cols),
    transpose(transpose) {
}

template<typename T>
void KroneckerTriplet<T>::product(Mat<T> &result, Mat<T>& xDimXZDimBuffer, Handle &hand) const{
    this->x.multKronecker(this->z, xDimXZDimBuffer, hand);
    xDimXZDimBuffer.multKronecker(this->y, result, hand);
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
void KroneckerTriplet<T>::mult(const SimpleArray<T>& other, SimpleArray<T>& result, Handle &hand)  const{
    auto buffer = SimpleArray<T>::create(result.size(), hand);
    mult(other, result, buffer, hand);
}

template<typename T>
void KroneckerTriplet<T>::mult(const SimpleArray<T>& other, SimpleArray<T>& result, const SimpleArray<T>& resultSizeBuffer, Handle &hand) const{
        multCols(other, result, hand);
        multDepths(result, resultSizeBuffer, hand);
        multRows(resultSizeBuffer, result, hand);
}

template<typename T>
void KroneckerTriplet<T>::mult(const Mat<T>& other, Mat<T>& result, Handle &hand)  const{
    auto buffer = SimpleArray<T>::create(result._rows, hand);
    mult(other, result, buffer, hand);
}

template<typename T>
void KroneckerTriplet<T>::mult(const Mat<T>& other, Mat<T>& result, SimpleArray<T>& resultHeightBuffer, Handle &hand)  const{
    for (size_t colInd = 0; colInd < other._cols; colInd++) {
        SimpleArray<T> resultCol = result.col(colInd);
        mult(other.col(colInd), resultCol, resultHeightBuffer, hand);
    }
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::xOperator(const GridDim& gridDim, const SquareMat<T>& forRows, bool transpose) {
    auto Iy = SquareMat<T>::create(gridDim.rows);
    auto Iz = SquareMat<T>::create(gridDim.layers);
    return {forRows, Iy, Iz, {transpose, false, false}};
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::yOperator(const GridDim& gridDim, const SquareMat<T>& forCols, bool transpose) {
    auto Ix = SquareMat<T>::create(gridDim.cols);
    auto Iz = SquareMat<T>::create(gridDim.layers);
    return {Ix, forCols, Iz, {false, transpose, false}};
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::zOperator(const GridDim& gridDim, const SquareMat<T>& forLayers, bool transpose) {
    auto Ix = SquareMat<T>::create(gridDim.cols);
    auto Iy = SquareMat<T>::create(gridDim.rows);
    return {Ix, Iy, forLayers, {false, false, transpose}};
}

template<typename T>
KroneckerTriplet<T>::KroneckerTriplet(const SquareMat<T> &X, const SquareMat<T> &Y, bool transpX, bool transpY) :
    KroneckerTriplet<T>(
        X,
        Y,
        GPUScalar<T>::get(1).matrix(1).sqSubMat(0,0,1),
        {transpX, transpY, false}
    ) {
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::xOperator2d(const GridDim &gridDim, const SquareMat<T> &forRows, bool transpose) {
    auto Iy = SquareMat<T>::create(gridDim.rows);
    return {forRows, Iy, transpose, false};
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::yOperator2d(const GridDim &gridDim, const SquareMat<T> &forCols, bool transpose) {
    auto Ix = SquareMat<T>::create(gridDim.cols);
    return {Ix, forCols, false, transpose};
}

template<typename T>
void KroneckerTriplet<T>::mult2d(const SimpleArray<T>& other, SimpleArray<T>& result, const SimpleArray<T>& resultSizeBuffer, Handle &hand) const {
    multCols(other, resultSizeBuffer, hand);
    multRows(resultSizeBuffer, result, hand);
}


template<typename Int, typename Real>
XYZ<size_t> getBufSize(XYZ<SquareMat<Real>>& mat, XYZ<bool>& orthonormal, Handle& hand) {
    size_t xBufferSize = orthonormal.x ? 0 : mat.x.template factorLUBufferSize<Int>(hand);
    size_t yBufferSize = orthonormal.y ? 0 : (mat.y._cols == mat.x._cols ? xBufferSize : mat.y.template factorLUBufferSize<Int>(hand));
    size_t zBufferSize = orthonormal.z ? 0 : (mat.z._cols == mat.x._cols ? xBufferSize : (mat.z._cols == mat.y._cols ? yBufferSize : mat.z.template factorLUBufferSize<Int>(hand)));
    return {xBufferSize, yBufferSize, zBufferSize};
}

template<typename Real>
XYZ<SimpleArray<Real>> getBuffer(XYZ<size_t> bufferSize, Handle& hand) {
    auto preBuffer = SimpleArray<Real>::create(bufferSize.x + bufferSize.y + bufferSize.z, hand);
    return {
        preBuffer.subArray(0, bufferSize.x),
        preBuffer.subArray(bufferSize.x, bufferSize.y),
        preBuffer.subArray(bufferSize.x + bufferSize.y, bufferSize.z)
    };
}

template<typename Int>
XYZ<SimpleArray<Int>> getPivot(SimpleArray<Int>& preRowOps, size_t xRows, size_t yRows, size_t zRows) {
    return {
        preRowOps.subArray(0, xRows),
        preRowOps.subArray(xRows, yRows),
        preRowOps.subArray(xRows + yRows, zRows)
    };
}


template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::generateInverse(Handle* hand3, XYZ<SquareMat<T>>& inverseGoesHere, Event* event2) const {

    XYZ<bool> orthonormal(inverseGoesHere.x.size() == 0, inverseGoesHere.y.size() == 0, inverseGoesHere.z.size() == 0);

    for (size_t i = 1; i < 3; i++) {
        event2[i - 1].record(hand3[0]);
        event2[i - 1].hold(hand3[i]);
    }
    XYZ<SquareMat<T>> copy(
        orthonormal.x ? SquareMat<T>::empty() : SquareMat<T>::create(this->x._rows),
        orthonormal.y ? SquareMat<T>::empty() : SquareMat<T>::create(this->y._rows),
        orthonormal.z ? SquareMat<T>::empty() : SquareMat<T>::create(this->z._rows)
    );

    for (size_t i = 0; i < 3; i++) if (!orthonormal[i]) copy[i].set((*this)[i], hand3[i]);
    for (size_t i = 1; i < 3; i++) if (!orthonormal[i]) {
        event2[i - 1].record(hand3[i]);
        event2[i - 1].hold(hand3[0]);
    }

    using Int = int32_t;//This should be made larger if x, y, and z are huge huge huge, which is porbbaly not possible with present technology.

    auto bufferSize = getBufSize<Int, T>(copy, orthonormal, hand3[0]);
    auto buffer = getBuffer<T>(bufferSize, hand3[0]);
    XYZ<size_t> size((!orthonormal.x) * dim.cols, (!orthonormal.y) * dim.rows, (!orthonormal.z) * dim.layers);
    auto preRowOps = SimpleArray<Int>::create(size.x + size.y + size.z + 3, hand3[0]);
    auto pivot = getPivot<Int>(preRowOps, size.x, size.y, size.z);
    size_t numSingletons = (!orthonormal.x) + (!orthonormal.y) + (!orthonormal.z);
    auto info = preRowOps.subArray(preRowOps.size() - numSingletons, numSingletons);


    for (size_t i = 1; i < 3; i++) if (!orthonormal[i]) {
        event2[i - 1].record(hand3[0]);
        event2[i - 1].hold(hand3[i]);
    }

    XYZ<SquareMat<T>> result(
        orthonormal.x ? this->x : inverseGoesHere.x,
        orthonormal.y ? this->y : inverseGoesHere.y,
        orthonormal.z ? this->z : inverseGoesHere.z
    );


    for (size_t i = 0, infoInd = 0; i < 3; i++)
        if (!orthonormal[i]) copy[i].inverse(result[i], pivot[i], hand3[i], info.get(infoInd++), buffer[i], false);//TODO:Test inverse

    for (size_t i = 1; i < 3; i++) {
        event2[i - 1].record(hand3[i]);
        event2[i - 1].hold(hand3[0]);
    }
    return {result, orthonormal};
}

template class KroneckerTriplet<float>;
template class KroneckerTriplet<double>;