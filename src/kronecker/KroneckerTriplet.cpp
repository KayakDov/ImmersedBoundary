#include "KroneckerTriplet.h"


template<typename T>
void KroneckerTriplet<T>::mult1D(
    Mat<T> kMat,
    bool transposeThis, bool transposeOperand,
    const Mat<T> &operand1, Mat<T> &dst1,
    size_t stride,
    Handle &hand,
    size_t batchCount
) const {

    const Mat<T> *a, *b;
    size_t aStride, bStride;
    bool transposeA, transposeB;

    if (transposeOperand) {
        a = &operand1;
        aStride = stride;
        b = &kMat;
        bStride = 0;
        transposeA = false;
        transposeB = transposeThis;
    }else {
        a = &kMat;
        aStride = 0;
        b = &operand1;
        bStride = stride;
        transposeA = transposeThis;
        transposeB = false;
    }

    Mat<T>::batchMult(
        *a, aStride,
        *b, bStride,
        dst1, stride,
        transposeA, transposeB, hand,
        batchCount, GPUConst<T>::get(1), GPUConst<T>::get(0)
    );
}

template<typename T>
void KroneckerTriplet<T>::multRows(const Mat<T> &other, const Mat<T> result, bool transposeThis, Handle &hand) {
    // mult1D(mat.x, transposeThis, true, other.layerRowCol(0), result.layerRowCol(0), other._rows, hand, other.layers);
    other.mult(mat.x, &result, &hand, false, !transposeThis);
}

template<typename T>
void KroneckerTriplet<T>::multCols(const Tensor<T> &other, Tensor<T> result, bool transposeThis, Handle &hand) {
    mult1D(mat.y, transposeThis, false, other.layerRowCol(0), result.layerRowCol(0), other._rows, hand, other.layers);
}

template<typename T>
void KroneckerTriplet<T>::multDepths(const Tensor<T> &other, Tensor<T> result, bool transposeThis, Handle &hand) {
    mult1D(mat.z, transposeThis, true, other.layerColDepth(0), result.layerColDepth(0), other.layers * other.rows, hand, other.cols);
}

template<typename T>
KroneckerTriplet<T>::KroneckerTriplet(const XYZ<Mat<T>> &mat): mat(mat) {
}

template<typename T>
void KroneckerTriplet<T>::product(Mat<T> &result, Mat<T>& yDimMultZDimBuffer, Handle &hand) {
    mat.y.multKronecker(mat.z, yDimMultZDimBuffer, hand);
    yDimMultZDimBuffer.multKronecker(mat.x, result, hand);
}

template<typename T>
Mat<T> KroneckerTriplet<T>::product(Handle &hand) {
    size_t yzRows = mat.y._rows * mat.z._rows, yzCols = mat.y._cols * mat.z._cols;
    auto result = Mat<T>::create(mat.x._rows * yzRows, mat.x._cols * yzCols);
    auto buffer = Mat<T>::create(yzRows, yzCols);
    product(result, buffer, hand);
    return result;
}

template<typename T>
GridDim KroneckerTriplet<T>::dim() {
    return GridDim(mat.y.cols, mat.x._cols, mat.z._cols);
}
//yzx
template<typename T>
void KroneckerTriplet<T>::mult(const SimpleArray<T>& other, SimpleArray<T>& result, bool transposeThis, SimpleArray<T> resultSizeBuffer, Handle &hand) {

    GridDim dim = this->dim();

    Tensor<T> otherTensor = other.tensor(dim.rows, dim.layers),
            resultTensor = result.tensor(dim.rows, dim.layers);
    Mat<T> otherMat = other.matrix(dim.rows), resultMat = result.matrix(dim.rows);
    multRows(otherMat, resultMat, transposeThis, hand);
    multDepths(resultTensor, resultSizeBuffer, transposeThis, hand);
    multCols(resultSizeBuffer, resultTensor, transposeThis, hand);
}

template<typename T>
void KroneckerTriplet<T>::mult(Mat<T> other, Mat<T> result, bool transposeThis, SimpleArray<T> resultHeightBuffer, Handle &hand) {
    for (size_t colInd = 0; colInd < other._cols; colInd++) {
        mult(other.col(colInd), result.col(colInd), transposeThis, resultHeightBuffer, hand);
    }
}
