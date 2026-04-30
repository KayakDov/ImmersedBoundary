#include "KroneckerTriplet.h"


// template<typename T>
// void KroneckerTriplet<T>::mult1D(
//     const Mat<T>& kMat,
//     bool transposeThis, bool transposeOperand,
//     const Mat<T> &operand1, Mat<T> &dst1,
//     size_t stride,
//     Handle &hand,
//     size_t batchCount
// ) const {
//
//     const Mat<T> *a, *b;
//     size_t aStride, bStride;
//     bool transposeA, transposeB;
//
//     if (transposeOperand) {
//         a = &operand1;
//         aStride = stride;
//         b = &kMat;
//         bStride = 0;
//         transposeA = false;
//         transposeB = transposeThis;
//     }else {
//         a = &kMat;
//         aStride = 0;
//         b = &operand1;
//         bStride = stride;
//         transposeA = transposeThis;
//         transposeB = false;
//     }
//
//     Mat<T>::batchMult(
//         *a, aStride,
//         *b, bStride,
//         dst1, stride,
//         transposeA, transposeB, hand,
//         batchCount, GPUConst<T>::get(1), GPUConst<T>::get(0)
//     );
// }

template<typename T>
void KroneckerTriplet<T>::multRows(const SimpleArray<T> &other, SimpleArray<T> result, bool transposeThis, Handle &hand) {
    Mat<T> otherMat = other.matrix(dim.rows * dim.layers), resultMat = result.matrix(dim.rows * dim.layers);
    otherMat.mult(this->x, &resultMat, &hand, false, !transposeThis);
}

template<typename T>
void KroneckerTriplet<T>::multCols(const SimpleArray<T> &other, SimpleArray<T> result, bool transposeThis, Handle &hand) {
    Mat<T> otherMat = other.matrix(dim.rows), resultMat = result.matrix(dim.rows);
    this->y.mult(otherMat, &resultMat, &hand, transposeThis, false);
}

template<typename T>
void KroneckerTriplet<T>::multDepths(const SimpleArray<T> &other, SimpleArray<T> result, bool transposeThis, Handle &hand) {
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
KroneckerTriplet<T>::KroneckerTriplet(const XYZ<Mat<T>> &mat): XYZ<Mat<T>>(mat), dim(mat.y._cols, mat.x._cols, mat.z._cols) {
    std::cout << dim << std::endl;
}

template<typename T>
KroneckerTriplet<T>::KroneckerTriplet(const Mat<T> x, const Mat<T> y, const Mat<T> z): XYZ<Mat<T>>(x, y, z), dim(y._cols, x._cols, z._cols) {
}

template<typename T>
void KroneckerTriplet<T>::product(Mat<T> &result, Mat<T>& xDimMultZDimBuffer, Handle &hand) {
    this->x.multKronecker(this->z, xDimMultZDimBuffer, hand);
    xDimMultZDimBuffer.multKronecker(this->y, result, hand);
}

template<typename T>
Mat<T> KroneckerTriplet<T>::product(Handle &hand) {
    size_t yzRows = this->y._rows * this->z._rows, yzCols = this->y._cols * this->z._cols;
    auto result = Mat<T>::create(this->x._rows * yzRows, this->x._cols * yzCols);
    auto buffer = Mat<T>::create(yzRows, yzCols);
    product(result, buffer, hand);
    return result;
}

template<typename T>
void KroneckerTriplet<T>::mult(const SimpleArray<T>& other, SimpleArray<T>& result, bool transposeThis, Handle &hand) {
    auto buffer = SimpleArray<T>::create(result.size(), hand);
    mult(other, result, transposeThis, buffer, hand);
}

template<typename T>
void KroneckerTriplet<T>::mult(const SimpleArray<T>& other, SimpleArray<T>& result, bool transposeThis, SimpleArray<T> resultSizeBuffer, Handle &hand) {

    multCols(other, result, transposeThis, hand);
    multDepths(result, resultSizeBuffer, transposeThis, hand);
    multRows(resultSizeBuffer, result, transposeThis, hand);

}

template<typename T>
void KroneckerTriplet<T>::mult(const Mat<T>& other, Mat<T>& result, bool transposeThis, Handle &hand) {
    auto buffer = SimpleArray<T>::create(result._rows, hand);
    mult(other, result, transposeThis, buffer, hand);
}

template<typename T>
void KroneckerTriplet<T>::mult(const Mat<T>& other, Mat<T>& result, bool transposeThis, SimpleArray<T>& resultHeightBuffer, Handle &hand) {
    for (size_t colInd = 0; colInd < other._cols; colInd++) {
        SimpleArray<T> resultCol = result.col(colInd);
        mult(other.col(colInd), resultCol, transposeThis, resultHeightBuffer, hand);
    }
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::xOperator(const GridDim& gridDim, const Mat<T>& forRows) {
    auto Iy = SquareMat<T>::create(gridDim.rows);
    auto Iz = SquareMat<T>::create(gridDim.layers);
    return {forRows, Iy, Iz};
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::yOperator(const GridDim& gridDim, const Mat<T>& forCols) {
    auto Ix = SquareMat<T>::create(gridDim.cols);
    auto Iz = SquareMat<T>::create(gridDim.layers);
    return {Ix, forCols, Iz};
}

template<typename T>
KroneckerTriplet<T> KroneckerTriplet<T>::zOperator(const GridDim& gridDim, const Mat<T>& forLayers) {
    auto Ix = SquareMat<T>::create(gridDim.cols);
    auto Iy = SquareMat<T>::create(gridDim.rows);
    return {Ix, Iy, forLayers};
}

template class KroneckerTriplet<float>;
template class KroneckerTriplet<double>;