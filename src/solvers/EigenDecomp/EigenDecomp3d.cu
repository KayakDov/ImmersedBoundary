
#include "EigenDecomp3d.cuh"



template<typename T>
__global__ void setUTildeKernel3d(DeviceData3d<T> uTilde,
      const DeviceData1d<T> eValsX,
      const DeviceData1d<T> eValsY,
      const DeviceData1d<T> eValsZ,
      const DeviceData3d<T> fTilde) {
    if (GridInd3d ind; ind < uTilde)
        uTilde[ind] = fTilde[ind] / (eValsX[ind.col] + eValsY[ind.row] + eValsZ[ind.layer]);
}

template<typename T>
void EigenDecomp3d<T>::setUTilde(const Tensor<T> &f, Tensor<T> &u, Handle &hand) const {
    KernelPrep kp = f.kernelPrep();
    setUTildeKernel3d<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        u.toKernel3d(),
        this->eigen.vals.x.toKernel1d(),
        this->eigen.vals.y.toKernel1d(),
        this->eigen.vals.z.toKernel1d(),
        f.toKernel3d());
}

template<typename T>
void EigenDecomp3d<T>::multE(
    size_t i,
    bool transposeEigen,
    bool transposeOperand,
    const Mat<T> &operand1,
    Mat<T> &dst1,
    size_t stride,
    Handle &hand,
    size_t batchCount
) const {

    auto eigenMat = this->eigen.vecs[i];

    const Mat<T> *a, *b;
    size_t aStride, bStride;
    bool transposeA, transposeB;

    if (transposeOperand) {
        a = &operand1;
        aStride = stride;
        b = &eigenMat;
        bStride = 0;
        transposeA = false;
        transposeB = transposeEigen;
    }else {
        a = &eigenMat;
        aStride = 0;
        b = &operand1;
        bStride = stride;
        transposeA = transposeEigen;
        transposeB = false;
    }

    Mat<T>::batchMult( *a, aStride, *b, bStride, dst1, stride, transposeA, transposeB, hand, batchCount, GPUConst<T>::get(1), GPUConst<T>::get(0));
}

template<typename T>
void EigenDecomp3d<T>::multEX(const Mat<T> &src1, Mat<T> dst1, Handle &hand, bool transposeE) const {
    multE(0, transposeE, true, src1, dst1, src1._rows, hand, this->dim.layers);
}
template<typename T>
void EigenDecomp3d<T>::multEY(const Mat<T> &src1, Mat<T> dst1, Handle &hand, bool transposeE) const {
    multE(1, transposeE, false, src1, dst1, src1._rows, hand, this->dim.layers);
}

template<typename T>
void EigenDecomp3d<T>::multEZ(const Mat<T> &src1, Mat<T> dst1, Handle &hand, bool transposeE) const {
    multE(2, transposeE, true, src1, dst1, this->dim.layers * this->dim.rows, hand, this->dim.cols);
}

template<typename T>
void EigenDecomp3d<T>::multiplyEF(Handle &hand, const Tensor<T> &src, const Tensor<T>& dst, bool transposeE) const {

    multEY( src.layerRowCol(0), dst.layerRowCol(0), hand, transposeE);
    multEZ(dst.layerColDepth(0),  src.layerColDepth(0), hand, transposeE);//TODO: modify this so that src is not changed.  Can be done by passing another temp variable, if that variable is the same as rhs, then rhs will be overwritten, if it's different, tahn rhs will not be overwritten.  If this is done, then it may be possible the remove sizeOfB as a field.
    multEX(src.layerRowCol(0), dst.layerRowCol(0), hand, transposeE);
}

template<typename T>
EigenDecomp3d<T>::EigenDecomp3d(
    const poisson::Eigen<T>& eigen,
    SimpleArray<T> sizeOfB
) : EigenDecompSolver<T>(eigen, sizeOfB) {
}

template<typename T>
EigenDecomp3d<T>::EigenDecomp3d(
    BoundaryConfig<T> boundary,
    Handle* hand3,
    Event* event2
) : EigenDecompSolver<T>(boundary, hand3, event2) {
}

template<typename T>
EigenDecomp3d<T>::EigenDecomp3d(BoundaryConfig<T> boundary, Handle *hand3, Event *event2, SimpleArray<T> sizeOfB):
    EigenDecompSolver<T>(boundary, hand3, event2, sizeOfB){
}

template<typename T>
void EigenDecomp3d<T>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {
    auto bT = b.tensor(this->dim.rows, this->dim.layers);
    auto temp = this->sizeOfB.tensor(this->dim.rows, this->dim.layers);
    auto xT = x.tensor(this->dim.rows, this->dim.layers);

    this->multiplyEF(hand, bT, xT, true);

    this->setUTilde(xT, temp, hand);

    this->multiplyEF(hand, temp, xT, false);
}

template class EigenDecomp3d<double>;
template class EigenDecomp3d<float>;