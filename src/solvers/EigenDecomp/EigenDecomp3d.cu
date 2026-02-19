
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
        this->eVals[0].toKernel1d(),
        this->eVals[1].toKernel1d(),
        this->eVals[2].toKernel1d(),
        f.toKernel3d());
}

template<typename T>
void EigenDecomp3d<T>::multE(size_t i,
                                   bool transposeEigen,
                                   bool transpose,
                                   const Mat<T> &a1,
                                   Mat<T> &dst1,
                                   size_t stride,
                                   Handle &hand,
                                   size_t batchCount
) const {
    Mat<T>::batchMult(
        transpose ? a1 : this->eVecs[i], transpose ? stride : 0,
        transpose ? this->eVecs[i] : a1, transpose ? 0 : stride,
        dst1, stride,
        transpose ? false : transposeEigen,
        transpose ? transposeEigen : false,
        hand, batchCount
    );
}

template<typename T>
void EigenDecomp3d<T>::multEX(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const {
    multE(0, transposeE, true, src, dst, src._rows, hand, this->dim.layers);
}
template<typename T>
void EigenDecomp3d<T>::multEY(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const {
    multE(1, transposeE, false, src, dst, src._rows, hand, this->dim.layers);
}

template<typename T>
void EigenDecomp3d<T>::multEZ(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const {
    multE(2, transposeE, true, src, dst, this->dim.layers * this->dim.rows, hand, this->dim.cols);
}

template<typename T>
void EigenDecomp3d<T>::multiplyEF(Handle &hand, const Tensor<T> &src, Tensor<T> &dst, bool transposeE) const {
    const auto xF = src.layerRowCol(0);
    auto dx1 = dst.layerRowCol(0);
    auto yF = dst.layerRowCol(0);
    auto sizeOfBT = this->sizeOfB.tensor(this->dim.rows, this->dim.layers);
    auto dyF = sizeOfBT.layerRowCol(0);
    auto zS = sizeOfBT.layerColDepth(0);
    auto dzS = dst.layerColDepth(0);

    if (transposeE) {
        multEX(xF, dx1, hand, transposeE);
        multEY(yF, dyF, hand, transposeE);
        multEZ(zS, dzS, hand, transposeE);
    } else {
        multEZ(zS, dzS, hand, transposeE);
        multEY(yF, dyF, hand, transposeE);
        multEX(xF, dx1, hand, transposeE);
    }
}

template<typename T>
void EigenDecomp3d<T>::setEigens(Handle* hand3, Real3d delta, Event* event3) {
    this->eigenL(0, delta, hand3[1]);
    event3[0].record(hand3[1]);

    this->eigenL(1, delta, hand3[2]);
    event3[1].record(hand3[2]);
    event3[1].hold(hand3[0]);

    this->eigenL(2, delta, hand3[0]);
    event3[0].hold(hand3[0]);
}

template<typename T>
EigenDecomp3d<T>::EigenDecomp3d(
    Mat<T> &rowsXRowsP1,
    Mat<T> &colsXColsP1,
    Mat<T> &depthsXDepthsP1,
    SimpleArray<T> sizeOfB,
    Handle* hand3,
    Real3d delta,
    Event* event
) : EigenDecompSolver<T>({colsXColsP1, rowsXRowsP1, depthsXDepthsP1}, sizeOfB) {
    setEigens(hand3, delta, event);
}

template<typename T>
EigenDecomp3d<T>::EigenDecomp3d(
    const GridDim& dim,
    Handle* hand3,
    const Real3d& delta,
    Event* event
) : EigenDecompSolver<T>(dim, delta, hand3[0]) {
    setEigens(hand3, delta, event);
}

template<typename T>
EigenDecomp3d<T>::EigenDecomp3d(const GridDim &dim, Handle *hand3, const Real3d &delta, SimpleArray<T> sizeOfB, Event *event):
    EigenDecompSolver<T>(dim, delta, sizeOfB){
    setEigens(hand3, delta, event);
}

template<typename T>
void EigenDecomp3d<T>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {
    const auto bT = b.tensor(this->dim.rows, this->dim.layers);
    auto bWorkSpaceT = this->sizeOfB.tensor(this->dim.rows, this->dim.layers);
    auto xT = x.tensor(this->dim.rows, this->dim.layers);

    this->multiplyEF(hand, bT, xT, true);

    this->setUTilde(xT, bWorkSpaceT, hand);

    this->multiplyEF(hand, bWorkSpaceT, xT, false);
}



template class EigenDecomp3d<double>;
template class EigenDecomp3d<float>;