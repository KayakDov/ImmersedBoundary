

#include "EigenDecomp2d.h"

template<typename T>
__global__ void setUTildeKernel2d(DeviceData2d<T> uTilde,
                                  const DeviceData1d<T> eValsX,
                                  const DeviceData1d<T> eValsY,
                                  const DeviceData2d<T> fTilde) {
    if (GridInd2d ind; ind < uTilde)
        uTilde[ind] = fTilde[ind] / (eValsX[ind.col] + eValsY[ind.row]);
}


template<typename T>
void EigenDecomp2d<T>::setUTilde(const Mat<T> &f, Mat<T> &u, Handle &hand) const {
    KernelPrep kp = f.kernelPrep();
    setUTildeKernel2d<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        u.toKernel2d(),
        this->eVals[0].toKernel1d(),
        this->eVals[1].toKernel1d(),
        f.toKernel2d());
}

template<typename T>
void EigenDecomp2d<T>::setEigens(Handle* hand2, const Real2d delta, Event& event){
    this->eigenL(1, delta, hand2[1]);
    event.record(hand2[1]);

    this->eigenL(0, delta, hand2[0]);
    event.hold(hand2[0]);
}

template<typename T>
EigenDecomp2d<T>::EigenDecomp2d(SquareMat<T> &rowsXRowsP1, SquareMat<T> &colsXColsP1, SimpleArray<T> &sizeOfB, Handle* hand2, const Real2d delta, Event& event) :
    EigenDecompSolver<T>({colsXColsP1, rowsXRowsP1}, sizeOfB) {
    setEigens(hand2, delta, event);
}

template<typename T>
EigenDecomp2d<T>::EigenDecomp2d(GridDim dim, Handle* hand2, const Real2d delta, Event& event) :
    EigenDecompSolver<T>(dim, delta, hand2[0]) {
    setEigens(hand2, delta, event);
}

template<typename T>
void EigenDecomp2d<T>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {

    const auto bM = b.matrix(this->dim.rows);
    auto soBM = this->sizeOfB.matrix(this->dim.rows);
    auto xM = x.matrix(this->dim.rows);

    bM.mult(this->eVecs[0], &xM, &hand, false, false);
    this->eVecs[1].mult(xM, &soBM, &hand, true, false);

    setUTilde(soBM, xM, hand);

    this->eVecs[1].mult(xM, &soBM, &hand, false, false);
    soBM.mult(this->eVecs[0], &xM, &hand, false, true);
}

template class EigenDecomp2d<double>;
template class EigenDecomp2d<float>;