

#include "EigenDecomp2d.h"

template<typename T>
__global__ void eValsLInvMultKernel(DeviceData2d<T> dst,
                                  const DeviceData1d<T> eValsX,
                                  const DeviceData1d<T> eValsY,
                                  const DeviceData2d<T> src,
                                  bool isSingular
                                  ) {
    if (GridInd2d ind; ind < dst) {

        bool den0 = isSingular && ind.col == 0 && ind.row == 0;

        dst[ind] = den0 ? 0 :src[ind] / (eValsX[ind.col] + eValsY[ind.row]);
    }
}


template<typename T>
void EigenDecomp2d<T>::eValsLInvMult(const Mat<T> &src, Mat<T> &dst, Handle &hand) const {
    KernelPrep kp = src.kernelPrep();
    eValsLInvMultKernel<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        dst.toKernel2d(),
        this->eigen.vals.x.toKernel1d(),
        this->eigen.vals.y.toKernel1d(),
        src.toKernel2d(),
        this->isSingular
    );
}

template<typename T>
EigenDecomp2d<T>::EigenDecomp2d(const BoundaryConfig<T>& boundary, Handle* hand2, Event& event) :
    EigenDecompSolver<T>(boundary, hand2, &event) {
}

template<typename T>
void EigenDecomp2d<T>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {

    const auto bM = b.matrix(this->dim.rows);
    auto temp = this->sizeOfB.matrix(this->dim.rows);
    auto xM = x.matrix(this->dim.rows);

    this->eigen.vecs.y.mult(bM, &xM, &hand, true, false);
    xM.mult(this->eigen.vecs.x, &temp, &hand, false, false);

    eValsLInvMult(temp, xM, hand);

    this->eigen.vecs.y.mult(xM, &temp, &hand, false, false);
    temp.mult(this->eigen.vecs.x, &xM, &hand, false, true);
}

template class EigenDecomp2d<double>;
template class EigenDecomp2d<float>;