

#include "EigenDecomp2d.h"

template<typename T>
__global__ void eValsLInvMultKernel(DeviceData2d<T> dst,
                                  const DeviceData1d<T> eValsX,
                                  const DeviceData1d<T> eValsY,
                                  const DeviceData2d<T> src,
                                  T tolerance
                                  ) {
    if (GridInd2d ind; ind < dst) {
        T den = eValsX[ind.col] + eValsY[ind.row];
        bool denNot0 = abs(den) > tolerance;
        dst[ind] = denNot0 ? src[ind] / den : 0;
    }
}


template<typename T>
void EigenDecomp2d<T>::eValsLInvMult(const Mat<T> &src, Mat<T> &dst, T tolerance, Handle &hand) const {
    KernelPrep kp = src.kernelPrep();
    eValsLInvMultKernel<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        dst.toKernel2d(),
        this->eigen.vals.x.toKernel1d(),
        this->eigen.vals.y.toKernel1d(),
        src.toKernel2d(),
        tolerance
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