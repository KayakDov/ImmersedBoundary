

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
void EigenDecomp2d<T>::eValsLInvMult(const SimpleArray<T> &src, SimpleArray<T> &dst, Handle &hand) const {
    auto srcMat = src.matrix(this->dim.rows);
    KernelPrep kp = srcMat.kernelPrep();
    eValsLInvMultKernel<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        dst.matrix(this->dim.rows).toKernel2d(),
        this->eigen.vals.x.toKernel1d(),
        this->eigen.vals.y.toKernel1d(),
        srcMat.toKernel2d(),
        this->isSingular
    );
}

template<typename T>
template<typename BoundaryConfigT>
EigenDecomp2d<T>::EigenDecomp2d(const BoundaryConfigT& boundary, Handle* hand2, Event& event) :
    EigenDecompSolver<T>(boundary, hand2, &event) {
}

template<typename T>
void EigenDecomp2d<T>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {

    this->eigen.vecsInv.mult2d(b, this->sizeOfB, x, hand);

    eValsLInvMult(this->sizeOfB, x, hand);

    this->eigen.vecs.mult2d(x, x, this->sizeOfB, hand);
}

template class EigenDecomp2d<double>;
template class EigenDecomp2d<float>;

#define INSTANTIATE_EIGEN2D_BOUNDARY(Real, SegX, SegY, SegZ) \
template EigenDecomp2d<Real>::EigenDecomp2d( \
const BoundaryConfig<Real, SegX, SegY, SegZ>&, Handle*, Event&);

#define INSTANTIATE_EIGEN2D_ALL(Real) \
INSTANTIATE_EIGEN2D_BOUNDARY(Real, UniformSegment<Real>,  UniformSegment<Real>,  UniformSegment<Real>)  \
INSTANTIATE_EIGEN2D_BOUNDARY(Real, UniformSegment<Real>,  UniformSegment<Real>,  VariableSegment<Real>) \
INSTANTIATE_EIGEN2D_BOUNDARY(Real, UniformSegment<Real>,  VariableSegment<Real>, UniformSegment<Real>)  \
INSTANTIATE_EIGEN2D_BOUNDARY(Real, UniformSegment<Real>,  VariableSegment<Real>, VariableSegment<Real>) \
INSTANTIATE_EIGEN2D_BOUNDARY(Real, VariableSegment<Real>, UniformSegment<Real>,  UniformSegment<Real>)  \
INSTANTIATE_EIGEN2D_BOUNDARY(Real, VariableSegment<Real>, UniformSegment<Real>,  VariableSegment<Real>) \
INSTANTIATE_EIGEN2D_BOUNDARY(Real, VariableSegment<Real>, VariableSegment<Real>, UniformSegment<Real>)  \
INSTANTIATE_EIGEN2D_BOUNDARY(Real, VariableSegment<Real>, VariableSegment<Real>, VariableSegment<Real>)

INSTANTIATE_EIGEN2D_ALL(float)
INSTANTIATE_EIGEN2D_ALL(double)