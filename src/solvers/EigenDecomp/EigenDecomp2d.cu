

#include "EigenDecomp2d.h"
#include "poisson/BoundaryConfig.cuh"

template<typename T>
__global__ void eValsLInvMultKernel(DeviceData2d<T> dst,
                                  const DeviceData1d<T> eValsX,
                                  const DeviceData1d<T> eValsY,
                                  const DeviceData2d<T> src,
                                  bool isSingular,
                                  T helmholtzShift
                                  ) {
    if (GridInd2d ind; ind < dst) {

        bool den0 = isSingular && ind.col == 0 && ind.row == 0;

        dst[ind] = den0 ? 0 :src[ind] / (eValsX[ind.col] + eValsY[ind.row] - helmholtzShift);
    }
}


template<typename T>
void EigenDecomp2d<T>::eValsLInvMult(const SimpleArray<T> &src, SimpleArray<T> &dst, Handle &hand) const {
    auto srcMat = src.matrix(this->dim.rows);
    KernelPrep kp = srcMat.kernelPrep();
    eValsLInvMultKernel<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        dst.matrix(this->dim.rows).toKernel2d(),
        this->lapEigen.vals.x.toKernel1d(),
        this->lapEigen.vals.y.toKernel1d(),
        srcMat.toKernel2d(),
        this->isSingular,
        this->helmholtzShift
    );
}

template<typename T>
template<typename BoundaryConfigT>
EigenDecomp2d<T>::EigenDecomp2d(const BoundaryConfigT& boundary, Handle* hand2, Event& event, T helmholtzShift) :
    EigenDecompSolver<T>(boundary, hand2, &event, helmholtzShift) {
}

template<typename T>
void EigenDecomp2d<T>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {

    this->lapEigen.vecsInv.mult2d(b, this->sizeOfB, x, hand);

    eValsLInvMult(this->sizeOfB, x, hand);

    this->lapEigen.vecs.mult2d(x, x, this->sizeOfB, hand);
}

// Instantiate the class for both float and double
template class EigenDecomp2d<double>;
template class EigenDecomp2d<float>;

// Macro for the 2D constructor
#define INSTANTIATE_EIGEN_DECOMP_2D_CONSTRUCTORS(Real, SegX, SegY, SegZ) \
template EigenDecomp2d<Real>::EigenDecomp2d(                         \
const BoundaryConfig<Real, SegX, SegY, SegZ>& boundary,          \
Handle* hands,                                                   \
Event& event,                                                    \
Real helmholtzShift                                              \
);

// Apply permutations
APPLY_TO_ALL_SEGMENT_COMBOS(double, INSTANTIATE_EIGEN_DECOMP_2D_CONSTRUCTORS)
APPLY_TO_ALL_SEGMENT_COMBOS(float, INSTANTIATE_EIGEN_DECOMP_2D_CONSTRUCTORS)