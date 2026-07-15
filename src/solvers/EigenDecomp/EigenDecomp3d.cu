
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"
#include "math/XYZ.cuh"
#include "deviceArrays/headers/Support/Streamable.h"
#include <fstream>
#include <iomanip>
#include <filesystem>


template<typename T>
__global__ void setLEigenValInverseKernel3d(
    DeviceData3d<T> dst,
    const XYZ<DeviceData1d<T>> eVals,
    const DeviceData3d<T> src,
    bool isSingular,
    T helmholtzShift //TODO: create a different version for when helmholtzShift = 0.
) {
    if (GridInd3d ind; ind < dst) {

        bool den0 = isSingular && ind.layer == 0 && ind.row == 0 && ind.col == 0;

        dst[ind] = den0 ? 0 : src[ind] / (eVals.x[ind.col] + eVals.y[ind.row] + eVals.z[ind.layer] - helmholtzShift);
    }
}

template<typename T>
void EigenDecomp3d<T>::multLEigenValInverse(const SimpleArray<T> &src, SimpleArray<T> &dst, Handle &hand) const {

    auto srcTensor = src.tensor(this->dim.rows, this->dim.layers);

    KernelPrep kp = srcTensor.kernelPrep();
    setLEigenValInverseKernel3d<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        dst.tensor(this->dim.rows, this->dim.layers).toKernel3d(),
        {this->lapEigen.vals.x.toKernel1d(), this->lapEigen.vals.y.toKernel1d(), this->lapEigen.vals.z.toKernel1d()},
        srcTensor.toKernel3d(),
        this->isSingular,
        this->helmholtzShift
    );
}

template<typename T>
EigenDecomp3d<T>::EigenDecomp3d(
    const Eigen<T>& eigen,
    SimpleArray<T> sizeOfB,
    bool isSingular,
    T helmholtzShift

) : EigenDecompSolver<T>(eigen, sizeOfB, isSingular, helmholtzShift) {
}

template<typename T>
template<typename BoundaryConfigT>
EigenDecomp3d<T>::EigenDecomp3d(const BoundaryConfigT boundary, Handle* hand3, Event* event2, T helmholtzShift) :
    EigenDecompSolver<T>(boundary, hand3, event2, helmholtzShift) {

}

template<typename T>
template<typename BoundaryConfigT>
EigenDecomp3d<T>::EigenDecomp3d(const BoundaryConfigT boundary, Handle *hand3, Event *event2, SimpleArray<T> sizeOfB, T helmholtzShift) :
    EigenDecompSolver<T>(boundary, hand3, event2, sizeOfB, helmholtzShift){
}

template<typename T>
void EigenDecomp3d<T>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {
    this->lapEigen.vecsInv.mult( b , x, this->sizeOfB, hand);

    this->multLEigenValInverse(x, this->sizeOfB, hand);

    this->lapEigen.vecs.mult(this->sizeOfB, x, this->sizeOfB, hand);
}


template class EigenDecomp3d<double>;
template class EigenDecomp3d<float>;

#define INSTANTIATE_EIGENDECOMP3D_CTORS(Real, SegX, SegY, SegZ) \
template EigenDecomp3d<Real>::EigenDecomp3d(const BoundaryConfig<Real, SegX, SegY, SegZ>, Handle*, Event*, Real); \
template EigenDecomp3d<Real>::EigenDecomp3d(const BoundaryConfig<Real, SegX, SegY, SegZ>, Handle*, Event*, SimpleArray<Real>, Real);

APPLY_TO_ALL_SEGMENT_COMBOS(double, INSTANTIATE_EIGENDECOMP3D_CTORS)
APPLY_TO_ALL_SEGMENT_COMBOS(float,  INSTANTIATE_EIGENDECOMP3D_CTORS)

