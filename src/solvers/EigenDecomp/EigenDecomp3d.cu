
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"
#include "math/XYZ.cuh"
#include "deviceArrays/headers/Support/Streamable.h"

template<typename T>
__global__ void setLEigenValInverseKernel3d(
    DeviceData3d<T> dst,
    const XYZ<DeviceData1d<T>> eVals,
    const DeviceData3d<T> src,
    bool isSingular
) {
    if (GridInd3d ind; ind < dst) {

        bool den0 = isSingular && ind.layer == 0 && ind.row == 0 && ind.col == 0;

        dst[ind] = den0 ? 0 : src[ind] / (eVals.x[ind.col] + eVals.y[ind.row] + eVals.z[ind.layer]);
    }
}

template<typename T>
void EigenDecomp3d<T>::multLEigenValInverse(const SimpleArray<T> &src, SimpleArray<T> &dst, Handle &hand) const {

    auto srcTensor = src.tensor(this->dim.rows, this->dim.layers);

    KernelPrep kp = srcTensor.kernelPrep();
    setLEigenValInverseKernel3d<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        dst.tensor(this->dim.rows, this->dim.layers).toKernel3d(),
        {this->eigen.vals.x.toKernel1d(), this->eigen.vals.y.toKernel1d(), this->eigen.vals.z.toKernel1d()},
        srcTensor.toKernel3d(),
        this->isSingular
    );
}

template<typename T>
EigenDecomp3d<T>::EigenDecomp3d(
    const Eigen<T>& eigen,
    SimpleArray<T> sizeOfB,
    bool isSingular
) : EigenDecompSolver<T>(eigen, sizeOfB, isSingular) {
}

template<typename T>
template<typename BoundaryConfigT>
EigenDecomp3d<T>::EigenDecomp3d(BoundaryConfigT boundary, Handle* hand3, Event* event2) :
    EigenDecompSolver<T>(boundary, hand3, event2) {

}

template<typename T>
template<typename BoundaryConfigT>
EigenDecomp3d<T>::EigenDecomp3d(BoundaryConfigT boundary, Handle *hand3, Event *event2, SimpleArray<T> sizeOfB) :
    EigenDecompSolver<T>(boundary, hand3, event2, sizeOfB){
}

template<typename T>
void EigenDecomp3d<T>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {

    this->eigen.vecs.mult( b , x, true, this->sizeOfB, hand);


    this->multLEigenValInverse(x, this->sizeOfB, hand);


    this->eigen.vecs.mult(this->sizeOfB, x, false, this->sizeOfB, hand);
}

template class EigenDecomp3d<double>;
template class EigenDecomp3d<float>;