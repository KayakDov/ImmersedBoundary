
#include "EigenDecomp3d.cuh"

#include "EigenDecompThomas.cuh"


template<typename T>
__global__ void setUTildeKernel3d(DeviceData3d<T> uTilde,
      const XYZ<DeviceData1d<T>> eVals,
      const DeviceData3d<T> fTilde,
      bool isSingular) {
    if (GridInd3d ind; ind < uTilde) {

        bool den0 = isSingular && ind.layer == 0 && ind.row == 0 && ind.col == 0;

        uTilde[ind] = den0 ? 0 : fTilde[ind] / (eVals.x[ind.col] + eVals.y[ind.row] + eVals.z[ind.layer]);
    }
}

template<typename T>
void EigenDecomp3d<T>::setUTilde(const SimpleArray<T> &f, SimpleArray<T> &u, Handle &hand) const {
    auto uTensor = u.tensor(this->dim.rows, this->dim.layers);
    KernelPrep kp = uTensor.kernelPrep();
    setUTildeKernel3d<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        uTensor.toKernel3d(),
        {this->eigen.vals.x.toKernel1d(), this->eigen.vals.y.toKernel1d(), this->eigen.vals.z.toKernel1d()},
        f.tensor(this->dim.rows, this->dim.layers).toKernel3d(),
        this->isSingular);
}

template<typename T>
EigenDecomp3d<T>::EigenDecomp3d(
    const poisson::Eigen<T>& eigen,
    SimpleArray<T> sizeOfB,
    bool isSingular
) : EigenDecompSolver<T>(eigen, sizeOfB, isSingular) {
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

    this->eigen.vecs.mult(b, x, true, this->sizeOfB, hand);

    this->setUTilde(x, this->sizeOfB, hand);

    this->eigen.vecs.mult(this->sizeOfB, x, false, this->sizeOfB, hand);

}

template class EigenDecomp3d<double>;
template class EigenDecomp3d<float>;