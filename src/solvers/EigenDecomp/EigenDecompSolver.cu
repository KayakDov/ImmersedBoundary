#include "EigenDecompSolver.h"

#include "../Event.h"
#include "deviceArrays/headers/Support/Streamable.h"


template<typename T>
bool EigenDecompSolver<T>::isInLColSpace(const Vec<T> &rhs, Vec<T> &bufferSizeOfB, Singleton<T> &bufferSing, double tolerance, Handle &hand) const {

    bufferSizeOfB.fill(1, hand);

    bufferSizeOfB.mult(rhs, bufferSing, &hand);

    T result = bufferSing.get(hand);

    return std::abs(result) < tolerance;
}



template<typename T>
EigenDecompSolver<T>::EigenDecompSolver(const Eigen<T>& eMatsAndVecs, SimpleArray<T> &sizeOfB, bool isSingular) :
    dim(
        eMatsAndVecs.vecs.y._rows,
        eMatsAndVecs.vecs.x._rows,
        eMatsAndVecs.vecs.z._rows
    ),
    eigen(eMatsAndVecs),
    sizeOfB(sizeOfB),
    isSingular(isSingular){

}

template<typename T>
template<typename BoundaryConfigT>
EigenDecompSolver<T>::EigenDecompSolver(const BoundaryConfigT& boundary, Handle* hands, Event* events, SimpleArray<T> sizeOfB) :
    EigenDecompSolver(
        Eigen<T>::make(boundary, hands, events),
        sizeOfB,
        boundary.allNeumann()
    ){
}

template<typename T>
template<typename BoundaryConfigT>
EigenDecompSolver<T>::EigenDecompSolver(const BoundaryConfigT& boundary, Handle* hands, Event* events):
    EigenDecompSolver(
        boundary,
        hands,
        events,
        SimpleArray<T>::create(boundary.dim().size(), hands[0])
    ) {}

template<typename T>
SquareMat<T> EigenDecompSolver<T>::inverseL(Handle &hand) const {
    auto id = SquareMat<T>::create(this->dim.size());
    id.setToIdentity(hand);
    auto result = SquareMat<T>::create(id._rows);
    for (size_t i = 0; i < result._rows; ++i) {
        auto src = id.col(i);
        auto dst = result.col(i);
        solve(dst, src, hand);
    }
    return result;
}

template class EigenDecompSolver<double>;
template class EigenDecompSolver<float>;


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
EigenDecomp3d<T>::EigenDecomp3d(const BoundaryConfigT& boundary, Handle* hand3, Event* event2) :
    EigenDecompSolver<T>(boundary, hand3, event2) {

}

template<typename T>
template<typename BoundaryConfigT>
EigenDecomp3d<T>::EigenDecomp3d(const BoundaryConfigT& boundary, Handle *hand3, Event *event2, SimpleArray<T> sizeOfB) :
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

#define INSTANTIATE_EIGEN_SOLVER_BOUNDARY(Real, SegX, SegY, SegZ) \
template EigenDecompSolver<Real>::EigenDecompSolver( \
const BoundaryConfig<Real, SegX, SegY, SegZ>&, Handle*, Event*, SimpleArray<Real>); \
template EigenDecompSolver<Real>::EigenDecompSolver( \
const BoundaryConfig<Real, SegX, SegY, SegZ>&, Handle*, Event*);

#define INSTANTIATE_EIGEN_SOLVER_ALL(Real) \
INSTANTIATE_EIGEN_SOLVER_BOUNDARY(Real, UniformSegment<Real>,  UniformSegment<Real>,  UniformSegment<Real>)  \
INSTANTIATE_EIGEN_SOLVER_BOUNDARY(Real, UniformSegment<Real>,  UniformSegment<Real>,  VariableSegment<Real>) \
INSTANTIATE_EIGEN_SOLVER_BOUNDARY(Real, UniformSegment<Real>,  VariableSegment<Real>, UniformSegment<Real>)  \
INSTANTIATE_EIGEN_SOLVER_BOUNDARY(Real, UniformSegment<Real>,  VariableSegment<Real>, VariableSegment<Real>) \
INSTANTIATE_EIGEN_SOLVER_BOUNDARY(Real, VariableSegment<Real>, UniformSegment<Real>,  UniformSegment<Real>)  \
INSTANTIATE_EIGEN_SOLVER_BOUNDARY(Real, VariableSegment<Real>, UniformSegment<Real>,  VariableSegment<Real>) \
INSTANTIATE_EIGEN_SOLVER_BOUNDARY(Real, VariableSegment<Real>, VariableSegment<Real>, UniformSegment<Real>)  \
INSTANTIATE_EIGEN_SOLVER_BOUNDARY(Real, VariableSegment<Real>, VariableSegment<Real>, VariableSegment<Real>)

INSTANTIATE_EIGEN_SOLVER_ALL(float)
INSTANTIATE_EIGEN_SOLVER_ALL(double)