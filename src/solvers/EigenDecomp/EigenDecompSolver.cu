#include "EigenDecompSolver.h"

#include "../Event.h"


template<typename T>
void EigenDecompSolver<T>::set0Avg(const Vec<T>& src, Vec<T>& dst, Vec<T>& bufferSizeOfB, Handle &hand) const {
    bufferSizeOfB.fill(1, hand);

    Singleton<T> sum = dst.get(0);
    src.mult(bufferSizeOfB, sum, &hand);

    Singleton<T> negInvSize = bufferSizeOfB.get(0);
    negInvSize.set(-1.0/src.size(), hand);

    negInvSize.mult(sum, &hand);

    Singleton<T>& negAvg = negInvSize;

    dst.fill(negAvg, hand);//TODO: we would benefit here from multiple handles.;
    dst.add(src, &GPUConst<T>::get(1), &hand);
}

template<typename T>
EigenDecompSolver<T>::EigenDecompSolver(const poisson::Eigen<T>& eMatsAndVecs, SimpleArray<T> &sizeOfB, bool isSingular) :
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
EigenDecompSolver<T>::EigenDecompSolver(const BoundaryConfig<T>& boundary, Handle* hands, Event* events, SimpleArray<T> sizeOfB) :
    EigenDecompSolver(
        poisson::Eigen<T>::make(boundary, hands, events),
        sizeOfB,
        boundary.allNeumann()
    ){
}

template<typename T>
EigenDecompSolver<T>::EigenDecompSolver(const BoundaryConfig<T>& boundary, Handle* hands, Event* events):
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

