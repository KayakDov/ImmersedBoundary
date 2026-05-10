#include "EigenDecompSolver.h"

#include "../Event.h"
#include "deviceArrays/headers/Support/Streamable.h"


template<typename T>
void EigenDecompSolver<T>::set0Avg(const Vec<T>& src, Vec<T>& dst, Vec<T>& bufferSizeOfB, Handle &hand) const {

    bufferSizeOfB.fill(1, hand);

    Singleton<T> sum = dst.get(0);
    src.mult(bufferSizeOfB, sum, &hand); //dst[0] = sum(src), bufferSizeOfB is not used again for its ones, and it's now safe to write there.

    Singleton<T> negInvSize = bufferSizeOfB.get(0);
    negInvSize.set(-1.0/src.size(), hand); //bufferSizeOfB[0] = -1/N

    negInvSize.mult(sum, &hand);//bufferSizeOfB[0] = -dst[0]/N, dts[0] is not used again and it's now safe to write to dst.

    src.add(static_cast<const Singleton<T>&>(negInvSize), dst, hand);
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

