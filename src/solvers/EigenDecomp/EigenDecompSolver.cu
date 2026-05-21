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

