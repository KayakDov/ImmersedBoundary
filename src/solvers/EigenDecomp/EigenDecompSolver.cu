#include "EigenDecompSolver.h"

#include "../Event.h"




template<typename T>
EigenDecompSolver<T>::EigenDecompSolver(const LaplacianEigen<T>& eMatsAndVecs, SimpleArray<T> &sizeOfB) :
    dim(
        eMatsAndVecs.vecs.y._rows,
        eMatsAndVecs.vecs.x._rows,
        eMatsAndVecs.vecs.y._rows
    ),
    eigen(eMatsAndVecs),
    sizeOfB(sizeOfB) {
}

template<typename T>
EigenDecompSolver<T>::EigenDecompSolver(const BoundaryConfig<T>& boundary, Handle* hands, Event* events, SimpleArray<T> sizeOfB):
    EigenDecompSolver(
        LaplacianEigen<T>::make(boundary, hands, events),
        sizeOfB
    ){
}

template<typename T>
EigenDecompSolver<T>::EigenDecompSolver(const BoundaryConfig<T>& boundary, Handle* hands, Event* events):
    EigenDecompSolver(boundary, hands, events, SimpleArray<T>::create(dim.size(), hands[0]))
{}

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

