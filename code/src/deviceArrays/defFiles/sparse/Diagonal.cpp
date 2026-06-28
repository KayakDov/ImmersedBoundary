//
// Created by usr on 6/18/26.
//

#include "../../headers/sparse/Diagonal.h"
#include "deviceArrays/headers/Support/Streamable.h"
#include "deviceArrays/headers/sparse/BandedMat.h"
#include "deviceArrays/headers/Singleton.h"


template<typename T>
const Singleton<int32_t>& Diagonal<T>::getSharedIndices(Handle &hand) {
    // Magic static ensures safe thread initialization once per compilation target
    static const Singleton<int32_t> sharedIndices = [] (Handle &h) {
        return Singleton<int32_t>::create(0, h);
    }(hand);

    return sharedIndices;
}

template<typename T>
Diagonal<T>::Diagonal(size_t denseSqMatDim, Handle &hand)
    : BandedMat<T>(
        Mat<T>::create(denseSqMatDim, 1),
        getSharedIndices(hand)
    ) {
}

template<typename T>
Diagonal<T>::Diagonal(const Mat<T> &windowInto, Handle &hand)
    : BandedMat<T>(windowInto, getSharedIndices(hand)) {
    if (windowInto._cols != 1 && windowInto._rows != 0)
        throw std::invalid_argument("Cannot construct TriDiagonal: Matrix payload must contain exactly 3 columns.");
}

template<typename T>
Diagonal<T>::Diagonal(const SimpleArray<T> &windowInto, Handle &hand): Diagonal(windowInto.matrix(windowInto.size()), hand) {
}

// Mirroring your system's explicit instantiations
template class Diagonal<float>;
template class Diagonal<double>;
template class Diagonal<size_t>;
template class Diagonal<int32_t>;
template class Diagonal<unsigned char>;