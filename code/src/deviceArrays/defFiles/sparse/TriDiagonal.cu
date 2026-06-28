//
// Created by usr on 06/11/26.
//

#include "../../headers/sparse/TriDiagonal.cuh"

#include "deviceArrays/headers/Support/Streamable.h"

template<typename T>
const Vec<int32_t>& TriDiagonal<T>::getSharedIndices(Handle &hand) {
    // Magic static ensures safe thread initialization once per compilation target
    static const Vec<int32_t> sharedIndices = [] (Handle &h) {
        auto inds = Vec<int32_t>::create(3, h);
        std::vector<AdjacencyInd> adjacencys = {TriDiagonal<T>::primary, TriDiagonal<T>::prevNext.left, TriDiagonal<T>::prevNext.right};
        AdjacencyPatern::loadMapRowToDiag(inds, adjacencys, h);
        return inds;
    }(hand);

    return sharedIndices;
}

template<typename T>
TriDiagonal<T>::TriDiagonal(size_t denseSqMatDim, Handle &hand)
    : BandedMat<T>(Mat<T>::create(denseSqMatDim, 3), getSharedIndices(hand)) {
}

template<typename T>
TriDiagonal<T>::TriDiagonal(const Mat<T> &copyFrom, Handle &hand)
    : BandedMat<T>(copyFrom, getSharedIndices(hand)) {
    if (copyFrom._cols != 3 && copyFrom._rows != 0)
        throw std::invalid_argument("Cannot construct TriDiagonal: Matrix payload must contain exactly 3 columns.");

}

// Mirroring your system's explicit instantiations
template class TriDiagonal<float>;
template class TriDiagonal<double>;
template class TriDiagonal<size_t>;
template class TriDiagonal<int32_t>;
template class TriDiagonal<unsigned char>;