//
// Created by usr on 1/21/26.
//

#ifndef CUDABANDED_BASEDATAOUT_H
#define CUDABANDED_BASEDATAOUT_H
#include "Streamable.h"
#include "BaseData.h"

template <typename Real, typename Int = uint32_t>
struct BaseDataOut {
    const BaseData<Real, Int>& data;
    Handle& handle;

    BaseDataOut(const BaseData<Real, Int>& d, Handle& h) : data(d), handle(h) {}
};


template <typename Real, typename Int>
std::ostream& operator<<(std::ostream& os, const BaseDataOut<Real, Int>& out) {
    const auto& data = out.data;

    os << "BaseData Debug Output\n";
    os << "GridDim: rows=" << data.dim.rows
       << ", cols=" << data.dim.cols
       << ", layers=" << data.dim.layers
       << ", layerSize=" << data.dim.layerSize << std::endl;

    os << "Delta: " << data.delta << std::endl;

    os << "f (SimpleArray):\n" << GpuOut<Real>(out.data.f, out.handle) << std::endl;

    os << "p (SimpleArray):\n" <<  GpuOut<Real>(out.data.p, out.handle) << std::endl;

    os << "maxB (SparseCSC) non-zero entries: " << data.maxB.nnz() << std::endl;

    if (data.B)
        os << "B: \n" << SparseCSCOut<Real, Int>(*out.data.B, out.handle) << std::endl;

    return os;
}

#endif //CUDABANDED_BASEDATAOUT_H