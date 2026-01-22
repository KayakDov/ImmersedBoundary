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
    const auto& basedata = out.data;

    os << "BaseData Debug Output\n";
    os << "GridDim: " << basedata.dim << std::endl;

    os << "Delta: " << basedata.delta << std::endl;

    os << "f: " << GpuOut<Real>(basedata.f, out.handle) << std::endl;

    os << "p: " <<  GpuOut<Real>(basedata.p, out.handle) << std::endl;

    os << "maxB: " << SparseCSCOut<Real, Int>(basedata.maxB, out.handle) << std::endl;

    if (basedata.B) os << "B: " << SparseCSCOut<Real, Int>(*basedata.B, out.handle) << std::endl;
    else os << "B is not initialized." << std::endl;

    return os;
}

#endif //CUDABANDED_BASEDATAOUT_H