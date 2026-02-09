//
// Created by usr on 2/9/26.
//

#ifndef CUDABANDED_ADITHOMAS_CUH
#define CUDABANDED_ADITHOMAS_CUH
#include "SimpleArray.h"
#include "Thomas.cuh"
#include "Support/GridDim.hpp"

template <typename Real>
class ADIThomas {
    const GridDim dim;
    SimpleArray<Real> r;
    const size_t maxIterations;
    Thomas<Real> thomas;
    Singleton<Real> residual;
    Real tolerance;

public:

    void solve(SimpleArray<Real> &x, const SimpleArray<Real> &b, Handle &hand);
};



#endif //CUDABANDED_ADITHOMAS_CUH
