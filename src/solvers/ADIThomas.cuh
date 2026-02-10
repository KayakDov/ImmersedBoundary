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
    const size_t maxIterations;
    Real tolerance;

    SimpleArray<Real> r;
    Mat<Real> thomasSratch;
    Thomas<Real> thomasRows, thomasCols, thomasDepths;
    Singleton<Real> rNorm, bNorm;


public:
    ADIThomas(const GridDim &dim, size_t max_iterations, const Real &tolerance);
    /**
     * @brief Solves the system $Ax = b$ using the ADI iterative process.
     * * The iteration continues until the $L_2$ norm of the residual $r = b - Lx$
     * relative to the $L_2$ norm of $b$ is less than the tolerance.
     * * @param x [in/out] Solution array (initial guess provided as input).
     * @param b [in] Source term array.
     * @param hand CUDA Handle for stream management.
     */
    void solve(SimpleArray<Real> &x, const SimpleArray<Real> &b, Handle &hand);
};



#endif //CUDABANDED_ADITHOMAS_CUH
