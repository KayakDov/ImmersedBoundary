/**
 * @file ODE.cpp
 * @brief Allocation-free classical (4-stage) RK4 implementation for ODE.
 */

#include "ODE.h"
#include "deviceArrays/headers/Singleton.h"
#include "deviceArrays/headers/Vec.h"
#include "deviceArrays/headers/Mat.h"
#include <stdexcept>

template<typename Real>
ODE<Real>::ODE(Real stepSizeScalar, Handle& hand):
    scalars(SimpleArray<Real>::create(4, hand)),
    stepSizeOver2(scalars.get(0)),
    stepSizeOver3(scalars.get(1)),
    stepSizeOver6(scalars.get(2)),
    stepSize(scalars.get(3)),
    stepSizeScalar(stepSizeScalar)
{
    stepSizeOver2.set(stepSizeScalar/2, hand);
    stepSizeOver3.set(stepSizeScalar/3, hand);
    stepSizeOver6.set(stepSizeScalar/6, hand);
    stepSize.set(stepSizeScalar, hand);
    hand.synch();
}

template<typename Real>
void ODE<Real>::rungeKutta4(
    Real t,
    const Vec<Real>& x,
    Vec<Real> dst,
    Mat<Real> scratch,
    Handle& hand
) const {
    scratch.fill(0, hand);
    auto kSum = scratch.col(0);
    auto col1 = scratch.col(1);
    auto col2 = scratch.col(2);

    dxdt(t, x, col1, col2, scalars.get(0), hand);
    kSum.add(col1, &stepSizeOver6, &hand);

    dxdt(t + this->stepSizeScalar/2, x, col2, col1, stepSizeOver2, hand);

    kSum.add(col2, &stepSizeOver3, &hand);

    dxdt(t + this->stepSizeScalar/2, x, col1, col2, stepSizeOver2, hand);

    kSum.add(col1, &stepSizeOver3, &hand);

    dxdt(t + this->stepSizeScalar, x, col2, col1, stepSize, hand);

    kSum.add(col2, &stepSizeOver6, &hand);

    auto one = GPUScalar<Real>::get(1, hand);

    dst.setSum(kSum, x, one, one, &hand);
}


template class ODE<float>;
template class ODE<double>;
