/**
 * @file ODE.cu
 * @brief Allocation-free SSPRK(3,3) implementation for ODE.
 */

#include "ODE.h"
#include "deviceArrays/headers/Singleton.h"
#include "deviceArrays/headers/Vec.h"
#include "deviceArrays/headers/Mat.h"
#include <stdexcept>

template<typename Real>
ODE<Real>::ODE(size_t numDims, Real stepSizeScalar, Handle& hand):
    buffers(Mat<Real>::create(numDims, 3, hand)),
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
}

template<typename Real>
void ODE<Real>::rungeKutta4(Real t, Vec<Real> &x, Handle &handle) const {

    buffers.fill(0, handle);
    auto kSum = buffers.col(0);
    auto col1 = buffers.col(1);
    auto col2 = buffers.col(2);

    dxdt(t, x, col1, col2, scalars.get(0), handle);//col1 <- k1
    kSum.add(col1, &stepSizeOver6, &handle);

    dxdt(t + this->stepSizeScalar/2, x, col2, col1, stepSizeOver2, handle);

    kSum.add(col2, &stepSizeOver3, &handle);

    col1.fill(0, handle);
    dxdt(t + this->stepSizeScalar/2, x, col1, col2, stepSizeOver2, handle);

    kSum.add(col1, &stepSizeOver3, &handle);

    col2.fill(0, handle);
    dxdt(t + this->stepSizeScalar, x, col2, col1, stepSize, handle);

    kSum.add(col2, &stepSizeOver6, &handle);

    x.add(kSum, &GPUScalar<Real>::get(1, handle), &handle);
}

template<typename Real>
Mat<Real> ODE<Real>::trajectory(
    Real startTime,
    Mat<Real>& points,
    Real timeIncrement,
    Handle& handle

) const {
    const size_t numberOfPoints = points.toKernel2d().cols;

    for (size_t col = 1; col < numberOfPoints; ++col) {
        auto previous = points.col(col - 1);
        auto current = points.col(col);
        current.set(previous, handle);
        rungeKutta4(startTime + (col - 1) * timeIncrement, current, handle);
    }

    return points;
}

template class ODE<float>;
template class ODE<double>;