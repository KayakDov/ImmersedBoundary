#ifndef LORENZ_CUH
#define LORENZ_CUH

#include "deviceArrays/headers/Mat.h"
#include "ODE/ODE.h"


__global__ void lorenzDerivative(
    DeviceData1d<double> state,
    DeviceData1d<double> offset,
    DeviceData1d<double> dst,
    const double* a
) {

    const double x = state[0] + *a * offset[0];
    const double y = state[1] + *a * offset[1];
    const double z = state[2] + *a * offset[2];

    dst[0] = 10.0 * (y - x);
    dst[1] = x * (28.0 - z) - y;
    dst[2] = x * y - (8.0 / 3.0) * z;
}

/**
 * @brief Lorenz system with sigma=10, rho=28, and beta=8/3.
 *
 * The three state entries represent x, y, and z.
 * Uses the existing ODE class's RK4 integrator.
 */
template <typename Real>
class Lorenz final : public ODE<Real> {
public:
    explicit Lorenz(double stepSize, Handle& handle)
        : ODE<Real>(3, stepSize, handle) {}

    /**
     * @brief Overwrites dst with f(x + scalarForAddToX * addToX).
     *
     * The Lorenz equations have no explicit time dependence.
     * Input vectors and dst must each contain three entries.
     * dst must not overlap either input.
     */
    void dxdt(
        double t,
        const Vec<double>& x,
        Vec<double> dst,
        Vec<double> addToX,
        const Singleton<double> scalarForAddToX,
        Handle& handle
        ) const override {

        lorenzDerivative<<<1, 1, 0, handle>>>(
            x.toKernel1d(),
            addToX.toKernel1d(),
            dst.toKernel1d(),
            scalarForAddToX.data()
        );

        CHECK_CUDA_ERROR(cudaGetLastError());
        }
};

#endif