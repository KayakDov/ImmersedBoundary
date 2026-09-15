//
// Created by usr on 9/15/26.
//
/**
 * @file LorenzTest.cu
 * @brief Generates and optionally displays a Lorenz trajectory using GTest.
 */

#include "deviceArrays/headers/Mat.h"
#include "ODE/ODE.h"
#include "ODE/PhasePortrait3d.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

/**
 * @brief Evaluates the Lorenz derivative at state + scale * offset.
 *
 * Launched with exactly one thread. All vectors have three components,
 * and the destination does not overlap either input.
 */
__global__ void lorenzTestDerivative(
    DeviceData1d<double> state,
    DeviceData1d<double> offset,
    DeviceData1d<double> dst,
    const double* scale
) {
    const double a = *scale;
    const double x = state[0] + a * offset[0];
    const double y = state[1] + a * offset[1];
    const double z = state[2] + a * offset[2];

    dst[0] = 10.0 * (y - x);
    dst[1] = x * (28.0 - z) - y;
    dst[2] = x * y - (8.0 / 3.0) * z;
}

/**
 * @brief Lorenz equation using the production ODE integrator.
 */
class LorenzTestEquation final : public ODE<double> {
public:
    /**
     * @brief Creates the three-dimensional equation.
     * @param h Physical timestep.
     * @param handle GPU execution handle.
     */
    LorenzTestEquation(double h, Handle& handle)
        : ODE<double>(3, h, handle) {}

    /**
     * @brief Overwrites dst with f(x + scalarForAddToX * addToX).
     *
     * Time is unused because the Lorenz system is autonomous.
     * Uses the supplied stream without allocating derivative storage.
     */
    void dxdt(
        double /*t*/,
        const Vec<double>& x,
        Vec<double> dst,
        Vec<double> addToX,
        const Singleton<double> scalarForAddToX,
        Handle& handle
    ) const override {
        lorenzTestDerivative<<<1, 1, 0, handle>>>(
            x.toKernel1d(),
            addToX.toKernel1d(),
            dst.toKernel1d(),
            scalarForAddToX.data()
        );

        CHECK_CUDA_ERROR(cudaGetLastError());
    }
};

/**
 * @brief Checks trajectory creation and optionally opens its VTK window.
 *
 * Verifies the initial point, finite coordinates, and nonstationary motion.
 * This is a smoke test, not a numerical convergence test.
 */
TEST(ODETrajectory, CreatesLorenzCurve) {
    constexpr double h = 0.005;
    constexpr size_t pointCount = 4001;

    Handle handle;
    LorenzTestEquation equation(h, handle);

    auto points = Mat<double>::create(3, pointCount, handle);

    const double initial[3] = {1.0, 1.0, 1.0};
    auto first = points.col(0);
    first.set(initial, size_t{1}, handle);

    auto curve = equation.trajectory(0.0, points, 1, handle);

    // One bulk transfer, with consecutive xyz tuples on the CPU.
    std::vector<double> coordinates(3 * pointCount);
    curve.get(coordinates.data(), size_t{3}, handle);

    const cudaError_t status = cudaStreamSynchronize(handle);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);

    EXPECT_EQ(curve._rows, size_t{3});
    EXPECT_EQ(curve._cols, pointCount);

    for (size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(coordinates[i], initial[i]);

    ASSERT_TRUE(std::all_of(
        coordinates.begin(),
        coordinates.end(),
        [](double value) { return std::isfinite(value); }
    ));

    // At the initial state, dy/dt = 26, so y should initially increase.
    EXPECT_GT(coordinates[4], initial[1]);

    const char* show = std::getenv("SHOW_LORENZ_PORTRAIT");

    if (show && show[0] == '1') {
        PhasePortrait<double> portrait("Lorenz trajectory test");
        portrait.draw(curve, handle);
        portrait.show();
    }
}

} // namespace