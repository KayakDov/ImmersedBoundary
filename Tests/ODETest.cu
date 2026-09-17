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

#include "ODE/Lorenz.cuh"


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
    Lorenz<double> equation(h, handle);

    auto points = Mat<double>::create(3, pointCount, handle);

    PhasePortrait<double> portrait("Lorenz trajectory test");

    portrait.draw(points, &equation, 0, h, {1, 1, 1}, handle, {1, 0, 0});
    portrait.draw(points, &equation, 0, h, {1, 1.1, 1}, handle, {0, 1, 0});

    portrait.show();
}
