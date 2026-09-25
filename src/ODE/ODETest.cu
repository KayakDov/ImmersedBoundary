//
// Created by usr on 9/15/26.
//
/**
 * @file LorenzTest.cu
 * @brief Generates and optionally displays a Lorenz trajectory using GTest.
 */

#include "deviceArrays/headers/Mat.h"
#include "ODE/PhasePortrait3d.h"
#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "ODE/Lorenz.cuh"

/**
 * @file LyapunovTest.cu
 * @brief Numerical validation test for largest Lyapunov exponent calculations using GTest.
 */

#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/Vec.h"
#include "deviceArrays/headers/handle.h"
#include "ODE/Lorenz.cuh"
#include "ODE/TimeSequence.h"

#include <gtest/gtest.h>
#include <cmath>


/**
 * @brief Checks trajectory creation and optionally opens its VTK window.
 *
 * Verifies the initial point, finite coordinates, and nonstationary motion.
 * This is a smoke test, not a numerical convergence test.
 *
 * The interactive window is only opened when SHOW_LORENZ_PORTRAIT=1 is set
 * in the environment (set by CMake's SHOW_LORENZ_PORTRAIT option when run
 * via ctest -- see CMakeLists.txt). It defaults to skipped, not shown, when
 * the variable is absent entirely, since that's what happens for a plain
 * run_unit_tests invocation outside ctest (e.g. a raw binary run or most
 * IDE "Run" configurations) -- portrait.show() blocks the whole process
 * until a human closes the window, which would otherwise hang every other
 * test in the suite behind it, and hang indefinitely with no human present
 * at all under CI.
 */
TEST(ODETrajectory, CreatesLorenzCurve) {
    constexpr double h = 0.005;
    constexpr size_t pointCount = 4001;

    Handle handle;
    Lorenz<double> equation(h, handle);

    auto points = Mat<double>::create(3, pointCount, handle);

    PhasePortrait<double> portrait("Lorenz trajectory test");

    portrait.draw(points, &equation, {1, 1, 1}, handle, {1, 0, 0});
    portrait.draw(points, &equation, {1, 1.1, 1}, handle, {0, 1, 0});

    const char* showPortrait = std::getenv("SHOW_LORENZ_PORTRAIT");
    if (showPortrait && std::strcmp(showPortrait, "1") == 0) {
        portrait.show();
    }
}


/**
 * @brief Verifies that largestLyapunovExponent computes the maximal exponent for the Lorenz system.
 *
 * Runs trajectory integration across multiple renormalization intervals and compares the resulting
 * rate against the known chaotic attractor value of approximately 0.9056.
 */
TEST(ODELyapunov, ComputesLorenzLargestExponent) {
    constexpr double h = 0.005;
    constexpr size_t N = 3;

    Handle handle;
    Lorenz<double> equation(h, handle);

    auto initialPoint = Vec<double>::create(N, handle);
    double hostInitial[3] = {1.0, 1.0, 1.0};
    initialPoint.set(hostInitial, handle);

    auto buffer = Mat<double>::create(N, 3, handle);
    auto singletons = Vec<double>::create(2, handle);

    constexpr size_t numIntervals = 2500;
    constexpr size_t stepsPerInterval = 20;
    constexpr double epsilon = 1e-8;
    constexpr double expectedLambda = 0.9056;
    constexpr double tolerance = 0.08;

    double estimatedLambda = equation.largestLyapunovExponent(
        initialPoint,
        buffer,
        singletons,
        numIntervals,
        stepsPerInterval,
        epsilon,
        h
    );

    handle.synch();

    EXPECT_NEAR(estimatedLambda, expectedLambda, tolerance);
}
