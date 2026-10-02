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
#include "ODE/LargestLyapunovExponent.h"
#include "math/Real3d.h"

#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <string>
#include <random>


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
    constexpr double stepSize = 0.005;
    constexpr size_t pointCount = 4001;

    Handle handle;
    Lorenz<double> equation(stepSize, handle);

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
 * @brief Verifies that LargestLyapunovExponent computes the maximal exponent for the Lorenz system.
 *
 * Runs trajectory integration across multiple renormalization intervals and compares the resulting
 * rate against the known chaotic attractor value of approximately 0.9056.
 */
TEST(ODELyapunov, ComputesLorenzLargestExponent) {
    constexpr double stepSize = 0.005;
    constexpr size_t N = 3;

    Handle handle;
    Lorenz<double> equation(stepSize, handle);

    auto lyapunovBuffer = Mat<double>::create(N, 3, handle);
    auto lyapunovSingletons = Vec<double>::create(2, handle);
    LargestLyapunovExponent<double> lyapunov(equation, lyapunovBuffer, lyapunovSingletons);

    auto initialPoint = Vec<double>::create(N, handle);
    double hostInitial[3];
    Real3d(1.0, 1.0, 1.0).toArray(hostInitial);
    initialPoint.set(hostInitial, handle);

    constexpr size_t numIntervals = 2500;
    constexpr size_t stepsPerInterval = 20;
    constexpr double epsilon = 1e-8;
    constexpr double expectedLambda = 0.9056;
    constexpr double tolerance = 0.08;

    double estimatedLambda = lyapunov.compute(
        initialPoint,
        numIntervals,
        stepsPerInterval,
        epsilon,
        stepSize
    );

    handle.synch();

    EXPECT_NEAR(estimatedLambda, expectedLambda, tolerance);
}


namespace {

/**
 * @brief Runs one largest-Lyapunov-exponent estimate using an
 *        already-constructed LargestLyapunovExponent.
 *
 * Does not build a new Lorenz or LargestLyapunovExponent itself; only
 * start, numIntervals, stepsPerInterval, and warmupSteps vary per call.
 * stepSize is still taken as a parameter, purely for compute()'s dt
 * argument, and must match whatever stepSize the passed-in lyapunov's
 * underlying Lorenz was actually constructed with -- this function has
 * no way to check that. sequence must be the same underlying object
 * lyapunov wraps.
 *
 * If warmupSteps is nonzero, sequence is stepped forward that many times
 * via TimeSequence::pointAt() before measurement begins, so the
 * trajectory has settled onto the attractor rather than still being in
 * its transient approach from an arbitrary starting point. This matters
 * more for starting points far from the attractor than for nearby ones
 * -- without it, different starting points spend different fractions of
 * a fixed interval budget on an irrelevant transient, which is one of
 * the two reasons different starting points give different estimates
 * (the other is simply not having run enough intervals yet). Defaults
 * to 0 (no warm-up), unchanged from before this parameter existed.
 */
double runLyapunov(
    TimeSequence<double>& sequence,
    LargestLyapunovExponent<double>& lyapunov,
    double stepSize,
    size_t numIntervals,
    size_t stepsPerInterval,
    const Real3d& start,
    Handle& handle,
    size_t warmupSteps = 0
) {
    constexpr size_t N = 3;
    constexpr double epsilon = 1e-8;

    double hostStart[3];
    start.toArray(hostStart);

    auto rawStart = Vec<double>::create(N, handle);
    rawStart.set(hostStart, handle);

    auto initialPoint = Vec<double>::create(N, handle);
    sequence.pointAt(warmupSteps, rawStart, initialPoint);

    double lambda = lyapunov.compute(
        initialPoint, numIntervals, stepsPerInterval, epsilon, stepSize
    );

    handle.synch();
    return lambda;
}

/**
 * @brief Prints one CSV-formatted row to standard output.
 */
void writeRow(const std::string& row) {
    std::cout << row << std::endl;
}


/**
 * @brief Returns count values geometrically (log-) spaced from low to
 *        high inclusive.
 *
 * Used for the convergence-surface grid axes rather than linear spacing,
 * since Lyapunov-exponent convergence error is expected to behave
 * roughly like 1/sqrt(numIntervals) (Hartl's sigma/sqrt(N) argument) --
 * geometric spacing gives even visual coverage of that kind of curve
 * across the whole range, rather than clustering most points at the low
 * end or the high end.
 */
std::vector<double> geometricSpace(double low, double high, size_t count) {
    std::vector<double> values;
    values.reserve(count);
    double ratio = std::pow(high / low, 1.0 / static_cast<double>(count - 1));
    double v = low;
    for (size_t i = 0; i < count; ++i) {
        values.push_back(v);
        v *= ratio;
    }
    return values;
}

} // namespace

/**
 * @brief Generates one convergence table covering three sweeps, printed
 *        to standard output as CSV-formatted rows ready to copy into a
 *        spreadsheet. Every row uses the same ten columns --
 *        sweep,stepSize,stepsPerInterval,numIntervals,timePerInterval,
 *        warmupSteps,x,y,z,lambda -- regardless of which sweep produced
 *        it, so whichever parameter a given sweep holds fixed just
 *        repeats the same value down its column; the sweep column
 *        identifies which study each row belongs to. warmupSteps is 0
 *        for the resolution and sampleCount sweeps (no burn-in) and
 *        nonzero for the startingPoint sweep -- shown explicitly so two
 *        rows with the same displayed x,y,z are never silently different
 *        experiments: a warmed-up and an un-warmed-up run starting from
 *        the same raw point measure from different actual points, and
 *        give different lambda accordingly.
 *
 * Sweep "resolution" refines the RK4 timestep stepSize, holding the
 * renormalization interval's physical duration (timePerInterval =
 * stepsPerInterval * stepSize -- called tau in Hartl's paper) and the
 * number of intervals fixed. Because stepSize is fixed at Lorenz's
 * construction, this sweep rebuilds Lorenz and LargestLyapunovExponent
 * once per stepSize value.
 *
 * Sweep "sampleCount" increases the number of renormalization intervals,
 * holding stepSize and stepsPerInterval fixed. Each interval is one
 * "sample" in Hartl's rescaled-deviation-vector sense (Eq. 2.7-2.8 of
 * "Lyapunov exponents in constrained and unconstrained ordinary
 * differential equations," 2003) -- this sweep is the direct analogue of
 * his sigma/sqrt(N) convergence argument, and demonstrates that
 * numIntervals=1000 is not yet converged: the estimate is still rising
 * noticeably between 1000 and 5000 intervals.
 *
 * Sweep "startingPoint" varies the starting point, holding stepSize,
 * stepsPerInterval, and numIntervals fixed, to check that the estimate
 * doesn't depend on where a trajectory starts. numIntervals is raised to
 * 2500 here (matching ComputesLorenzLargestExponent's validated value,
 * rather than the under-converged 1000 used previously), and each
 * starting point is burned in for warmupSteps steps first via
 * runLyapunov()'s warmupSteps argument, so a starting point far from the
 * attractor isn't penalized relative to one that starts close to it.
 *
 * Not part of the routine fast suite -- run on demand with
 *   --gtest_filter=ODELyapunov.ConvergenceStudy
 * Expect roughly a minute total across all three sweeps.
 */
TEST(ODELyapunov, ConvergenceStudy) {
    Handle handle;
    constexpr size_t N = 3;

    auto lyapunovBuffer = Mat<double>::create(N, 3, handle);
    auto lyapunovSingletons = Vec<double>::create(2, handle);

    const Real3d defaultStart(1.0, 1.0, 1.0);

    writeRow("sweep,stepSize,stepsPerInterval,numIntervals,timePerInterval,warmupSteps,x,y,z,lambda");

    {
        constexpr double timePerInterval = 0.1;
        constexpr size_t numIntervals = 500;
        const double stepSizeValues[] = {0.02, 0.01, 0.005, 0.0025, 0.00125};

        for (double stepSize : stepSizeValues) {
            size_t stepsPerInterval = static_cast<size_t>(std::round(timePerInterval / stepSize));

            Lorenz<double> equation(stepSize, handle);
            LargestLyapunovExponent<double> lyapunov(equation, lyapunovBuffer, lyapunovSingletons);

            double lambda = runLyapunov(
                equation, lyapunov, stepSize, numIntervals, stepsPerInterval, defaultStart, handle
            );

            std::ostringstream row;
            row << "resolution," << std::setprecision(8)
                << stepSize << "," << stepsPerInterval << "," << numIntervals << ","
                << (stepsPerInterval * stepSize) << "," << 0 << ","
                << defaultStart.x << "," << defaultStart.y << "," << defaultStart.z << ","
                << lambda;
            writeRow(row.str());
        }
    }

    double mostRefinedLambda = 0;
    {
        constexpr double stepSize = 0.005;
        constexpr size_t stepsPerInterval = 20;
        const size_t intervalCounts[] = {100, 250, 500, 1000, 2500, 5000};

        Lorenz<double> equation(stepSize, handle);
        LargestLyapunovExponent<double> lyapunov(equation, lyapunovBuffer, lyapunovSingletons);

        for (size_t numIntervals : intervalCounts) {
            double lambda = runLyapunov(
                equation, lyapunov, stepSize, numIntervals, stepsPerInterval, defaultStart, handle
            );

            std::ostringstream row;
            row << "sampleCount," << stepSize << "," << stepsPerInterval << "," << numIntervals << ","
                << (stepsPerInterval * stepSize) << "," << 0 << ","
                << defaultStart.x << "," << defaultStart.y << "," << defaultStart.z << ","
                << std::setprecision(8) << lambda;
            writeRow(row.str());

            mostRefinedLambda = lambda;
        }
    }

    {
        constexpr double stepSize = 0.005;
        constexpr size_t stepsPerInterval = 20;
        constexpr size_t numIntervals = 2500;
        constexpr size_t warmupSteps = 2000;

        Lorenz<double> equation(stepSize, handle);
        LargestLyapunovExponent<double> lyapunov(equation, lyapunovBuffer, lyapunovSingletons);

        const Real3d startingPoints[] = {
            {1.0, 1.0, 1.0},
            {5.0, 5.0, 5.0},
            {-3.0, 4.0, 10.0},
            {0.1, 0.1, 0.1},
            {10.0, -10.0, 20.0}
        };

        for (const auto& p : startingPoints) {
            double lambda = runLyapunov(
                equation, lyapunov, stepSize, numIntervals, stepsPerInterval, p, handle, warmupSteps
            );

            std::ostringstream row;
            row << "startingPoint," << stepSize << "," << stepsPerInterval << "," << numIntervals << ","
                << (stepsPerInterval * stepSize) << "," << warmupSteps << ","
                << p.x << "," << p.y << "," << p.z << ","
                << std::setprecision(8) << lambda;
            writeRow(row.str());
        }
    }

    EXPECT_NEAR(mostRefinedLambda, 0.9056, 0.08);
}


/**
 * @brief Generates a 15x15 grid of (numIntervals, timePerInterval) ->
 *        average error from the expected Lyapunov exponent, printed as
 *        CSV rows suitable for a heatmap or 3D surface in a spreadsheet.
 *
 * numIntervals and timePerInterval are each swept over 15
 * geometrically-spaced values -- 100 to 4000, and 0.02 to 0.4
 * respectively, the same endpoints used before, just with more
 * intermediate points (see geometricSpace() above).
 *
 * Starting points are drawn fresh -- from a normal distribution (mean 0,
 * standard deviation 10, independently per axis) -- for every one of the
 * three trials at every single grid cell, rather than reusing the same
 * points across cells. This deliberately leaves real point-to-point
 * noise in the surface instead of removing it: at 15x15 (225 cells) the
 * grid is dense enough that the underlying trend across numIntervals and
 * timePerInterval should still read as a clear overall gradient, with a
 * reader's eye smoothing past the local cell-to-cell scatter, the way
 * one reads noisy experimental data generally. The generator is still
 * seeded (42), so rerunning this test reproduces the identical noise
 * pattern rather than a different one each time.
 *
 * Each point is still burned in via warmupSteps before being counted,
 * the same reasoning as ConvergenceStudy's starting-point sweep -- more
 * important here, since these points aren't curated to be reasonably
 * close to the attractor, and a draw landing very close to the origin
 * (the system's unstable equilibrium) could still need longer than
 * warmupSteps to actually leave it; a handful of visibly anomalous cells
 * is the symptom to look for if that happens, in which case increasing
 * warmupSteps or changing the seed is the fix, not necessarily more
 * intervals.
 *
 * Lorenz and LargestLyapunovExponent are built once, outside every loop,
 * and reused for all 225 cells and all 675 trials -- stepSize never
 * changes anywhere in this grid.
 *
 * Not part of the routine fast suite -- run on demand with
 *   --gtest_filter=ODELyapunov.ConvergenceSurface
 * Expect roughly half an hour: 225 cells x 3 starting points, with the
 * largest cells running several hundred thousand RK4 steps each, plus
 * warmup. If that's too long while iterating, temporarily lowering
 * numRandomStarts or the top of the numIntervals range is the quickest
 * way to shorten it.
 */
TEST(ODELyapunov, ConvergenceSurface) {
    Handle handle;
    constexpr size_t N = 3;
    constexpr double stepSize = 0.005;
    constexpr double expectedLambda = 0.9056;
    constexpr size_t warmupSteps = 2000;
    constexpr size_t numRandomStarts = 3;
    constexpr size_t gridSize = 15;

    auto lyapunovBuffer = Mat<double>::create(N, 3, handle);
    auto lyapunovSingletons = Vec<double>::create(2, handle);

    Lorenz<double> equation(stepSize, handle);
    LargestLyapunovExponent<double> lyapunov(equation, lyapunovBuffer, lyapunovSingletons);

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 10.0);

    const std::vector<double> numIntervalsValues = geometricSpace(100.0, 4000.0, gridSize);
    const std::vector<double> timePerIntervalValues = geometricSpace(0.02, 0.4, gridSize);

    writeRow("numIntervals,timePerInterval,avgLambda,error");

    for (double numIntervalsExact : numIntervalsValues) {
        size_t numIntervals = static_cast<size_t>(std::round(numIntervalsExact));

        for (double timePerInterval : timePerIntervalValues) {
            size_t stepsPerInterval = static_cast<size_t>(std::round(timePerInterval / stepSize));

            double lambdaSum = 0;
            for (size_t i = 0; i < numRandomStarts; ++i) {
                Real3d start(dist(rng), dist(rng), dist(rng));
                lambdaSum += runLyapunov(
                    equation, lyapunov, stepSize, numIntervals, stepsPerInterval,
                    start, handle, warmupSteps
                );
            }
            double avgLambda = lambdaSum / numRandomStarts;
            double error = std::abs(avgLambda - expectedLambda);

            std::ostringstream row;
            row << numIntervals << "," << timePerInterval << ","
                << std::setprecision(8) << avgLambda << "," << error;
            writeRow(row.str());
        }
    }
}
