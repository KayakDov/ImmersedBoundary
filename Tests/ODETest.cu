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
 * @brief Numerical validation test for Lyapunov exponent calculations using GTest.
 */

#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/Vec.h"
#include "deviceArrays/headers/handle.h"
#include "deviceArrays/headers/Handles.h"
#include "ODE/Lorenz.cuh"
#include "ODE/TimeSequence.h"
#include "ODE/LyapunovExponent.h"
#include "ODE/AttractorDim.h"
#include "math/Real3d.h"

#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <string>
#include <random>
#include <iterator>
#include <vector>


/**
 * @brief Checks trajectory creation and optionally opens its VTK window.
 *
 * Verifies the initial point, finite coordinates, and nonstationary motion.
 * This is a smoke test, not a numerical convergence test.
 *
 * Each curve is generated with TimeSequence::trajectory() into one shared
 * matrix, then drawn; PhasePortrait keeps its own snapshot of every
 * curve, so the matrix is reused for the second one.
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
    auto scratch = Mat<double>::create(3, equation.scratchColumns(), handle);

    PhasePortrait<double> portrait("Lorenz trajectory test");

    const Real3d starts[] = {{1.0, 1.0, 1.0}, {1.0, 1.1, 1.0}};
    const XYZ<double> colors[] = {{1, 0, 0}, {0, 1, 0}};

    for (size_t curveIndex = 0; curveIndex < std::size(starts); ++curveIndex) {
        double hostStart[3];
        starts[curveIndex].toArray(hostStart);
        points.col(0).set(hostStart, size_t{1}, handle);

        equation.trajectory(points, handle, scratch);
        portrait.draw(points, handle, colors[curveIndex]);
    }

    const char* showPortrait = std::getenv("SHOW_LORENZ_PORTRAIT");
    if (showPortrait && std::strcmp(showPortrait, "1") == 0) {
        portrait.show();
    }
}


/**
 * @brief Verifies that LyapunovExponent computes the maximal exponent for
 *        the Lorenz system, on a single handle, under two norms.
 *
 * Runs trajectory integration across multiple renormalization intervals
 * and compares the resulting rate against the known chaotic attractor
 * value of approximately 0.9056, once with the Euclidean norm and once
 * measuring growth by the x component of each deviation alone, with
 * Gram-Schmidt and the reset still Euclidean. Both are held to the same
 * tolerance: the single-component norm is expected to converge just as
 * well, and this is where that gets checked.
 */
TEST(ODELyapunov, ComputesLorenzLargestExponent) {
    constexpr double stepSize = 0.005;
    constexpr size_t N = 3;
    constexpr size_t numExponents = 1;

    Handles handles(1);
    Handle& handle = handles[0];
    Lorenz<double> equation(stepSize, handle);

    auto lyapunovBuffer = Mat<double>::create(N, LyapunovExponent<double>::bufferColumns(equation, numExponents), handle);
    auto lyapunovSingletons = Vec<double>::create(LyapunovExponent<double>::numSingletons(numExponents), handle);
    LyapunovExponent<double> lyapunov(equation, lyapunovBuffer, lyapunovSingletons, handles);

    auto initialPoint = Vec<double>::create(N, handle);
    double hostInitial[3];
    Real3d(1.0, 1.0, 1.0).toArray(hostInitial);
    initialPoint.set(hostInitial, handle);

    constexpr size_t numIntervals = 2500;
    constexpr size_t stepsPerInterval = 20;
    constexpr double epsilon = 1e-8;
    constexpr double expectedLambda = 0.9056;
    constexpr double tolerance = 0.08;

    const Norm1d<double> xComponent(0);
    const Norm<double>* norms[] = {&euclideanNorm<double>(), &xComponent};

    for (size_t normIndex = 0; normIndex < std::size(norms); ++normIndex) {
        SCOPED_TRACE(normIndex == 0 ? "Euclidean norm" : "x component only");

        std::vector<double> exponents = lyapunov.compute(
            initialPoint,
            numIntervals,
            stepsPerInterval,
            epsilon,
            stepSize,
            0,
            *norms[normIndex]
        );

        EXPECT_NEAR(exponents[0], expectedLambda, tolerance);
    }
}


/**
 * @brief Verifies the top three Lyapunov exponents of the Lorenz system,
 *        one handle per exponent.
 *
 * Checks that the largest is near 0.9056 and the middle one near zero
 * (the direction along the flow), that the three come back in descending
 * order, and that they sum to the trace of the Jacobian. Lorenz's
 * divergence is the constant -(sigma + 1 + beta) = -41/3, so volume
 * contraction is deterministic: the sum converges far faster than any
 * single exponent does, which makes it the tightest check on the
 * orthonormalization -- the third exponent, near -14.57, is covered by
 * it rather than by a tolerance of its own. Expected values from
 * Hartl's "Lyapunov exponents in constrained and unconstrained ordinary
 * differential equations," Eq. 2.35 and 2.40.
 *
 * The deviations share columns with the scratch of handles that are
 * still stepping, so a missing join between streams would show up here
 * as garbage in the sum.
 *
 * The tolerances are the existing 0.08 for the individual exponents and
 * 0.01 for the sum; neither has been checked against a run.
 */
TEST(ODELyapunov, ComputesLorenzTopThreeExponents) {
    constexpr double stepSize = 0.005;
    constexpr size_t N = 3;
    constexpr size_t numExponents = 3;

    Handles hands(numExponents);
    Lorenz<double> equation(stepSize, hands[0]);

    auto lyapunovBuffer = Mat<double>::create(N, LyapunovExponent<double>::bufferColumns(equation, numExponents), hands[0]);
    auto lyapunovSingletons = Vec<double>::create(LyapunovExponent<double>::numSingletons(numExponents), hands[0]);
    LyapunovExponent<double> lyapunov(equation, lyapunovBuffer, lyapunovSingletons, hands);

    auto initialPoint = Vec<double>::create(N, hands[0]);
    double hostInitial[3];
    Real3d(1.0, 1.0, 1.0).toArray(hostInitial);
    initialPoint.set(hostInitial, hands[0]);

    constexpr size_t numIntervals = 2500;
    constexpr size_t stepsPerInterval = 20;
    constexpr double epsilon = 1e-8;
    constexpr double expectedLargest = 0.9056;
    constexpr double expectedTrace = -(10.0 + 1.0 + 8.0 / 3.0);
    constexpr double tolerance = 0.08;
    constexpr double traceTolerance = 0.01;

    std::vector<double> exponents = lyapunov.compute(
        initialPoint,
        numIntervals,
        stepsPerInterval,
        epsilon,
        stepSize
    );

    EXPECT_NEAR(exponents[0], expectedLargest, tolerance);
    EXPECT_NEAR(exponents[1], 0.0, tolerance);
    EXPECT_NEAR(exponents[0] + exponents[1] + exponents[2], expectedTrace, traceTolerance);
    EXPECT_GT(exponents[0], exponents[1]);
    EXPECT_GT(exponents[1], exponents[2]);
}


/**
 * @brief Verifies the top three Lorenz exponents when growth is measured
 *        under a pseudo-norm, on three handles.
 *
 * Runs with the x component alone and with the x and y rows together.
 * Gram-Schmidt and the reset stay Euclidean; only the measurement of
 * growth changes, and it divides by the start length each perturbation
 * had under the pseudo-norm rather than by epsilon. Over a run that
 * differs from the Euclidean result only by end terms of the form
 * ln N(q) / T, so the individual exponents keep the usual 0.08 and the
 * trace check loosens from 0.01 to 0.05. A numpy model of the same
 * scheme on the same system landed within about 0.01 of the Euclidean
 * result and within 0.01 of the trace; neither tolerance has been
 * checked against a run of this code.
 */
TEST(ODELyapunov, ComputesLorenzTopThreeExponentsUnderPseudoNorms) {
    constexpr double stepSize = 0.005;
    constexpr size_t N = 3;
    constexpr size_t numExponents = 3;

    Handles hands(numExponents);
    Lorenz<double> equation(stepSize, hands[0]);

    auto lyapunovBuffer = Mat<double>::create(N, LyapunovExponent<double>::bufferColumns(equation, numExponents), hands[0]);
    auto lyapunovSingletons = Vec<double>::create(LyapunovExponent<double>::numSingletons(numExponents), hands[0]);
    LyapunovExponent<double> lyapunov(equation, lyapunovBuffer, lyapunovSingletons, hands);

    auto initialPoint = Vec<double>::create(N, hands[0]);
    double hostInitial[3];
    Real3d(1.0, 1.0, 1.0).toArray(hostInitial);
    initialPoint.set(hostInitial, hands[0]);

    constexpr size_t numIntervals = 2500;
    constexpr size_t stepsPerInterval = 20;
    constexpr double epsilon = 1e-8;
    constexpr double expectedLargest = 0.9056;
    constexpr double expectedTrace = -(10.0 + 1.0 + 8.0 / 3.0);
    constexpr double tolerance = 0.08;
    constexpr double traceTolerance = 0.05;

    const Norm1d<double> xComponent(0);
    const NormRows<double> xyRows(0, 2);
    const Norm<double>* norms[] = {&xComponent, &xyRows};

    for (size_t normIndex = 0; normIndex < std::size(norms); ++normIndex) {
        SCOPED_TRACE(normIndex == 0 ? "x component only" : "x and y rows");

        std::vector<double> exponents = lyapunov.compute(
            initialPoint,
            numIntervals,
            stepsPerInterval,
            epsilon,
            stepSize,
            0,
            *norms[normIndex]
        );

        EXPECT_NEAR(exponents[0], expectedLargest, tolerance);
        EXPECT_NEAR(exponents[1], 0.0, tolerance);
        EXPECT_NEAR(exponents[0] + exponents[1] + exponents[2], expectedTrace, traceTolerance);
        EXPECT_GT(exponents[0], exponents[1]);
        EXPECT_GT(exponents[1], exponents[2]);
    }
}


/**
 * @brief Checks the Kaplan-Yorke formula against spectra with known
 *        answers. No GPU work.
 *
 * The Lorenz spectrum (0.9056, 0, -14.5723) gives 2 + 0.9056 / 14.5723,
 * about 2.0621. A spectrum with no non-negative exponent has dimension
 * zero, one whose running sum never goes negative saturates at its own
 * length, and the input need not arrive sorted.
 */
TEST(AttractorDimension, KaplanYorkeFromKnownSpectra) {
    EXPECT_NEAR(AttractorDim<double>::kaplanYorke({0.9056, 0.0, -14.5723}), 2.0621, 1e-3);
    EXPECT_NEAR(AttractorDim<double>::kaplanYorke({-14.5723, 0.0, 0.9056}), 2.0621, 1e-3);
    EXPECT_DOUBLE_EQ(AttractorDim<double>::kaplanYorke({-0.5, -1.0}), 0.0);
    EXPECT_DOUBLE_EQ(AttractorDim<double>::kaplanYorke({0.5, 0.25}), 2.0);
}


/**
 * @brief Computes the Kaplan-Yorke dimension of the Lorenz attractor
 *        through AttractorDim, on three handles.
 *
 * The known value is about 2.06. The tolerance of 0.05 is generous: it
 * is the exponents' error divided by |lambda_3|, near 14.6, so the
 * 0.08 that the individual exponents are held to costs about 0.005
 * here.
 */
TEST(ODELyapunov, ComputesLorenzAttractorDimension) {
    constexpr double stepSize = 0.005;
    constexpr size_t N = 3;
    constexpr size_t numExponents = 3;

    Handles hands(numExponents);
    Lorenz<double> equation(stepSize, hands[0]);

    auto lyapunovBuffer = Mat<double>::create(N, LyapunovExponent<double>::bufferColumns(equation, numExponents), hands[0]);
    auto lyapunovSingletons = Vec<double>::create(LyapunovExponent<double>::numSingletons(numExponents), hands[0]);
    LyapunovExponent<double> lyapunov(equation, lyapunovBuffer, lyapunovSingletons, hands);
    AttractorDim<double> attractorDim(lyapunov);

    auto initialPoint = Vec<double>::create(N, hands[0]);
    double hostInitial[3];
    Real3d(1.0, 1.0, 1.0).toArray(hostInitial);
    initialPoint.set(hostInitial, hands[0]);

    constexpr size_t numIntervals = 2500;
    constexpr size_t stepsPerInterval = 20;
    constexpr double epsilon = 1e-8;

    const double dimension = attractorDim.compute(initialPoint, numIntervals, stepsPerInterval, epsilon, stepSize);

    EXPECT_NEAR(dimension, 2.0621, 0.05);
}

namespace {

/**
 * @brief Runs one largest-Lyapunov-exponent estimate using an
 *        already-constructed LyapunovExponent computing a single
 *        exponent.
 *
 * Does not build a new Lorenz or LyapunovExponent itself; only start,
 * numIntervals, stepsPerInterval, and warmupSteps vary per call.
 * stepSize is still taken as a parameter, purely for compute()'s dt
 * argument, and must match whatever stepSize the passed-in lyapunov's
 * underlying Lorenz was actually constructed with -- this function has
 * no way to check that. handle must be the handle lyapunov was
 * constructed with, since the starting point is uploaded on it.
 *
 * If warmupSteps is nonzero, the reference path is stepped forward that
 * many times before measurement begins, so the trajectory has settled
 * onto the attractor rather than still being in its transient approach
 * from an arbitrary starting point. This matters more for starting
 * points far from the attractor than for nearby ones -- without it,
 * different starting points spend different fractions of a fixed
 * interval budget on an irrelevant transient, which is one of the two
 * reasons different starting points give different estimates (the other
 * is simply not having run enough intervals yet). Defaults to 0 (no
 * warm-up).
 */
double runLyapunov(
    LyapunovExponent<double>& lyapunov,
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

    auto initialPoint = Vec<double>::create(N, handle);
    initialPoint.set(hostStart, handle);

    return lyapunov.compute(
        initialPoint, numIntervals, stepsPerInterval, epsilon, stepSize, warmupSteps
    )[0];
}

/**
 * @brief Prints one CSV-formatted row to standard output.
 */
void writeRow(const std::string& row) {
    std::cout << row << std::endl;
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
 * Every sweep computes the largest exponent only, on a single handle.
 *
 * Sweep "resolution" refines the RK4 timestep stepSize, holding the
 * renormalization interval's physical duration (timePerInterval =
 * stepsPerInterval * stepSize -- called tau in Hartl's paper) and the
 * number of intervals fixed. Because stepSize is fixed at Lorenz's
 * construction, this sweep rebuilds Lorenz and LyapunovExponent once per
 * stepSize value.
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
 * The sampleCount and startingPoint sweeps both use stepSize 0.005, so
 * they share one Lorenz and one LyapunovExponent; that same Lorenz sizes
 * the workspace buffer every sweep reuses.
 *
 * Disabled by default so that running every test stays quick and quiet. Run on demand with
 *   --gtest_also_run_disabled_tests --gtest_filter=ODELyapunov.DISABLED_ConvergenceStudy
 * Expect roughly a minute total across all three sweeps.
 */
TEST(ODELyapunov, DISABLED_ConvergenceStudy) {
    Handles handles(1);
    Handle& handle = handles[0];
    constexpr size_t N = 3;
    constexpr size_t numExponents = 1;
    constexpr double fixedStepSize = 0.005;

    Lorenz<double> fixedEquation(fixedStepSize, handle);

    auto lyapunovBuffer = Mat<double>::create(N, LyapunovExponent<double>::bufferColumns(fixedEquation, numExponents), handle);
    auto lyapunovSingletons = Vec<double>::create(LyapunovExponent<double>::numSingletons(numExponents), handle);
    LyapunovExponent<double> fixedLyapunov(fixedEquation, lyapunovBuffer, lyapunovSingletons, handles);

    const Real3d defaultStart(1.0, 1.0, 1.0);

    writeRow("sweep,stepSize,stepsPerInterval,numIntervals,timePerInterval,warmupSteps,x,y,z,lambda");

    {
        constexpr double timePerInterval = 0.1;
        constexpr size_t numIntervals = 500;

        for (double stepSize = 0.02; stepSize >= 0.00125; stepSize /= 2) {
            size_t stepsPerInterval = std::round(timePerInterval / stepSize);

            Lorenz<double> equation(stepSize, handle);
            LyapunovExponent<double> lyapunov(equation, lyapunovBuffer, lyapunovSingletons, handles);

            double lambda = runLyapunov(
                lyapunov, stepSize, numIntervals, stepsPerInterval, defaultStart, handle
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
        constexpr size_t stepsPerInterval = 20;
        const size_t intervalCounts[] = {100, 250, 500, 1000, 2500, 5000};

        for (size_t intervalCountIndex = 0; intervalCountIndex < std::size(intervalCounts); ++intervalCountIndex) {
            const size_t numIntervals = intervalCounts[intervalCountIndex];
            double lambda = runLyapunov(
                fixedLyapunov, fixedStepSize, numIntervals, stepsPerInterval, defaultStart, handle
            );

            std::ostringstream row;
            row << "sampleCount," << fixedStepSize << "," << stepsPerInterval << "," << numIntervals << ","
                << (stepsPerInterval * fixedStepSize) << "," << 0 << ","
                << defaultStart.x << "," << defaultStart.y << "," << defaultStart.z << ","
                << std::setprecision(8) << lambda;
            writeRow(row.str());

            mostRefinedLambda = lambda;
        }
    }

    {
        constexpr size_t stepsPerInterval = 20;
        constexpr size_t numIntervals = 2500;
        constexpr size_t warmupSteps = 2000;

        const Real3d startingPoints[] = {
            {1.0, 1.0, 1.0},
            {5.0, 5.0, 5.0},
            {-3.0, 4.0, 10.0},
            {0.1, 0.1, 0.1},
            {10.0, -10.0, 20.0}
        };

        for (size_t startingPointIndex = 0; startingPointIndex < std::size(startingPoints); ++startingPointIndex) {
            const Real3d& start = startingPoints[startingPointIndex];
            double lambda = runLyapunov(
                fixedLyapunov, fixedStepSize, numIntervals, stepsPerInterval, start, handle, warmupSteps
            );

            std::ostringstream row;
            row << "startingPoint," << fixedStepSize << "," << stepsPerInterval << "," << numIntervals << ","
                << (stepsPerInterval * fixedStepSize) << "," << warmupSteps << ","
                << start.x << "," << start.y << "," << start.z << ","
                << std::setprecision(8) << lambda;
            writeRow(row.str());
        }
    }

    EXPECT_NEAR(mostRefinedLambda, 0.9056, 0.08);
}


/**
 * @brief Generates a 15x15 grid of (numIntervals, stepsPerInterval) ->
 *        average error from the expected Lyapunov exponent, printed as
 *        CSV rows suitable for a heatmap or 3D surface in a spreadsheet.
 *
 * numIntervals is how many renormalization samples are averaged, and
 * stepsPerInterval is how many RK4 steps pass between renormalizations.
 * stepSize is held constant and small (0.001) rather than swept:
 * ConvergenceStudy's resolution sweep showed no clear trend in the
 * estimate across stepSize from 0.02 down to 0.00125, so the variation
 * worth mapping is in how much sampling is done and how often it
 * happens. A small stepSize means each step covers little physical time
 * (numIntervals * stepsPerInterval * stepSize is the total), so the
 * stepsPerInterval range starts at 30 and runs to 442, giving interval
 * lengths of 0.03 to 0.44 time units; the Lyapunov time for this system
 * is about 1.1, and intervals much shorter than that would leave most
 * of the grid covering too little total time to converge.
 *
 * Both axes have 15 values and grow geometrically, so the grid covers
 * their ranges evenly on a log scale. numIntervals starts at 100 and
 * grows by a factor of 1.3 up to 4000 (the last value is 3896).
 * stepsPerInterval starts at 30 and grows by a factor of 1.22 up to 500
 * (the last value is 442). Both are whole numbers, so each
 * multiplication truncates back to an integer, which keeps each
 * sequence strictly increasing for these starting values.
 *
 * Starting points are drawn fresh, from a normal distribution (mean 0,
 * standard deviation 10, independently per axis), for every one of the
 * three trials at every grid cell -- not reused across cells -- so the
 * grid is dense enough (225 cells) for the underlying trend to read as
 * a gradient despite real cell-to-cell noise. Seeded (42) for
 * reproducibility. Each point is burned in via warmupSteps before being
 * counted; a draw landing very close to the origin (the system's
 * unstable equilibrium) could still need longer than warmupSteps to
 * actually leave it, so a handful of visibly anomalous cells is the
 * symptom to look for, fixed by raising warmupSteps or changing the
 * seed rather than by anything on this grid's two axes.
 *
 * Lorenz and LyapunovExponent are built once, outside both loops, since
 * stepSize never changes; the grid computes the largest exponent only,
 * on a single handle.
 *
 * Disabled by default so that running every test stays quick and quiet. Run on demand with
 *   --gtest_also_run_disabled_tests --gtest_filter=ODELyapunov.DISABLED_ConvergenceSurface
 * Expect roughly three hours, since the larger stepsPerInterval values
 * multiply the step count. If that's too long while iterating,
 * temporarily lowering numRandomStarts or narrowing either axis (the
 * numIntervals upper bound especially) is the quickest way to shorten
 * it -- expect a noisier, less-converged surface in exchange.
 */
TEST(ODELyapunov, DISABLED_ConvergenceSurface) {
    Handles handles(1);
    Handle& handle = handles[0];
    constexpr size_t N = 3;
    constexpr size_t numExponents = 1;
    constexpr double stepSize = 0.001;
    constexpr double expectedLambda = 0.9056;
    constexpr size_t warmupSteps = 2000;
    constexpr size_t numRandomStarts = 3;

    Lorenz<double> equation(stepSize, handle);

    auto lyapunovBuffer = Mat<double>::create(N, LyapunovExponent<double>::bufferColumns(equation, numExponents), handle);
    auto lyapunovSingletons = Vec<double>::create(LyapunovExponent<double>::numSingletons(numExponents), handle);
    LyapunovExponent<double> lyapunov(equation, lyapunovBuffer, lyapunovSingletons, handles);

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 10.0);

    writeRow("numIntervals,stepsPerInterval,stepSize,avgLambda,error");

    for (size_t numIntervals = 100; numIntervals <= 4000; numIntervals *= 1.3) {
        for (size_t stepsPerInterval = 30; stepsPerInterval <= 500; stepsPerInterval *= 1.22) {
            double lambdaSum = 0;
            for (size_t randomStartIndex = 0; randomStartIndex < numRandomStarts; ++randomStartIndex) {
                Real3d start(dist(rng), dist(rng), dist(rng));
                lambdaSum += runLyapunov(
                    lyapunov, stepSize, numIntervals, stepsPerInterval,
                    start, handle, warmupSteps
                );
            }
            double avgLambda = lambdaSum / numRandomStarts;
            double error = std::abs(avgLambda - expectedLambda);

            std::ostringstream row;
            row << std::setprecision(8)
                << numIntervals << "," << stepsPerInterval << "," << stepSize << ","
                << avgLambda << "," << error;
            writeRow(row.str());
        }
    }
}
