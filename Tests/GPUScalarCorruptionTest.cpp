/**
 * @file GPUScalarCorruptionTest.cpp
 * @brief Shows that ImmersedEq's "prime" solve corrupts the shared
 *        GPUScalar constant 1.0 for the rest of the process, and that this
 *        makes an ordinary, unrelated computation come out wrong immediately
 *        afterward.
 *
 * ImmersedEq::setRHSFPrime (ImmersedEquation.cu) calls
 *
 *   RHSF.subtract(lagrangeVec(LagrangeInd::UGamma), &dT, GPUScalar<Real>::get(1, hand), &hand);
 *
 * passing GPUScalar<Real>::get(1, hand) as subtract's `buffer` scratch
 * argument. subtract immediately overwrites buffer
 * (buffer.set(-1,hand); buffer.mult(alpha,hand);) before ever reading it --
 * harmless for a private scratch buffer, but GPUScalar<Real>::get(1, hand)
 * is not one: it is a single, process-wide cached Singleton
 * (GPUScalar<T>::instances is a static map, built once per device and never
 * rebuilt), and Singleton's underlying GpuPointer::window() aliases memory
 * "exactly like shared_ptr's own aliasing constructor" (GpuPointer.h)
 * rather than copying it. So this one call permanently overwrites the GPU
 * memory that every other GPUScalar<Real>::get(1, ...) call in the process
 * reads as "1.0", from that point on.
 *
 * This reuses the exact fixture from ImmersedEq.SolvesPrimes_3x2x1 (the one
 * existing test that reaches setRHSFPrime at all -- confirmed by grepping
 * every ".solve(" call in the whole test tree) to trigger the real code
 * path, then checks the constant directly, and with one trivial arithmetic
 * computation no one could argue about the correct answer to.
 *
 * Fails on the current code; passes once setRHSFPrime's subtract(...) call
 * goes back to using a private scratch buffer (e.g. sparseMultBuffer->get(0),
 * what it used before commit 704cd57) instead of GPUScalar<Real>::get(1, hand).
 */

#include <gtest/gtest.h>

#include "solvers/EigenDecomp/EigenDecomp2d.h"
#include "solvers/EigenDecomp/EigenDecompThomas.cuh"
#include "poisson/Poisson.cuh"
#include <cmath>
#include <random>
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"

#include "wrapper/FortranBindings.hpp"
#include "deviceArrays/headers/DeviceMemory.h"
#include "immersedBoundary/ImmersedEquation.h"
#include "kronecker/KroneckerTriplet.h"
#include "poisson/Laplacian1d.cuh"

TEST(GPUScalarCorruption, ImmersedEqPrimeSolveCorruptsSharedConstantOne) {
    using Real = double;
    using Int  = int32_t;

    // --- Fixture copied verbatim from ImmersedEq.SolvesPrimes_3x2x1 ---
    GridDim dim(3, 2, 1);
    Real3d delta(1, 1, 1);
    Handle hand;

    auto boundaryHost = makeUniformBoundaryConfigHost<Real>(
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        delta, dim, false
    );

    size_t uStarSize =
        dim.numDims() * dim.size()
        + dim.cols * dim.layers
        + dim.rows * dim.layers
        + dim.cols * dim.rows * (dim.layers > 1);

    std::vector<Real> uStar(uStarSize, 0);
    uStar[3] = 1;
    uStar[9] = 1;

    std::vector<Int> colOffsetsR(3);
    colOffsetsR[0] = 0;
    colOffsetsR[1] = uStarSize;
    colOffsetsR[2] = uStarSize;

    std::vector<Int> rowIndsR(uStarSize);
    for (size_t i = 0; i < uStarSize; ++i) rowIndsR[i] = static_cast<Int>(i);

    std::vector<Real> valsR(uStarSize, 1);
    std::vector<Real> UGamma(2, 3);
    double deltaT = 3.0 / 2.0;

    std::vector<Int> rowOffsetsB = {0, 1, 2};
    std::vector<Int> colIndsB    = {0, 1};
    std::vector<Real> valuesB    = {1, 1};
    std::vector<Real> f = {1, 2};

    std::vector<Real> p(dim.size(), 0);
    p[0] = 2;
    p[dim.size() - 1] = -2;

    std::vector<Real> resultP(dim.size(), 0);
    std::vector<Real> resultF(f.size(), 0);

    ImmersedEq<Real, Int> imEq(boundaryHost.forDevice(), f.size(), valsR.size(), p.data(), f.data(), deltaT, 1e-8, 1000);

    // Plain (non-prime) solve first -- setRHSFPrime is only reachable from
    // the "prime" overload below, so this just reproduces
    // SolvesPrimes_3x2x1's own call order.
    imEq.solve(resultP.data(), valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data());
    cudaDeviceSynchronize();

    // --- "Before": the baseline every other call site in the process relies on ---
    Real oneBefore = -12345.0;
    GPUScalar<Real>::get(1, hand).get(&oneBefore, hand);
    hand.synch();
    ASSERT_NEAR(oneBefore, 1.0, 1e-12)
        << "GPUScalar<Real>::get(1, hand) should read 1.0 before the prime solve runs at all.";

    // --- The prime solve: the only call anywhere in the suite that reaches
    // setRHSFPrime, and therefore the only one that reaches the buggy
    // subtract(..., GPUScalar<Real>::get(1, hand), ...) line. ---
    imEq.solve(
        resultP.data(),
        resultF.data(),
        valuesB.size(),
        rowOffsetsB.data(),
        colIndsB.data(),
        valuesB.data(),
        valsR.size(),
        colOffsetsR.data(),
        rowIndsR.data(),
        valsR.data(),
        UGamma.data(),
        uStar.data()
    );
    cudaDeviceSynchronize();

    // --- "After": nothing about solving a linear system should be able to
    // change what "1.0" means for the rest of the process. ---
    Real oneAfter = -54321.0;
    GPUScalar<Real>::get(1, hand).get(&oneAfter, hand);
    hand.synch();
    EXPECT_NEAR(oneAfter, 1.0, 1e-12)
        << "GPUScalar<Real>::get(1, hand) no longer reads 1.0 after ImmersedEq's prime solve -- "
        << "got " << oneAfter << ". setRHSFPrime's subtract(...) call wrote through it; "
        << "see ImmersedEquation.cu's setRHSFPrime.";

    // --- Make the consequence impossible to argue with: an ordinary
    // computation that trusts "the constant 1", done right after, comes out
    // wrong. 10 + 1*4 is 14 -- that was never in question. ---
    auto x = Vec<Real>::create(1, hand);
    auto y = Vec<Real>::create(1, hand);
    Real ten = 10.0, four = 4.0;
    x.set(&ten, hand);
    y.set(&four, hand);
    x.add(y, &GPUScalar<Real>::get(1, hand), &hand);
    hand.synch();

    Real result = -1.0;
    x.get(&result, hand);
    hand.synch();
    EXPECT_NEAR(result, 14.0, 1e-9)
        << "10 + GPUScalar<Real>::get(1,hand)*4 should be 14; got " << result
        << " -- the shared constant 1 was left corrupted by the prime solve above.";
}
