#pragma once

#include <gtest/gtest.h>

#include "immersedBoundary/ImerssedEquation.h"
#include "poisson/Laplacian.cuh"

template <typename Real, typename Int>
SparseCSR<Real, Int> basics(const BoundaryConfig<Real>& boundary, size_t n, std::vector<Real>& x0Host, SquareMat<Real>& lhsOperator, std::vector<Int>& rowOffsetsB, std::vector<Real>& valuesB, std::vector<Int>& colIndsB, Handle& hand) {

    std::mt19937 rng(42); // Deterministic seed for reproducible testing
    std::uniform_real_distribution<Real> dist(-5.0, 5.0);
    for (size_t i = 0; i < n; ++i) x0Host[i] = dist(rng);

    auto B = SparseCSR<Real, Int>::create(valuesB.size(), rowOffsetsB.size() - 1, n, hand);
    B.set(rowOffsetsB.data(), colIndsB.data(), valuesB.data(), hand);

    poisson::laplacian(boundary, hand).getDense(lhsOperator, hand);
    auto denseB = Mat<Real>::create(rowOffsetsB.size() - 1, n);
    B.getDense(denseB, hand);
    denseB.mult(denseB, &lhsOperator, &hand, &GPUScalar<Real>::get(2), &GPUScalar<Real>::get(1), true, false);

    return B;
}

template <typename Real, typename Int>
void verifyImmersedEqPrimeWithBoundary(const BoundaryConfig<Real>& boundary, Handle& hand, Real tolerance, const std::string& locMsg, Mat<Real>& bufferNXNPlus) {
    std::stringstream errorMsg;
    errorMsg << locMsg << '\n';
    const size_t n = boundary.dim().size();

    // 1. Setup sparse boundary matrix B
    std::vector<Real> pPrimeHost(n);
    auto lhsOperator = bufferNXNPlus.sqSubMat(0, 0, n);

    std::vector<Int> rowOffsetsB = {0, 2, 4};
    std::vector<Int> colIndsB = {0, 1, 0, 1};
    std::vector<Real> valuesB = {1.0, -1.0, -1.0, 1.0};
    //const BoundaryConfig<Real>& boundary, size_t n, std::vector<Real>& x0Host, Mat<Real>& lhsOperator, std::vector<Real>& rowOffsetsB, std::vector<Real>& valuesB, std::vector<Real>& colIndsB, Handle& hand
    auto B = basics<Real, Int>(boundary, n, pPrimeHost, lhsOperator, rowOffsetsB, valuesB, colIndsB, hand);
    size_t numB = B.offsets.size() - 1;


    // 4. Populate exact manufactured solution field targets for p' and provisional velocity u*
    std::vector<Real> uStarHost(boundary.dim().velocitiesStaggeredSize());
    for (size_t i = 0; i < uStarHost.size(); ++i) uStarHost[i] = static_cast<Real>(0.125 * (i + 1.0));

    auto pPrimeExpected = SimpleArray<Real>::create(n, hand);
    pPrimeExpected.set(pPrimeHost.data(), hand);
    errorMsg << "pPrimeExpected = " << GpuOut<Real>(pPrimeExpected, hand) << '\n';

    // 5. Generate matching restriction operator R and derive consistent U^Gamma boundary conditions
    std::vector<Int> colOffsetsR(numB + 1), rowIndsR(numB);
    std::vector<Real> valuesR(numB, 1), rhsFPrimeHost(numB), UGamma(numB);

    for (size_t j = 0; j < numB; ++j) {
        colOffsetsR[j] = rowIndsR[j] = static_cast<Int>(j);
        Real rTransposeU = 0;
        for (Int k = colOffsetsR[j]; k < colOffsetsR[j + 1]; ++k) rTransposeU += valuesR[k] * uStarHost[rowIndsR[k]];
        rhsFPrimeHost[j] = static_cast<Real>(0.25 * (j + 1));
        UGamma[j] = rTransposeU - rhsFPrimeHost[j] / 1.5;
    }

    colOffsetsR[numB] = static_cast<Int>(numB);

    printVec("uStar", uStarHost); printVec("R offsets", colOffsetsR); printVec("R inds", rowIndsR);
    printVec("R vals", valuesR); printVec("rhsFPrime", rhsFPrimeHost); printVec("UGamma", UGamma);

    // 6. Stage device-side storage allocations for testing variables
    auto rhsFPrime = SimpleArray<Real>::create(numB, hand);
    rhsFPrime.set(rhsFPrimeHost.data(), hand);
    auto rhsPPrime = SimpleArray<Real>::create(n, hand);
    rhsPPrime.fill(0, hand);

    // 7. Apply Eulerian discrete Laplacian operator: L * p'
    poisson::laplacian(boundary, hand).bandedMult(pPrimeExpected, rhsPPrime, &hand, GPUScalar<Real>::get(1), GPUScalar<Real>::get(0), false);
    errorMsg << "L pPrimeExpected = " << GpuOut<Real>(rhsPPrime, hand) << '\n';

    // 8. Lambda managing localized scratch spaces and execution handles for sparse SpMV operations
    auto multB = [&](const auto& src, auto& dst, Real alpha, Real beta, bool trans) {
        auto ws = SimpleArray<Real>::create(B.multWorkspaceSize(src, dst, GPUScalar<Real>::get(alpha), GPUScalar<Real>::get(beta), trans, hand), hand);
        B.mult(src, dst, GPUScalar<Real>::get(alpha), GPUScalar<Real>::get(beta), trans, ws, hand);
    };
    auto tempB = SimpleArray<Real>::create(numB, hand);
    multB(pPrimeExpected, tempB, 1, 0, false); // tempB = B * p'
    multB(tempB, rhsPPrime, 2, 1, true);       // rhsPPrime += 2 * B^T * B * p'
    multB(rhsFPrime, rhsPPrime, -2, 1, true);  // rhsPPrime -= 2 * B^T * RHS_F'
    rhsPPrime.add(poisson::boundaryCorrection(boundary, hand), &GPUScalar<Real>::get(-1), &hand); // Account for physical boundary fluxes
    errorMsg << "Manufactured RHSPPrime = " << GpuOut<Real>(rhsPPrime, hand) << '\n';

    // 9. Compute expected exact boundary force field: F' = 2 * (B * p' - RHS_F')
    std::vector<Real> bPPrimeHost(numB, 0), fPrimeExpectedHost(numB);
    tempB.get(bPPrimeHost.data(), hand);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < numB; ++i) fPrimeExpectedHost[i] = static_cast<Real>(2) * (bPPrimeHost[i] - rhsFPrimeHost[i]);

    // 10. Pull back RHS definitions and dispatch execution via the system solver pipeline
    std::vector<Real> rhsPPrimeHost(n, 0), unusedF(numB, 0), resultP(n, 0), resultF(numB, 0);
    rhsPPrime.get(rhsPPrimeHost.data(), hand);
    cudaDeviceSynchronize();

    ImmersedEq<Real, Int> imEq(boundary, numB, std::max(valuesB.size(), valuesR.size()), rhsPPrimeHost.data(), unusedF.data(), Real3d(1, 1, 1), 1, static_cast<Real>(1e-12), 1000);
    imEq.solve(resultP.data(), resultF.data(), valuesB.size(), rowOffsetsB.data(), colIndsB.data(), valuesB.data(), valuesR.size(), colOffsetsR.data(), rowIndsR.data(), valuesR.data(), UGamma.data(), uStarHost.data());
    cudaDeviceSynchronize();

    // 11. Evaluate floating reference offsets if the system constraints map onto a pure Neumann nullspace
    Real offset = boundary.allNeumann() ? pPrimeHost[0] - resultP[0] : 0;

    // 12. Conduct final relative error validations against target Eulerian and Lagrangian fields
    for (size_t i = 0; i < n; ++i)
        ASSERT_NEAR(resultP[i] + offset, pPrimeHost[i], tolerance * std::max<Real>(1, std::abs(pPrimeHost[i])))
            << locMsg << " - Prime pressure mismatch at flat index " << i << "\nexpected = " << pPrimeHost[i] << "\nactual   = " << resultP[i] << "\noffset   = " << offset << "\n" << errorMsg.str();

    for (size_t i = 0; i < numB; ++i)
        ASSERT_NEAR(resultF[i], fPrimeExpectedHost[i], tolerance * std::max<Real>(1, std::abs(fPrimeExpectedHost[i])))
            << locMsg << " - Prime force mismatch at index " << i << "\nexpected = " << fPrimeExpectedHost[i] << "\nactual   = " << resultF[i] << "\n" << errorMsg.str();
}