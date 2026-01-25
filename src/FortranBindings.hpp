
#include <memory>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "immersedBoundary/ImerssedEquation.h"


/**
 * @file ImmersedEquationInterface.hpp
 * @brief Global interface for initializing and solving the Immersed Boundary Equation.
 * * This file provides a stateful global pointer to an ImmersedEq instance and
 * helper functions to manage its lifecycle.
 */

/**
 * @brief Global variable template holding the unique instance of the Immersed Boundary solver.
 * * @tparam Real The floating-point precision (float or double).
 * @tparam Int The integer type used for sparse indexing (e.g., int32_t, int64_t).
 */
template<typename Real, typename Int> static std::unique_ptr<ImmersedEq<Real, Int>> eq = nullptr;

/**
 * @brief Initializes the global Immersed Boundary solver instance.
 * * This function constructs the underlying BaseData and ImmersedEq objects.
 * It prepares the system to solve the following linear system:
 * * $$ (L + 2B^T B)x = B^T f + p $$ or equivalently:
 * * $$ (I + 2L^{-1}B^T B)x = L^{-1}(B^T f + p) $$
 * * Where:
 * - \f$ L \f$ is the Discrete Laplacian operator (via EigenDecompSolver).
 * - \f$ B \f$ is the Sparse Immersed Boundary interpolation/spreading matrix.
 * - \f$ f \f$ and \f$ p \f$ are the force and pressure vectors.
 * * @tparam Real Floating-point type.
 * @tparam Int Integer type for sparse indices.
 * * @param height Eulerian grid height (number of rows).
 * @param width Eulerian grid width (number of columns).
 * @param depth Eulerian grid depth (number of layers).
 * @param fSize Size of the Lagrangian force vector.
 * @param nnzMaxB Maximum number of non-zero elements allowed in matrix B.
 * @param p Initial Eulerian pressure/scalar field array (device pointer).
 * @param f Initial Lagrangian force field array (device pointer).
 * @param deltaX Grid spacing in X direction.
 * @param deltaY Grid spacing in Y direction.
 * @param deltaZ Grid spacing in Z direction.
 * @param tolerance Convergence tolerance for the BiCGSTAB solver.
 * @param maxBCGIterations Maximum iterations allowed for the linear solver.
 */
template<typename Real, typename Int>
void initImmersedEq(
    const size_t height, const size_t width, const size_t depth,
    const size_t fSize, const size_t nnzMaxB,
    Real* p, Real* f,
    const double deltaX, const double deltaY, const double deltaZ,
    const double tolerance, const size_t maxBCGIterations
) {

    GridDim dim(height, width, depth);

    Real3d delta(deltaX, deltaY, deltaZ);

    eq<Real, Int> = std::make_unique<ImmersedEq<Real, Int>>(dim, fSize, nnzMaxB, p, f, delta, tolerance, maxBCGIterations);
}

/**
 * @brief Solves the Immersed Boundary system using the initialized global instance.
 * * Updates the internal matrix \f$ B \f$ with current coefficients and solves for
 * the unknown vector \f$ x \f$.
 * * @tparam Real Floating-point type.
 * @tparam Int Integer type for sparse indices (defaults to uint32_t).
 * * @param result Pointer to the device memory where the solution will be stored.
 * @param nnzB Number of non-zero elements in the current matrix B.
 * @param rowPointersB CSR/CSC row pointers for matrix B.
 * @param colPointersB CSR/CSC column pointers for matrix B.
 * @param valuesB Values of the non-zero elements in matrix B.
 * @param multiStream If true, uses multiple CUDA streams for asynchronous BCGSTAB operations.
 * * @exception std::runtime_error Thrown if initImmersedEq has not been called for these types.
 */
template<typename Real, typename Int = uint32_t>
void solveImmersedEq(
    Real* result,
    size_t nnzB,
    Int* rowPointersB,
    Int* colPointersB,
    Real* valuesB,
    bool multiStream = true
) {
    if (!eq<Real, Int>) throw std::runtime_error("ImmersedEq not initialized. Call initImmersedEq first.");
    eq<Real, Int>->solve(result, nnzB, rowPointersB, colPointersB, valuesB, multiStream);
}
