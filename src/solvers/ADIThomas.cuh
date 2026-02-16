/**
* @file ADIThomas.cuh
 *
 * @brief ADI-based linear solver using Thomas algorithm line solves.
 *
 * This file defines the ADIThomas class, which implements an
 * Alternating Direction Implicit (ADI) iterative solver for structured
 * grid problems. Each ADI substep reduces to a collection of independent
 * tridiagonal systems, which are solved using the Thomas algorithm on the GPU.
 */
#ifndef CUDABANDED_ADITHOMAS_CUH
#define CUDABANDED_ADITHOMAS_CUH
#include "SimpleArray.h"
#include "Thomas.cuh"
#include "Support/GridDim.hpp"

/**
 * @class ADIThomas
 * @tparam Real Floating-point type (e.g. float or double).
 *
 * @brief ADI iterative solver for structured-grid linear systems.
 *
 * This class solves linear systems of the form
 * \f[
 *   L x = b
 * \f]
 * where the operator \f$L\f$ is the laplacian.
 *
 * The solver uses an Alternating Direction Implicit (ADI) iteration,
 * in which each iteration is decomposed into one-dimensional implicit
 * solves along rows, columns, and depths. Each implicit subproblem
 * is tridiagonal and is solved efficiently using the Thomas algorithm.
 *
 * Convergence is determined by the relative \f$L_2\f$ norm of the residual
 * \f$r = b - A x\f$.
 */
template <typename Real>
class ADIThomas {
    /** @brief Grid dimensions and strides of the underlying problem. */
    const GridDim dim;
    /** @brief Maximum number of ADI iterations allowed. */
    const size_t maxIterations;
    /** @brief Relative convergence tolerance on the residual norm. */
    Real tolerance;

    /** @brief Residual array \f$r = b - A x\f$. */
    SimpleArray<Real> rhs, xThirdStep;
    /**
     * @brief Scratch workspace used by Thomas solvers.
     *
     * This buffer is reused across row, column, and depth solves
     * to avoid repeated allocations.
     */
    Mat<Real> thomasSratchMaxDim;
    /** @brief Thomas solvers tridiagonal systems. */
    Thomas<Real> thomasRows, thomasCols, thomasDepths;

    /** @brief \f$L_2\f$ norm of the residual. */
    Singleton<Real> rNorm, bNorm;


public:
    /**
     * @brief Construct an ADIThomas solver.
     *
     * @param dim Grid dimensions describing the structured domain.
     * @param max_iterations Maximum number of ADI iterations.
     * @param dimSizeX2 a matrix used for scratch work.  This matrix should have the same number of rows as b and two columns.
     * @param tolerance Relative convergence tolerance on
     *        \f$\|r\|_2 / \|b\|_2\f$.
     */
    ADIThomas(const GridDim &dim, size_t max_iterations, const Real &tolerance, Mat<Real> dimSizeX2, Handle &hand);

    void solveType1(SimpleArray<Real> &x, const SimpleArray<Real> &b, Handle &hand);

    /**
     * @brief Solves the system $Ax = b$ using the ADI iterative process.
     * * The iteration continues until the $L_2$ norm of the residual $r = b - Lx$
     * relative to the $L_2$ norm of $b$ is less than the tolerance.
     * * @param x [in/out] Solution array (initial guess provided as input).
     * @param b [in] Source term array.
     * @param hand CUDA Handle for stream management.
     */
    void solveType2(SimpleArray<Real> &x, const SimpleArray<Real> &b, Handle &hand);

    static void test1();

    static void test2();
};



#endif //CUDABANDED_ADITHOMAS_CUH
