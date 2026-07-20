/**
 * @file EigenDecompForFortran.h
 * @brief Declares the public API for EigenDecompForFortran.h.
 * @ingroup eigen_decomp
 *
 * @details
 * Solver interfaces consume the GPU container layer and CUDA library handles to implement direct, iterative, and eigen-decomposition based algorithms.
 */

#ifndef CUDABANDED_EIGENDECOMPFORFORTRAN_H
#define CUDABANDED_EIGENDECOMPFORFORTRAN_H

#include "../solvers/EigenDecomp/EigenDecompThomas.cuh"

/**
 * A wrapper for an eigen decomposition solver of any type with reusable resources for calling the solve method.
 * @tparam Real
 */
template<typename Real>
class EigenDecompForFortran {
    std::unique_ptr<EigenDecompSolver<Real>> eds = nullptr;//This may hold any type of eigen solver, 2d, 3d, or Thomas.
    Handle hand;
    SimpleArray<Real> x, b, adjToB;
public:
    /**
 * @brief Constructs an eigen-decomposition solver for a separable Cartesian Laplacian or Helmholtz operator.
 *
 * The solver supports uniform, variable-spacing, and flux-form Laplacian discretizations
 * independently along each dimension. The grid is specified in internal storage order:
 * dim1 (fastest varying index), dim2, then dim3 (slowest varying index).
 *
 * @param dim Grid dimensions in internal storage order.
 * @param delta Grid spacing for each dimension.
 *        - For UniformDeltaCellCenteredLapl and UniformDeltaCellStaggeredLapl, each
 *          vector contains a single value representing the uniform spacing.
 *        - For VariableDeltaLapl and FluxLapl, each vector contains one spacing value
 *          for every interval along that dimension (typically dimensionLength + 1 values).
 * @param startIsNeumann True if the lower boundary of each dimension uses a Neumann
 *        boundary condition; false for Dirichlet.
 * @param endIsNeumann True if the upper boundary of each dimension uses a Neumann
 *        boundary condition; false for Dirichlet.
 * @param startVal Boundary values (Dirichlet value or Neumann derivative) at the
 *        lower end of each dimension.
 * @param endVal Boundary values (Dirichlet value or Neumann derivative) at the
 *        upper end of each dimension.
 * @param segType Discretization used for each dimension:
 *        UniformDeltaCellCenteredLapl,
 *        UniformDeltaCellStaggeredLapl,
 *        VariableDeltaLapl, or
 *        FluxLapl.
 * @param thomas True to use the Thomas algorithm in the transformed dim3 systems;
 *        false to use the general banded solver.
 * @param helmholtzShift Solves
 *        \f$(L - \sigma I)x = b\f$,
 *        where this parameter is \f$\sigma\f$.
 * @param sizeOfBForX Preallocated GPU workspace containing at least
 *        dim.height * dim.width * dim.depth elements. Contents are overwritten.
 * @param sizeOfBForRHS Preallocated GPU workspace containing at least
 *        dim.height * dim.width * dim.depth elements. Contents are overwritten.
 * @param sizeOfBForBAdj Preallocated GPU workspace containing at least
 *        dim.height * dim.width * dim.depth elements. Contents are overwritten.
 */
    EigenDecompForFortran(GridDim dim, const XYZ<std::vector<Real>> &delta, XYZ<bool> startIsNeumann,
                          XYZ<bool> endIsNeumann, XYZ<Real> startVal, XYZ<Real> endVal,
                          XYZ<eigen::LaplOperatorT> segType,
                          bool thomas, Real helmholtzShift, SimpleArray<Real> sizeOfBForX,
                          SimpleArray<Real> sizeOfBForRHS,
                          SimpleArray<Real> sizeOfBForBAdj);
    /**
     * Solves the equation L x = b.
     * @param xHost The solution overwrites this array.
     * @param bHost The rhs of the equation is input here.
     */
    void solve(Real* xHost, Real* bHost);
};

#endif //CUDABANDED_EIGENDECOMPFORFORTRAN_H
