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
 *
 * @par Zero-copy staging buffer
 * This class owns a single pinned host buffer, exposed via pinnedPtr(), used
 * for BOTH the right-hand side going in and the solution coming out. The
 * caller (typically Fortran, via get_pinned_ptr_d/s in FortranBindings.hpp)
 * writes the RHS directly into that buffer -- once, aliasing the pointer,
 * not once per timestep -- then calls solve(), then synch(), then reads the
 * solution back out of the exact same buffer. No host-side copy happens
 * inside this class at all: solve()'s H2D transfer reads pinnedPtr()
 * directly, and its D2H transfer writes the solution back into it,
 * overwriting the RHS that has, by that point, already been fully consumed.
 *
 * That overwrite is safe specifically because everything in solve() runs on
 * one CUDA stream (hand): the H2D read of pinnedPtr() is enqueued strictly
 * before the D2H write back into it, and same-stream operations execute in
 * enqueue order, so the write can never race the read.
 *
 * A single shared buffer (rather than separate RHS/solution buffers) is
 * possible because x and b are always the same size for a given solver:
 * L x = b is a square system, and both are built as same-length columns of
 * one Mat in initEigenDecompSolver (see FortranBindings.hpp) -- there's no
 * configuration in which they could differ.
 *
 * @tparam Real
 */
template<typename Real>
class EigenDecompForFortran {
    std::unique_ptr<EigenDecompSolver<Real>> eds = nullptr;//This may hold any type of eigen solver, 2d, 3d, or Thomas.
    SimpleArray<Real> x, b, adjToB;

    /// Single pinned staging buffer, sized to x.size() (== b.size(), see
    /// class-level comment). Used for both the incoming RHS and the
    /// outgoing solution -- see solve()/synch().
    std::unique_ptr<Real[], decltype(&cudaFreeHost)> pinnedBuf{nullptr, &cudaFreeHost};

    static Real* allocPinned(size_t n) {
        Real* p = nullptr;
        cudaMallocHost(&p, n * sizeof(Real));
        return p;
    }
public:
    Handle hand;
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
 * @param gpuIndex Which device this solver's data and computation live on.
 */
    EigenDecompForFortran(GridDim dim, const XYZ<std::vector<Real>> &delta, XYZ<bool> startIsNeumann,
                          XYZ<bool> endIsNeumann, XYZ<Real> startVal, XYZ<Real> endVal,
                          XYZ<eigen::LaplOperatorT> segType,
                          bool thomas, Real helmholtzShift, SimpleArray<Real> sizeOfBForX,
                          SimpleArray<Real> sizeOfBForRHS,
                          SimpleArray<Real> sizeOfBForBAdj, size_t gpuIndex);

    /**
     * @brief The pinned host buffer used for both the RHS (before solve())
     * and the solution (after synch()).
     *
     * Intended to be fetched once, right after construction, and aliased
     * (e.g. via ISO_C_BINDING's C_F_POINTER on the Fortran side) rather
     * than re-fetched every timestep -- the address never changes for the
     * lifetime of this solver.
     *
     * @return Raw pointer to dim1*dim2*dim3 elements of pinned host memory.
     *         Writing to it before solve() supplies the RHS; reading from
     *         it after synch() retrieves the solution.
     */
    Real* pinnedPtr() const { return pinnedBuf.get(); }

    /**
     * @brief Launches L x = b asynchronously, where b is whatever is
     * currently sitting in the buffer returned by pinnedPtr().
     *
     * The caller must have written the RHS into pinnedPtr() before calling
     * this. Returns immediately once the GPU work is enqueued -- call
     * synch() before reading the solution back out of pinnedPtr().
     */
    void solve();

    /**
     * @brief Waits for this handle's launched solve to finish.
     *
     * After this returns, the solution is available by reading from
     * pinnedPtr() -- the same buffer the RHS was written into before
     * solve(). No separate output buffer or copy is involved.
     */
    void synch();
};

#endif //CUDABANDED_EIGENDECOMPFORFORTRAN_H
