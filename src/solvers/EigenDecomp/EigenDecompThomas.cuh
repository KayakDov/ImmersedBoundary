/**
 * @file EigenDecompThomas.cuh
 * @brief Eigen-decomposition solver that applies the Thomas algorithm along one axis.
 * @ingroup eigen_decomp
 */

#ifndef CUDABANDED_EIGENDECOMPTHOMAS_CUH
#define CUDABANDED_EIGENDECOMPTHOMAS_CUH
#include "EigenDecomp3d.cuh"


/**
 * @class EigenDecompThomas
 * @brief A hybrid 3D Poisson/Helmholtz solver: eigen-decomposition along X and Y, the Thomas algorithm along Z.
 *
 * Diagonalizes only X (columns) and Y (rows). In the transformed space, Z (layers) is left as one
 * independent tridiagonal system per (x, y) pair, each solved with the Thomas algorithm (TDMA).
 * Z is the Thomas axis because, with the memory layout y + rows * (z + layers * x), the Z transform
 * (KroneckerTriplet::multDepths) is a batched GEMM while the X and Y transforms are single GEMMs, so
 * skipping Z skips the most expensive transform.
 *
 * Memory: besides the caller's x and b, the solver needs one workspace the size of b, the same as
 * EigenDecomp3d. The Thomas step keeps its modified right-hand side in the output x until the final
 * values replace it, and keeps its modified super-diagonal in the workspace.
 *
 * Requires at least 2 layers; the constructors throw std::invalid_argument otherwise.
 *
 * @tparam T Floating-point type (float or double).
 * @tparam SegmentT The segment type of the Z boundary.
 */
template<typename T, typename SegmentT>
class EigenDecompThomas : public EigenDecomp3d<T> {
protected:

    /** The boundary conditions for the z dimension. */
    const AxisSegmentHost<SegmentT> boundaryZ;

    /**
     * @brief Solves the tridiagonal systems along Z in the eigen-transformed space.
     *
     * For each (i, j) in the transformed XY plane, solves
     * \f[ ( L_z + \lambda_x[i] + \lambda_y[j] - \sigma I ) \tilde{u} = \tilde{f} \f]
     * where \f$ L_z \f$ is the 1D tridiagonal operator for Z and \f$ \sigma \f$ is the Helmholtz shift.
     *
     * Memory use, with nothing beyond rhs and dst:
     * - rhs doubles as the workspace. Each entry is overwritten with a modified super-diagonal
     *   coefficient once it has been used, so its values are lost. solve() passes the solver's own
     *   workspace, which holds the transformed right-hand side and is not needed afterwards.
     * - dst holds the modified right-hand side while the systems are eliminated, and the solution
     *   once back substitution finishes.
     *
     * Not to be confused with EigenDecomp3d::multLEigenValInverse, which this class does not
     * override and does not use: EigenDecomp3d::solve is the only caller of that method, and this
     * class has its own solve.
     *
     * @param rhs Input in transformed space (f-tilde); dim.size() values. Overwritten (see above).
     * @param dst Output in transformed space (u-tilde); dim.size() values. Must not be the same
     * memory as rhs.
     * @param hand CUDA handle for the kernel launch.
     * @throws std::invalid_argument if dst is the same memory as rhs.
     */
    void solveTridiagonalSystems(SimpleArray<T> &rhs, SimpleArray<T> &dst, Handle &hand) const;

    /**
     * @brief Throws if the grid has fewer than 2 layers, which the Thomas sweep along Z cannot handle.
     * @throws std::invalid_argument if dim.layers < 2.
     */
    void requireTwoLayers() const;

public:
    /**
     * @brief Constructs the solver from precomputed eigenvectors and eigenvalues.
     * @param eigen The eigen vectors and values.
     * @param boundZ The boundary conditions in the z dimension.
     * @param sizeOfB Workspace the size of b. If the caller will not need b after solve(), b's own
     * memory may be passed here; solve() then overwrites b. Otherwise b is never modified. Must not
     * be the same memory as the x passed to solve().
     * @param isSingular true if the Laplacian is singular.
     * @param helmholtzShift set to a non zero value to solve (L - sigma I)x = b where the helmholtzShift is sigma.
     */
    EigenDecompThomas(const Eigen<T> &eigen, const AxisSegmentHost<SegmentT> &boundZ, SimpleArray<T> sizeOfB, bool isSingular, T helmholtzShift = 0);

    /**
     * @brief Constructs the solver on a caller-supplied workspace.
     * @param boundary The boundary conditions.
     * @param hands At least 3 handles.
     * @param sizeOfB Workspace the size of b. If the caller will not need b after solve(), b's own
     * memory may be passed here; solve() then overwrites b. Otherwise b is never modified. Must not
     * be the same memory as the x passed to solve().
     * @param helmholtzShift set to a non zero value to solve (L - sigma I)x = b where the helmholtzShift is sigma.
     */
    template<typename SegX, typename SegY>
    EigenDecompThomas(const BoundaryConfigHost<T, SegX, SegY, SegmentT>& boundary, Handles &hands, SimpleArray<T> sizeOfB, T helmholtzShift = 0);

    /**
     * @brief Constructs the solver and allocates its own workspace (one array the size of b).
     * @param boundary The boundary conditions.
     * @param hands At least 3 handles.
     * @param helmholtzShift set to a non zero value to solve (L - sigma I)x = b where the helmholtzShift is sigma.
     */
    template<class SegX, class SegY>
    EigenDecompThomas(const BoundaryConfigHost<T, SegX, SegY, SegmentT> &boundary, Handles &hands, T helmholtzShift = 0);

    /**
     * @brief Solves (L - sigma I) x = b.
     *
     * b is not modified, unless its memory was also given to this solver as its workspace (sizeOfB).
     * x is used as scratch before the solution is written into it.
     *
     * Internally, the Thomas step overwrites the transformed right-hand side in the workspace with
     * its super-diagonal coefficients (see solveTridiagonalSystems), since the transformed
     * right-hand side is not needed again.
     *
     * Be sure that b is in the column space of L. Otherwise you will receive a projection onto the
     * column space that will not actually solve Lx = b.
     *
     * @param x Receives the solution. Must not be the same memory as b or as the workspace.
     * @param b The right-hand side.
     * @param hand The handle to run on.
     */
    void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const override;
};


// Deduction guide for the constructor without a workspace
template<typename Real, typename SegX, typename SegY, typename SegZ>
EigenDecompThomas(const BoundaryConfigHost<Real, SegX, SegY, SegZ>&, Handles&, Real)
    -> EigenDecompThomas<Real, SegZ>;

// Deduction guide for the constructor with a workspace
template<typename Real, typename SegX, typename SegY, typename SegZ>
EigenDecompThomas(const BoundaryConfigHost<Real, SegX, SegY, SegZ>&, Handles&, SimpleArray<Real>, Real)
    -> EigenDecompThomas<Real, SegZ>;

#endif //CUDABANDED_EIGENDECOMPTHOMAS_CUH
