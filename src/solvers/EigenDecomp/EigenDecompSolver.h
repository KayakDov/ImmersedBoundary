#ifndef EIGENDECOMPSOLVER_H
#define EIGENDECOMPSOLVER_H

#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/SquareMat.h"
#include "deviceArrays/headers/Vec.h"
#include "../../poisson/PoissonRHS.h"
#include "../../deviceArrays/headers/Support/Streamable.h"

#include <cstddef>
#include <vector>

#include "../Event.h"
#include "math/Real3d.h"

/**
 * @brief Direct Poisson solver using eigen-decomposition (Fast Diagonalization Method).
 *
 * This class diagonalizes the 3D discrete Laplacian operator using the
 * Kronecker structure:
 *static EigenDecompSolver<Real>* createEDS(
        SquareMat<Real>& dim,
        Mat<Real>& maxDim,
        SimpleArray<Real>& sizeP,
        Handle* hand,
        Real3d delta)
    {
        if (maxDim._cols == 3) {
            // Standard 3D construction
            return new EigenDecompSolver3d<Real>(dim, dim, dim, maxDim, sizeP, hand, delta);
        } else {
            // Explicitly slice Real3d to Real2d to satisfy the 2D constructor
            return new EigenDecompSolver2d<Real>(dim, dim, maxDim, sizeP, hand, Real2d(delta.x, delta.y));
        }
    }
 *     L = L_x ⊕ L_y ⊕ L_z
 *
 * where each L_i is diagonalized as:
 *
 *     L_i = E_i Λ_i E_i^T
 *
 * The solver:
 *   1. Computes eigenvectors/eigenvalues of each 1D Laplacian.
 *   2. Applies E_x, E_y, E_z to transform f → f̃.
 *   3. Solves ũ(i,j,k) = f̃(i,j,k)/(λ_x(i)+λ_y(j)+λ_z(k)).
 *   4. Applies the inverse transforms to recover u.
 *
 * @tparam T Floating-point type (float or double).
 */
template<typename T>
class EigenDecompSolver {
protected:
    GridDim dim;
    /**
     * @brief Eigenvector matrices for the three 1-D Laplacians.
     *
     * eVecs[0] = eigenvectors of L_x
     * eVecs[1] = eigenvectors of L_y
     * eVecs[2] = eigenvectors of L_z
     */
    std::vector<SquareMat<T>> eVecs;//stored x, y, z which is cols, rows, layers

    /**
     * @brief Eigenvalues matrix.
     *
     * Column 0: eigenvalues of L_x
     * Column 1: eigenvalues of L_y
     * Column 2: eigenvalues of L_z
     */
    std::vector<Vec<T>> eVals;


    /**
     * A workspace the size of b = L_cols.  You may store b itself here, but it will be overwritten.
     */
    mutable SimpleArray<T> sizeOfB;

    void eigenVecsL(size_t i, cudaStream_t stream);

    void eigenValsL(size_t i, double delta, cudaStream_t stream);

    /**
     * @brief Compute eigenvalues and eigenvectors for L[i].
     *
     * @param i Index (0=x, 1=y, 2=z).
     * @param stream CUDA stream to execute kernels on.
     */
    void eigenL(size_t i, Real3d delta, cudaStream_t stream);

    /**
     * takes in the a matrix.  Its last column will become space for an eigen vector and the first n X n columns
     * will store an eigen vector matrix.
     * @param src An n X n+1 matrix where n is height, width, or depth.
     */
    void appendMatAndVec(Mat<T> &src);



public:
    virtual ~EigenDecompSolver() = default;

    /**
     * @brief Construct and immediately solve the Poisson problem.
     *
     * Builds eigenbases for Lx, Ly, Lz, Where L is the left hand side matrix you'd use for solving the Poisson equation.
     * It's a banded matrix with 7 diagonals, etc...
     *
     * A must be the standard second-difference (Toeplitz) discrete Laplacian on a uniform grid with homogeneous Dirichlet boundary conditions.
     *
     * These matrices will be overwritten.
     * @param eMatsAndVecs Should have a number of elements equal to the number of dimensions.  If two dimensions are
     * equal in length, be sure that the matrices passed point to the same gpu memmory.  Each matrix passed should
     * be n X n+1 where n = x (cols), y (rows), and z (layers) if applicable in that order.
     * @param sizeOfB An array the size of b = xLength * yLength * zLength that will be overwritten.  You may use b for this.
     */
    EigenDecompSolver(std::vector<Mat<T>> eMatsAndVecs, SimpleArray<T> &sizeOfB);

    /**
     *
     * @param dim The dimensions of the grid.
     * @param delta The distance between grid points.
     * @param sizeOfB A cratch space that can hold the same number of elements as B.
     */
    EigenDecompSolver(const GridDim &dim, const Real3d &delta, SimpleArray<T> sizeOfB);

    /**
     * Created an eigen decomposition solver where all memory is owned by this object.
     * @param dim The dimensions of grid.
     * @param delta The distance between fgrid points.
     * @param hand The handle.
     */
    EigenDecompSolver(const GridDim &dim, const Real3d &delta, Handle &hand);

    /**
     * Solves for A x = b
     *
     *   2. Applies forward transform to f to obtain f̃.
     *   3. Solves diagonal system to obtain ũ.
     *   4. Applies inverse transform to obtain x (the output).
     * @param x Output buffer for the solution.
     * @param b Right-hand-side vector (will be overwritten).
     * @param hand
     */
    virtual void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const = 0;



    /**
     * This method computes the inverse of L.  It should only be used for debugging.  It is not efficient.
     * @param hand
     */
    virtual SquareMat<T> inverseL(Handle &hand) const;
};

#endif // EIGENDECOMPSOLVER_H
