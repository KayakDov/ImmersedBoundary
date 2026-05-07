#ifndef EIGENDECOMPSOLVER_H
#define EIGENDECOMPSOLVER_H

#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/SquareMat.h"
#include "deviceArrays/headers/Vec.h"
#include "../Event.h"
#include "poisson/Laplacian.cuh"


template<typename T>
class Set0Avg {
    Vec<T> ones;
public:
    Set0Avg(Vec<T> sizeOfArrays, Handle& hand);
    Set0Avg();

    void operator()(Vec<T> needsAverageSet, Handle &hand) const;
};


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

    /**
     * A workspace the size of b = L_cols.  You may store b itself here, but it will be overwritten.
     */
    mutable SimpleArray<T> sizeOfB;

public:
    /**
     * The eigen vectors and values.
     */
    poisson::Eigen<T> eigen;

    /**
     * The dimensions of the grid.
     */
    const GridDim dim;
    const bool isSingular;

    virtual ~EigenDecompSolver() = default;

    /**
     * Sets the average of an array to 0
     * @param src The array that will be translated by its average.
     * @param dst
     * @param bufferSizeOfB
     * @param hand
     */
    void set0Avg(const Vec<T> &src, Vec<T> &dst, Vec<T> &bufferSizeOfB, Handle &hand) const;




    /**
     * @brief Construct and immediately solve the Poisson problem.
     *
     * Builds eigenbases for Lx, Ly, Lz, Where L is the left hand side matrix you'd use for solving the Poisson equation.
     * It's a banded matrix with 7 diagonals, etc...
     *
     * A must be the standard second-difference (Toeplitz) discrete Laplacian on a uniform grid with homogeneous Dirichlet boundary conditions.
     *
     * These matrices will be overwritten.
     * @param eMatsAndVecs The eigen matrices and values for the laplacian.
     * @param sizeOfB An array the size of b = xLength * yLength * zLength that will be overwritten.  You may use b for this.
     * @param isSingular set to true if singular.
     */
    EigenDecompSolver(const poisson::Eigen<T> &eMatsAndVecs, SimpleArray<T> &sizeOfB, bool isSingular);
    /**
     *
     * @param boundary The boundary conditions.
     * @param hands A handle for each dimension.
     * @param events If 3d, then 2 events, if 2d then 1 event.
     */
    EigenDecompSolver(const BoundaryConfig<T> &boundary, Handle *hands, Event *events, SimpleArray<T> sizeOfB);

    /**
     * Created an eigen decomposition solver where all memory is owned by this object.
     * @param boundary The boundary conditions.
     * @param hands A handle for each dimension.
     * @param events 2 for 3d and 1 for 2d.
     */
    EigenDecompSolver(const BoundaryConfig<T> &boundary, Handle *hands, Event *events);

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
