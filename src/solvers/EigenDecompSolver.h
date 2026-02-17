#ifndef EIGENDECOMPSOLVER_H
#define EIGENDECOMPSOLVER_H

#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/SquareMat.h"
#include "deviceArrays/headers/Vec.h"
#include "../poisson/PoissonRHS.h"
#include "../deviceArrays/headers/Support/Streamable.h"

#include <cstddef>
#include <vector>

#include "Event.h"
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
     * Created an eigen decomposition solver where all memory is owned by this object.
     * @param dim The dimensions of grid.
     * @param delta The distance between fgrid points.
     * @param hand The handle.
     */
    EigenDecompSolver(const GridDim &dim, Real3d delta, Handle &hand);

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

template<typename T>
class EigenDecompSolver2d: public EigenDecompSolver<T> {
private:
    /**
     * @brief Compute ũ = f̃ / (λ_x + λ_y + λ_z).
     *
     * @param f  Input in eigen-space.
     * @param u  Output solution in eigen-space.
     * @param hand CUDA cuBLAS/cusolver handle.
     */
    void setUTilde(const Mat<T> &f, Mat<T> &u, Handle &hand) const;

    /**
     * Sets the eigen values.
     * @param hand2 Contexts for mutli threeading.
     * @param delta The distance between the gris points.
     * @param event An event to control stream flow.
     */
    void setEigens(Handle *hand2, Real2d delta, Event &event);

public:
    /**
     * @brief Creates an eigen decomposition solver for a 2D staggered MAC grid.
     * * This solver uses discrete sine/cosine transforms to invert the Laplacian.
     * * @param rowsXRowsP1 Matrix workspace for row-wise operations. Dimensions: [rows x rows + 1].
     * @param colsXColsP1 Matrix workspace for column-wise operations. Dimensions: [cols x cols + 1].
     * @note Optimization: If the grid is square (rows == cols), pass the same matrix
     * reference used for rowsXRows to reduce memory footprint.
     * @param sizeOfB Workspace vector. Must be the same size as the Eulerian Pressure grid (the system RHS).
     * @param hand2 Pointer to an array of at least two Handles for stream management.
     * @param delta The grid spacing (dx, dy).
     * @param event Event object used to synchronize and control multistreaming execution.
     */
    EigenDecompSolver2d(SquareMat<T> &rowsXRowsP1, SquareMat<T> &colsXColsP1, SimpleArray<T> &sizeOfB, Handle *hand2,
                        Real2d delta, Event &event);

    /**
     * Creates a solver that owns its own memory.
     * @param dim The dimensions of the grid.
     * @param hand2 2 contexts.
     * @param delta The distance between grid points.
     * @param event
     */
    EigenDecompSolver2d(GridDim dim, Handle *hand2, Real2d delta, Event &event);


    void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const;
};

template<typename T>
class EigenDecompSolver3d: public EigenDecompSolver<T> {


    /**
     * @brief Compute ũ = f̃ / (λ_x + λ_y + λ_z).
     *
     * @param f  Input in eigen-space.
     * @param u  Output solution in eigen-space.
     * @param hand CUDA cuBLAS/cusolver handle.
     */
    virtual void setUTilde(const Tensor<T> &f, Tensor<T> &u, Handle &hand) const;

    /**
     * @brief Multiply using E_i or E_iᵀ batched across layers.
     *
     * @param i Which eigenbasis to use (0=x,1=y,2=z).
     * @param transposeEigen Use E_iᵀ instead of E_i.
     * @param transpose Swap roles of left/right inputs in cuBLAS.  Set to true if the verctors in a1 need to be
     * transposed.  Otherwise, set to false.
     * @param a1 Input matrix batch.
     * @param dst1 Output matrix batch.
     * @param stride Matrix stride.
     * @param hand CUDA handle.
     * @param batchCount Number of batches.
     */
    void multE(size_t i, bool transposeEigen, bool transpose,
               const Mat<T> &a1, Mat<T> &dst1, size_t stride,
               Handle &hand, size_t batchCount) const;

    /** @brief Apply E_x or E_xᵀ across all z-layers. */
    void multEX(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const;

    /** @brief Apply E_y or E_yᵀ across all z-layers. */
    void multEY(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const;

    /** @brief Apply E_z or E_zᵀ across all x-y slices. */
    virtual void multEZ(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const;

    /**
     * @brief Apply full transform:
     *        f → E_zᵀ E_yᵀ E_xᵀ f    (forward)
     *        or
     *        u ← E_x E_y E_z ũ      (inverse)
     *
     * @param hand CUDA handle.
     * @param src Input 3D tensor.   Will be overwritten.
     * @param dst Output 3D tensor.
     * @param transposeE Whether to apply Eᵀ instead of E.
     */
    void multiplyEF(Handle &hand, const Tensor<T> &src, Tensor<T> &dst, bool transposeE) const;

    /**
     * Sets the eigen vectors and values.
     * @param hand3 Handles used to make the settings.
     * @param delta The distance between the grid points.
     * @param event3 3 Events to control the stream flow.
     */
    void setEigens(Handle *hand3, Real3d delta, Event *event3);
public:


    /**
     * @brief Creates an eigen decomposition solver for a 3D staggered MAC grid.
     * * @param rowsXRowsP1 Matrix workspace for X-direction operations. Dimensions: [rows x (rows + 1)].
     * @param colsXColsP1 Matrix workspace for Y-direction operations. Dimensions: [cols x (cols + 1)].
     * @note Optimization: If rows == cols, pass the same matrix as rowsXRows.
     * @param depthsXDepthsP1 Matrix workspace for Z-direction operations. Dimensions: [layers x (layers + 1)].
     * @note Optimization: If layers == rows or layers == cols, you may pass the corresponding
     * matrix used for those dimensions.
     * @param sizeOfB Workspace vector. Must be the same size as the Eulerian Pressure grid (the system RHS).
     * @param hand3 Pointer to an array of at least three Handles for concurrent 3D stream processing.
     * @param delta The grid spacing (dx, dy, dz).
     * @param event Pointer to an Event object or array used for multistream synchronization.
     */
    EigenDecompSolver3d(Mat<T> &rowsXRowsP1, Mat<T> &colsXColsP1, Mat<T> &depthsXDepthsP1, SimpleArray<T> &sizeOfB,
                        Handle *hand3, Real3d delta, Event *event);

    EigenDecompSolver3d(GridDim dim, Handle *hand3, Real3d delta, Event *event);

    void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const;

};

template<typename T>
class EigenDecompSolver3dThomas<T> : public EigenDecompSolver3d<T> {

    //TODO: eVals matrix should two cols.  There should only be two eVecs matrices.

    Tensor<T> workSpaceSuperPrime, workSpaceRHSPrime;
    double deltaZ;

    void multEZ(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE)  const override;
    void setUTilde(const Tensor<T> &src, Tensor<T> &dst, Handle &hand) const override;

public:
EigenDecompSolver3dThomas(SquareMat<T> &rowsXRows, SquareMat<T> &colsXCols, SquareMat<T> &depthsXDepths, Mat<T> &maxDimX3, SimpleArray<T> &sizeOfB, Handle *hand3, const Real3d &delta, Event *event, double delta_z);
};

#endif // EIGENDECOMPSOLVER_H
