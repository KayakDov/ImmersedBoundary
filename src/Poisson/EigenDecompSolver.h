#ifndef EIGENDECOMPSOLVER_H
#define EIGENDECOMPSOLVER_H

#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/SquareMat.h"
#include "deviceArrays/headers/Vec.h"
#include "Poisson.h"
#include "deviceArrays/headers/Streamable.h"

#include <array>
#include <cstddef>

/**
 * @brief Direct Poisson solver using eigen-decomposition (Fast Diagonalization Method).
 *
 * This class diagonalizes the 3D discrete Laplacian operator using the
 * Kronecker structure:
 *
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
class EigenDecompSolver : public Poisson<T> {
private:
    /**
     * @brief Eigenvector matrices for the three 1-D Laplacians.
     *
     * eVecs[0] = eigenvectors of L_x
     * eVecs[1] = eigenvectors of L_y
     * eVecs[2] = eigenvectors of L_z
     */
    std::array<SquareMat<T>, 3> eVecs;

    /**
     * @brief Eigenvalues matrix.
     *
     * Column 0: eigenvalues of L_x
     * Column 1: eigenvalues of L_y
     * Column 2: eigenvalues of L_z
     */
    Mat<T> eVals;

    /**
     * @brief Compute eigenvalues and eigenvectors for L[i].
     *
     * @param i Index (0=x, 1=y, 2=z).
     * @param stream CUDA stream to execute kernels on.
     */
    void eigenL(size_t i, cudaStream_t stream);

    /**
     * @brief Compute ũ = f̃ / (λ_x + λ_y + λ_z).
     *
     * @param f  Input in eigen-space.
     * @param u  Output solution in eigen-space.
     * @param hand CUDA cuBLAS/cusolver handle.
     */
    void setUTilde(const Tensor<T> &f, Tensor<T> &u, Handle &hand);

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
               Handle &hand, size_t batchCount);

    /** @brief Apply E_x or E_xᵀ across all z-layers. */
    void multEX(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE);

    /** @brief Apply E_y or E_yᵀ across all z-layers. */
    void multEY(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE);

    /** @brief Apply E_z or E_zᵀ across all x-y slices. */
    void multEZ(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE);

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
    void multiplyEF(Handle &hand, Tensor<T> &src,
                    Tensor<T> &dst, bool transposeE);

public:
    /**
     * @brief Construct and immediately solve the Poisson problem.
     *
     * The constructor:
     *   1. Builds eigenbases for Lx, Ly, Lz.
     *   2. Applies forward transform to f to obtain f̃.
     *   3. Solves diagonal system to obtain ũ.
     *   4. Applies inverse transform to obtain x (the output).
     *
     * @param boundary Boundary conditions for the grid.
     * @param x Output buffer for the solution.
     * @param f Right-hand-side vector (will be overwritten).
     * @param rowsXRows A space to work in.
     * @param colsXCols A space to work in.
     * @param depthsXDepths A space to work in.
     * @param maxDimX3 A space to work in.
     * @param hand 3 CUDA cuBLAS/cusolver handles.
     */
    EigenDecompSolver(const CubeBoundary<T> &boundary, Vec<T> &x, Vec<T> &f, SquareMat<T> &rowsXRows,
                      SquareMat<T> &colsXCols, SquareMat<T> &depthsXDepths, Mat<T> &maxDimX3,
                      std::array<Handle, 3>& hand3);

    /**
     * @brief Construct and immediately solve the Poisson problem.
     *
     * The constructor:
     *   1. Builds eigenbases for Lx, Ly, Lz.
     *   2. Applies forward transform to f to obtain f̃.
     *   3. Solves diagonal system to obtain ũ.
     *   4. Applies inverse transform to obtain x (the output).
     *
     * Orientation of the boundaries is as follows.  The front and back boundaries have there first row agains the top,
     * and first column against the left.  The left and right boundaries each have their first row against the top, and
     * first column against the back.  The top and bottom boundaries each have their first row against the back
     * and first column against the left.
     *
     * Each pair of matrices that are stored together have the 2nd matrix of the pair stored beneath the first,
     * so when thought of as a single matrix with two submatrices, the first half of each column belongs to the first
     * sub matrix, and the second half of each column belongs to the second sub matrix.
     *
     * This constructor is meant to be run as a fortran method.
     *
     * @param frontBack A pointer to the device front and back boundaries.  The back boundary matrix should be below the
     * front boundary matrix.
     * @param fbLd The leading dimension of the frontBack matrix.  The distance between the first element of each column.
     * @param leftRight
     * @param lrLd
     * @param topBottom
     * @param tbLd
     * @param x Output buffer for the solution. No padding is permitted.
     * @param xStride The distance between elements of the output data.
     * @param height Height of the grid.
     * @param width Width of the grid.
     * @param depth Depth of the grid.
     * @param f Right-hand-side of the Poisson equation (will be overwritten).  No padding is permitted.
     * @param fStride The distance between elements of the f vector.
     * @param rowsXRows A space to work in.
     * @param rowsXRowsLd
     * @param colsXCols A space to work in.
     * @param colsXColsLd
     * @param depthsXDepths A space to work in.
     * @param depthsXDepthsLd
     * @param maxDimX3 A space to work in.
     * @param maxDimX3Ld
     */
     static void solve(T *frontBack, size_t fbLd, T *leftRight, size_t lrLd, T *topBottom, size_t tbLd,
                       T *f, size_t fStride, T *x, size_t xStride,
                       size_t height, size_t width, size_t depth,
                       T *rowsXRows, size_t rowsXRowsLd,
                       T *colsXCols, size_t colsXColsLd,
                       T *depthsXDepths, size_t depthsXDepthsLd,
                       T *maxDimX3, size_t maxDimX3Ld
     );
};


#endif // EIGENDECOMPSOLVER_H
