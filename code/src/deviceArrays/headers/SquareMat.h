/**
 * @file SquareMat.h
 * @brief Defines the SquareMat class, a specialization of Mat for square matrices.
 * 
 * This class provides square-matrix-specific operations such as computing eigenvalues
 * and eigenvectors. It inherits from Mat<T> and uses GPU memory for storage.
 * @ingroup device_arrays
 */

#ifndef BICGSTAB_SQUAREMAT_H
#define BICGSTAB_SQUAREMAT_H

#include "Mat.h"
#include "GpuArray.h"


/**
 * @class SquareMat
 * @brief Represents a square matrix with GPU support.
 * 
 * SquareMat is a specialization of Mat<T> for square matrices. It provides
 * operations specific to square matrices, including computation of eigenvalues
 * and eigenvectors. Use the static create method to instantiate a SquareMat.
 * 
 * @tparam T The type of the matrix elements (e.g., float, double).
 */
template <typename T>
class SquareMat : public Mat<T> {
    friend class Mat<T>;

private:
    /**
     * @brief Private constructor for internal use.
     * 
     * Constructs a SquareMat with a given leading dimension and shared pointer to the data.
     * Users should typically use the static create() method instead.
     * 
     * @param rowsCols The number of rows and columns (matrix is square).
     * @param ld The leading dimension of the matrix storage.
     * @param _ptr Shared pointer to the underlying GPU memory.
     */
    SquareMat(size_t rowsCols, size_t ld, std::shared_ptr<T> _ptr);

public:
    using Mat<T>::factorLUBufferSize;
    /**
     * @brief Factory method to create a SquareMat of given size.
     * 
     * Allocates a square matrix of size `rowsCols x rowsCols` in GPU memory.
     * 
     * @param rowsCols The number of rows and columns (matrix is square).
     * @return A new SquareMat object.
     */
    static SquareMat<T> create(size_t rowsCols);

    /**
     * @brief Factory method to create a SquareMat from memory managed elsewhere.  Use this overload only when the device allocation is managed externally.
     *
     * Allocates a square matrix of size `rowsCols x rowsCols` in GPU memory.
     *
     * @param rowsCols The number of rows and columns (matrix is square).
     * @param ld the distance between the first elements of adjacent columns.
     * @param ptr The raw pointer to the data.  Memory is neither allocated or freed if you pass a raw pointer.
     * @return A new SquareMat object.
     */
    static SquareMat<T> create(size_t rowsCols, size_t ld, T* ptr);
    /**
     * @brief Factory method to create a SquareMat from memory managed elsewhere.  Use this overload only when the device allocation is managed externally.
     *
     * Allocates a square matrix of size `rowsCols x rowsCols` in GPU memory.
     *
     * @param rowsCols The number of rows and columns (matrix is square).
     * @param ld the distance between the first elements of adjacent columns.
     * @param ptr The raw pointer to the data.  Memory is neither allocated or freed if you pass a raw pointer.
     * @return A new SquareMat object.
     */
    static SquareMat<T> create(size_t rowsCols, size_t ld, const T* ptr);

    /**
     * @brief Factory method to create an empty SquareMat.
     *
     * @return A SquareMat<T> with zero rows, zero columns, zero leading dimension,
     *         and a null device pointer.
     *
     * @note No GPU memory is allocated.
     */
    static SquareMat<T> empty();


    /**
     * @brief Computes the eigenvalues and optionally the eigenvectors of the matrix.
     *
     * Will overwrite this matrix!
     * Allocates and frees additional memory.
     * 
     * @param eVals Vector to store the computed eigenvalues.
     * @param eVecs Pointer to a SquareMat to store eigenvectors, or nullptr if eigenvectors are not needed.
     * @param hand
     * @param hand
     * 
     * @note This function assumes the matrix is square.
     */
    void eigen(Vec<T> &eVals, SquareMat *eVecs, Handle& hand) const;

    /**
     * Sets this matrix to be the identity matrix.
     * @param stream
     */
    SquareMat setToIdentity(cudaStream_t stream);


    /**
     * @brief Solves the linear system $A\mathbf{x} = \mathbf{b}$ for $\mathbf{x}$. assuming $A$ is already factored
     *
     * @pre The matrix must already contain LU factors produced by factorLU.
     * @todo Consider introducing a dedicated LU-factorized matrix type.
     *
     * into $LU$.This method uses cuSOLVER's cusolverDn[D/S]getrs to perform the forward and backward substitution steps.
     * The solution $\mathbf{x}$ overwrites the right-hand side matrix $\mathbf{b}$.@warning Automatic Memory
     * Management (Leak-Free): If any pointer parameter (handle or info) is nullptr, the necessary resource is
     * automatically allocated on the device (or handle created), used, and then safely freed upon exit. Pre-allocate
     * parameters to persist or reuse the results.
     * @tparam T The element type (must be float or double).
     * @param b Input/Output Right-Hand Side: A constant reference to the Mat<T> representing the right-hand side(s)
     * $\mathbf{b}$. This matrix is overwritten with the solution $\mathbf{x}$.
     * @param rowSwaps Input Pivot Array: A non-const reference to the Vec<int32_t> containing the row pivot
     * indices (the exact output from the preceding factorLU call).
     * @param handle Input/Output Handle: A pointer to an existing Handle (cuSOLVER/cuBLAS handle).
     * @param info Input/Output Status Flag: A device Singleton<int32_t> for the status. $0$ is success; $i>0$ means
     * the system is singular.
     * @param transpose Input Transposition Flag: If true, solves $A^T\mathbf{x} = \mathbf{b}$ (uses CUBLAS_OP_T);
     * otherwise, solves $A\mathbf{x} = \mathbf{b}$ (uses CUBLAS_OP_N).
     * @pre The current matrix (this) must contain the $LU$ factors produced by factorLU.
     * @pre The dimensions must match: this->_rows (N) must equal rowSwaps.size() and b._rows.
     * @post The matrix b contains the solution vector(s) $\mathbf{x}$.
     */
    template<class Int>
    void solveLUDecomposed(Mat<T> &b, Vec<Int> &rowSwaps, Handle &handle, Singleton<int32_t> &info, bool transpose);



    /**
     * Destroys/overwrites this matrix.  Computes the inverse of this matrix.
     * @tparam Int
     * @param result The inverse will be placed here.
     * @param rowSwaps row swaps informationmation for lu decomposition.
     * @param handle
     * @param info info for lu decomposition.
     * @param buffer Get size information for LU decomposition buffer size.
     * @param transpose True to transpose the lhs operator.
     */
    template<class Int>
    void inverse(SquareMat<T> &result, SimpleArray<Int> &rowSwaps, Handle &handle, Singleton<int32_t> info, SimpleArray<T> &buffer, bool transpose);


    /**
     * Destroys/overwrites this matrix! Destroys rhs vector!
     * Solves the equation Ax = b
     * @param b The rhs of the equation.  It will be replaced  by the solution, x.
     * @param handle
     * @param info
     * @param buffer
     * @param rowSwaps
     * @param transpose
     */
    template<class Int>
    void solve(Mat<T> &b, Handle &handle, Singleton<int32_t> &info, SimpleArray<T> &buffer, SimpleArray<Int> &rowSwaps, bool transpose);

    /**
     * Allocates and frees memory.  Gets the determinent using LU factorization.  Destroys this matrix.
     * @param hand
     * @return
     */
    double determinant(Handle& hand) ;
    /**
     * Gets the determinent using LU factorization.  Destroys this matrix.
     * @param sizeOfNumRows Allocated memory.
     * @param info info about the LU decomposition will be stored here.
     * @param workSpaceForLUDecomp A workspace for the LU decomposition
     * @param handle
     * @return The determinant.
     */
    double determinant(SimpleArray<int32_t> &sizeOfNumRows, Singleton<int32_t> &info, SimpleArray<T> &workSpaceForLUDecomp, Handle &handle);


    /**
     * Checks if the matrix is singular by looking at the eigen values.  Allocates its own memory for each run.
     * Will destroy this matrix.
     * @param tolerance
     * @param hand
     * @return true if the matrix is singular, false otherwise.
     */
    bool isSingular(double tolerance, Handle& hand) const ;


    /**
     * @brief Query the device- and host-workspace sizes required by eigenSPDFromBuffer().
     *
     * This is the first of a two-phase eigendecomposition API for symmetric
     * positive (semi-)definite matrices.  Call this once, allocate the buffers it
     * returns, then pass them to eigenSPDFromBuffer() to perform the actual
     * computation without any internal heap allocation.
     *
     * The split is modelled on the existing factorLUBufferSize() / factorLU()
     * pattern in Mat.cu so that callers that run many decompositions in a loop
     * can reuse the same workspace across calls.
     *
     * @tparam T  Floating-point element type.  Must be @c float or @c double;
     *            calling with any other type throws @c std::invalid_argument.
     *
     * @param hand  Active cuSOLVER handle.
     *
     * @return A pair whose @c first element is the required device-workspace size
     *         in elements of @c T, and whose @c second element is the required
     *         host-workspace size in elements of @c T.
     *
     * @throws std::invalid_argument  If @c T is not @c float or @c double.
     * @throws std::runtime_error     If the cuSOLVER buffer-size query fails.
     */
    std::pair<size_t, size_t> eigenSPDBufferSize(Handle &hand) const;

    /**
     * @brief Compute the eigendecomposition of a symmetric positive (semi-)definite
     *        matrix using caller-supplied workspaces.
     *
     * This is the compute half of the two-phase API.  The caller must first invoke
     * eigenSPDBufferSize() to determine the required buffer sizes, allocate them,
     * and then call this method.
     *
     * @par Mathematical requirements
     * The matrix @b must satisfy all of the following conditions:
     *  - **Square**: rows == cols.
     *  - **Real**: complex-valued matrices are not supported.
     *  - **Symmetric**: only the lower triangle is read; the upper triangle is
     *    ignored.  If the matrix is not actually symmetric, results are undefined.
     *  - **Positive semi-definite**: all eigenvalues must be ≥ 0.  The cuSOLVER
     *    symmetric solver (dsytd2 / ssytd2 via cusolverDnXsyevd) is numerically
     *    stable for PSD matrices, but may produce incorrect or negative eigenvalues
     *    if the matrix is indefinite or has significant numerical asymmetry.
     *
     * @par In-place operation
     * cusolverDnXsyevd overwrites @p *this with the eigenvectors.  After the call,
     * column @c j of @p *this contains the unit eigenvector corresponding to
     * eigenvalue @c eVals[j].  If you need to preserve the original matrix, copy
     * it before calling this method.
     *
     * @tparam T  Floating-point element type.  Must be @c float or @c double.
     *
     * @param eVals         Output vector of length @c n.  On return, holds the
     *                      eigenvalues in ascending order.  Must already be
     *                      allocated to at least @c this->_rows elements.
     * @param hand          Active cuSOLVER handle.
     * @param deviceBuffer    Pre-allocated device workspace.  Must be at least as
     *                      large as the device element count returned by
     *                      eigenSPDBufferSize().
     * @param hostBufferSize
     * @param hostBufferSize
     * @param buffer      Pre-allocated host workspace.  Must be at least as
     *                      large as the host element count returned by
     *                      eigenSPDBufferSize().
     * @param info
     *
     * @throws std::invalid_argument  If @p eVals is too small, or if @c T is
     *                                unsupported.
     * @throws std::runtime_error     On cuSOLVER failure, with an info code
     *                                forwarded through processInfo().
     */
    void eigenSPD(Vec<T> &eVals, Handle &hand, SimpleArray<T> &deviceBuffer, T *hostBuffer, size_t hostBufferSize, Singleton<int32_t> &
                  info);

    /**
     * @brief Convenience wrapper: compute the eigendecomposition of a symmetric
     *        positive (semi-)definite matrix, allocating all workspaces internally.
     *
     * This method combines eigenSPDBufferSize() and eigenSPDFromBuffer() into a
     * single call for callers that do not need to amortise workspace allocation
     * across multiple decompositions.  For hot paths, prefer the two-phase API.
     *
     * @par Mathematical requirements
     * Identical to eigenSPDFromBuffer().  In summary:
     *  - The matrix must be **square**, **real**, **symmetric** (lower triangle
     *    used), and **positive semi-definite**.
     *  - The matrix is overwritten with eigenvectors in-place by cuSOLVER.
     *    Pass a non-null @p eVecs if you need the original matrix preserved.
     *
     * @tparam T  Floating-point element type.  Must be @c float or @c double.
     *
     * @param eVals  Output vector of length @c n.  Receives eigenvalues in
     *               ascending order.  Must be pre-allocated to at least @c n
     *               elements.
     * @param hand   Active cuSOLVER handle.
     *
     * @throws std::invalid_argument  If @p eVals is too small, or @c T is
     *                                unsupported.
     * @throws std::runtime_error     On cuSOLVER failure.
     */
    void eigenSPD(Vec<T> &eVals, Handle &hand);

    /**
     * Checks if the matrix is singular by looking at the eigen values.  Will destory this matrix.
     * @param tolerance
     * @param buffer2n A buffer with twice as many elements as this matrix has rows.
     * @param buffeNXN A buffer the same size as this matrix.
     * @param hand
     * @return true if this matrix is singular, false otherwise.
     */
    bool isSingular(double tolerance, SimpleArray<int32_t> &rowSwaps, Singleton<int32_t> &info, SimpleArray<T> &workSpace, Handle &hand);



};

#endif //BICGSTAB_SQUAREMAT_H
