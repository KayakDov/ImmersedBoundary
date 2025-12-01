/**
 * @file SquareMat.h
 * @brief Defines the SquareMat class, a specialization of Mat for square matrices.
 * 
 * This class provides square-matrix-specific operations such as computing eigenvalues
 * and eigenvectors. It inherits from Mat<T> and uses GPU memory for storage.
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
     * @brief Computes the eigenvalues and optionally the eigenvectors of the matrix.
     * 
     * @param eVals Vector to store the computed eigenvalues.
     * @param eVecs Pointer to a SquareMat to store eigenvectors, or nullptr if eigenvectors are not needed.
     * @param temp Optional temporary Mat used for internal computations (can be nullptr).  It should be the same size
     * as this matrix.
     * @param handle Optional GPU Handle for cuBLAS/cuSOLVER operations (can be nullptr to create a temporary handle).
     * 
     * @note This function assumes the matrix is square.
     */
    void eigen(Vec<T> &eVals, SquareMat<T> *eVecs, Mat<T> *temp = nullptr, Handle *handle = nullptr) const;

    /**
     * Sets this matrix to be the identity matrix.
     * @param stream
     */
    SquareMat setToIdentity(cudaStream_t stream);


    /**
     * @brief Solves the linear system $A\mathbf{x} = \mathbf{b}$ for $\mathbf{x}$, assuming $A$ is already factored
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
     * @post The matrix b contains the solution vector(s) $\mathbf{x}$.*/
    void solveLUDecomposed(Mat<T> &b, Vec<int32_t> &rowSwaps, Handle *handle = nullptr, Singleton<int32_t> *info = nullptr, bool transpose = false);



    void solve(Mat<T> &b, Handle *handle = nullptr, Singleton<int32_t> *info = nullptr, Vec<T> *workspace = nullptr, Vec<int32_t> *
                       rowSwaps = nullptr);

    void solve(Vec<T> &b, Handle *handle = nullptr, Singleton<int32_t> *info = nullptr, Vec<T> *workspace = nullptr, Vec<int32_t> *
                       rowSwaps = nullptr);

};

#endif //BICGSTAB_SQUAREMAT_H
