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
     * @brief Factory method to create a SquareMat from memory managed elsewhere.  Don't call this method unless you
     * intend to mannage the memory yourself, which is almost never worth it.
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
     * @brief Factory method to create a SquareMat from memory managed elsewhere.  Don't call this method unless you
     * intend to mannage the memory yourself, which is almost never worth it.
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
     * Will allocated and free additional memory.
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
     * Only call if this is already LU decomposed!  todo: create sepereate lu decompposed object.
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
     * @param rowSwaps row swaps information for lu decomposition.
     * @param handle
     * @param info info for lu deocmposition.
     * @param buffer Get size infor for LU decomposition buffer size.
     * @param transpose True to transpose the lhs operator.
     */
    template<class Int>
    void inverse(SquareMat<T> &result, SimpleArray<Int> &rowSwaps, Handle &handle, Singleton<int32_t> &info, SimpleArray<T> &buffer, bool transpose);


    /**
     * Destroys/overwrites this matrix! Destroys rhs vector!
     * Solves the equation Ax = b
     * @param b The rhs of the equation.  It will be replaced  by the soltion, x.
     * @param handle
     * @param info
     * @param buffer
     * @param rowSwaps
     * @param transpose
     */
    template<class Int>
    void solve(Mat<T> &b, Handle &handle, Singleton<int32_t> &info, SimpleArray<T> &buffer, SimpleArray<Int> &rowSwaps, bool transpose);

    /**
     * Allocates and frees memory.  Gets the determinent using LU facotrization.  Destroys this matrix.
     * @param hand
     * @return
     */
    double determinant(Handle& hand) ;
    /**
     * Gets the determinent using LU factorization.  Destorys this matrix.
     * @param sizeOfNumRows Allocated memory.
     * @param info info about the LU decomposition will be stored here.
     * @param workSpaceForLUDecomp A workspace for the LU decomposition
     * @param handle
     * @return The deteminant.
     */
    double determinant(SimpleArray<int32_t> &sizeOfNumRows, Singleton<int32_t> &info, SimpleArray<T> &workSpaceForLUDecomp, Handle &handle);


    /**
     * Checks if the matrix is singular by looking at the eigen values.  Allocates its own memory for each run.
     * Will destroy this matrix.
     * @param tolerance
     * @param hand
     * @return true if the matrix is singular, false otherwie.
     */
    bool isSingular(double tolerance, Handle& hand) const ;

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
