/**
 * @file SquareMat.h
 * @brief Defines the SquareMat class, a specialization of Mat for square matrices.
 * 
 * This class provides square-matrix-specific operations such as computing eigenvalues
 * and eigenvectors. It inherits from Mat<T> and uses GPU memory for storage.
 */

#ifndef BICGSTAB_SQUAREMAT_H
#define BICGSTAB_SQUAREMAT_H

#include "deviceArrays.h"

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
     * @param temp Optional temporary Mat used for internal computations (can be nullptr).
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
};

#endif //BICGSTAB_SQUAREMAT_H
