//
// Created by dov on 10/10/25.
//

#ifndef BICGSTAB_BANDEDMAT_H
#define BICGSTAB_BANDEDMAT_H

#include "vec.h"

/**
 * @brief Represents a square matrix stored in banded (diagonal) format.
 *
 * Each stored row corresponds to a matrix diagonal. The @c _indices vector defines
 * which diagonal each row represents:
 *
 *  - @c 0  → main (primary) diagonal.
 *  - @c >0 → superdiagonals. The value is the column offset of the first element (row 0).
 *  - @c <0 → subdiagonals. The value is the row offset of the first element (column 0).
 *
 * Internally, all diagonals have appended padding to match the length of the primary diagonal.
 * Padding elements hold unused values.
 *
 * @tparam T Numeric type stored in the banded matrix.
 *
 * @note No implicit deep copying is performed. Underlying GPU memory is shared through
 *       @c std::shared_ptr. Destruction only frees memory when the final reference is released.
 */

template<typename T>
class BandedMat final : public Mat<T> {
private:
    const Vec<int32_t> _indices;
protected:
    BandedMat(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> ptr, const Vec<int32_t> &indices);
public:
    /**
    * @brief Constructs a banded matrix by wrapping an existing dense matrix buffer.
    *
    * @param copyFrom Existing matrix (device or host-backed) whose memory and dimensions
    *                 are adopted without deep copying.
    * @param indices  Diagonal index vector defining which diagonal each row corresponds to.
    *
    * @note The data pointer is shared; no allocation or element copying is performed.
    */
    BandedMat(const Mat<T>& copyFrom, const Vec<int32_t>& indices);

    /**
    * @brief Allocates a new banded matrix on device memory.
    *
    * @param numDiagonals  Number of diagonals to store (height of the banded matrix).
    * @param cols          Dimension of the original square matrix (number of columns).
    * @param indices       Vector of size @p numDiagonals specifying diagonal offsets.
    *
    * @return A newly allocated @c BandedMat with zero-initialized data.
    */
    static BandedMat create(size_t numDiagonals, size_t cols, const Vec<int32_t> &indices);

    /**
    * @brief Extracts diagonals from a dense square matrix and writes them
    *        into this banded format on the GPU.  Be sure the indices set for this matrix match the diagonals of the
    *        square matrix passed here.
    *
    * @param denseMat Source dense @c SquareMat to read from.
    * @param handle   CUDA/cuBLAS handle providing stream and library contexts.
    *
    * @note This method launches a CUDA kernel. Elements outside each diagonal
    *       (padding) are filled with @c NaN.
    */
    void setFromDense(const SquareMat<T> &denseMat, Handle *handle);

    /**
    * @brief Multiplies this banded matrix with a vector: @f$ y = α A x + β y @f$.
    *
    * @param other   Input vector @c x.
    * @param result  Optional preallocated output vector @c y. If null, a new vector is created.
    * @param handle  Optional CUDA/cuBLAS handle. If null, a temporary handle may be created.
    * @param alpha   Scalar multiplier for @c A*x. Defaults to 1 when a nullptr is passed.
    * @param beta    Scalar multiplier for @c y. Defaults to 0 when a nullptr is passed.
    * @param transpose If true, computes @c Aᵗ * x instead of @c A * x.
    *
    * @return Result vector on device memory.
    */
    Vec<T> mult(const Vec<T>& other, Vec<T>* result = nullptr, Handle* handle = nullptr, const Singleton<T> *alpha = nullptr, const Singleton<T> *beta = nullptr, bool transpose = false) const override;

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    Mat<T> mult(const Mat<T>& other, Mat<T>* result = nullptr, Handle* handle = nullptr, const Singleton<T> *alpha = nullptr, const
                Singleton<T> *beta = nullptr, bool transposeA = false, bool transposeB = false) const;

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    void getDense(SquareMat<T> dense, Handle *handle = nullptr) const;

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    Mat<T> operator*(const Mat<T>& other) const override;

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    Mat<T> plus(const Mat<T>& x, Mat<T>* result = nullptr, const Singleton<T>* alpha = nullptr, const Singleton<T>* beta = nullptr, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr) override;

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    Mat<T> minus(const Mat<T>& x, Mat<T>* result = nullptr, const Singleton<T>* alpha = nullptr, const Singleton<T>* beta = nullptr, bool transposeA = false, bool transposeB = false, Handle* handle = nullptr) override;

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    void mult(const Singleton<T>& alpha, Handle* handle = nullptr) override;

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    void transpose(Mat<T>& result, Handle* handle = nullptr) const override;

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    void transpose(Handle* handle = nullptr, Mat<T>* preAlocatedMem = nullptr) override;

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    static Mat<T> create(size_t rows, size_t cols);

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    [[nodiscard]] Mat<T> subMat(size_t startRow, size_t startCol, size_t height, size_t width) const override;

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    Vec<T> col(size_t index) override;

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    Vec<T> row(size_t index) override;

    /**
    * @note This method is declared but not yet implemented. Calling it will result
    *       in undefined behavior until implementation is completed.
    */
    void normalizeCols(size_t setRowTo1, Handle* handle = nullptr) override;
};


#endif //BICGSTAB_BANDEDMAT_H