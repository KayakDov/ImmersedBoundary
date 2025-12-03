//
// Created by dov on 10/10/25.
//

#ifndef BICGSTAB_BANDEDMAT_H
#define BICGSTAB_BANDEDMAT_H

#include "Vec.h"
#include "Singleton.h"
#include <array>
#include <vector>

/**
 * @brief Represents a square matrix stored in banded (diagonal) format.
 *
 * Each stored column corresponds to a matrix diagonal. The @c _indices vector defines
 * which diagonal each column represents:
 *
 *  - @c 0  → main (primary) diagonal.
 *  - @c >0 → superdiagonals. The value is the column offset of the first element (row 0).
 *  - @c <0 → subdiagonals. The (absolute) value is the row offset of the first element (column 0).
 *
 * Internally, all diagonals have appended padding to match the length of the primary diagonal.
 * Padding elements hold unused values.
 *
 *There are two matrices here, the sparse matrix and the dense matrix.  The sparse matrix has only elements on the diagonals
 *in the indices vector.  The dense matrix's columns are the sparse matrices diagonals.  All inherited methods here
 *treat this matrix as the dense matrix.  Methods overwridden and implemented here treat this matrix as a sparse matrix.
 *
 * So for example, to set a diagonal of the sparse matrix one would call col(i).set(setTo).  The sparse diagonal at indices[i]
 * would then have values setTo.
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
    /**
     * Constructor.
     * @param rows The number of rows is the length of the longest diagonal.
     * @param cols The number of columns ie diagonals.
     * @param ld The distance between the first element of each row.
     * @param ptr  Be sure this is prealocated memory with a destruction plan.
     * @param indices Each values is the index of the coresponding row.
     */
    BandedMat(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> ptr, const Vec<int32_t> &indices);

public:
    /**const
    * @brief Constructs a banded matrix by wrapping an existing dense matrix buffer.
    *
    * @param copyFrom Existing matrix (device or host-backed) whose memory and dimensions
    *                 are adopted without deep copying.
    * @param indices  Diagonal index vector defining which diagonal each row corresponds to.
    *
    * @note The data pointer is shared; no allocation or element copying is performed.
    */
    BandedMat(const Mat<T> &copyFrom, const Vec<int32_t> &indices);

    /**
    * @brief Allocates a new banded matrix on device memory.
    *
    * @param denseSqMatDim          Dimension of the original square matrix (number of columns).
    * @param numDiagonals  Number of diagonals to store (height of the banded matrix).
    * @param indices       Vector of size @p numDiagonals specifying diagonal offsets.
    *
    * @return A newly allocated @c BandedMat with zero-initialized data.
    */
    static BandedMat create(size_t denseSqMatDim, size_t numDiagonals, const Vec<int32_t> &indices);

    /**
     *
     * @param denseSqMatDim The height and width of the dense square matrix.
     * @param numDiagonals The number of diagonals.
     * @param ld The number of elements between the first element of each diagonal.  Diagonals are stored consecutively.
     * @param data The values in the diagonals.
     * @param indices The indices of each diagonal.  The offeset from the primary diagonal.
     * @param indsStride The stride of the indices data.
     * @return A vanded matrix.  Memory management must be handled externally for banded matrices created here.
     */
    static BandedMat create(size_t denseSqMatDim, size_t numDiagonals, size_t ld, const T* data, const int32_t* indices, size_t indsStride);

    /**
    * @brief Extracts diagonals from a dense square matrix and writes them
    *        into the columns of this matrix.  Be sure the indices set for this matrix match the diagonals of the
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
    void bandedMult(const Vec<T> &other, Vec<T> &result, Handle *handle, const
                    Singleton<T> alpha = Singleton<T>::ONE, const Singleton<T> beta = Singleton<T>::ZERO, bool
                    transpose = false) const;


    /**
     * Creates a matrix with lots of 0s stored off the diagonals of ineterst.
     * @param dense Where the matrix with all the zeroes is stored.  It should be _rows x _rows.
     * @param handle
     */
    void getDense(SquareMat<T> dense, Handle *handle = nullptr) const;
};


#endif //BICGSTAB_BANDEDMAT_H
