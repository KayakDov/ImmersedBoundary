/**
 * @file BandedMat.h
 * @brief Defines banded sparse matrices stored as diagonal columns.
 * @ingroup sparse_matrices
 *
 * @details
 * Sparse-matrix classes build on the device array layer and expose storage-format-specific operations while preserving explicit CUDA handle and stream control.
 */

#ifndef BICGSTAB_BANDEDMAT_H
#define BICGSTAB_BANDEDMAT_H

#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/Singleton.h"


template <typename T> class Vec;
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
 * @details
 * BandedMat exposes two complementary interpretations of the same storage.
 * The dense representation stores one diagonal per column. The corresponding
 * sparse matrix contains only the diagonals listed in the index vector.
 * Inherited Mat operations act on the dense representation; operations
 * overridden by BandedMat interpret the object as a sparse banded matrix.
 *
 * To set a sparse diagonal, set the corresponding dense column, for example
 * @c col(i).set(values), where @c _indices[i] identifies the diagonal offset.
 *
 * @tparam T Numeric type stored in the banded matrix.
 *
 * @note No implicit deep copying is performed. Underlying GPU memory is shared through
 *       @c std::shared_ptr. Destruction only frees memory when the final reference is released.
 */

template<typename T>
class BandedMat : public Mat<T> {

protected:
    /**
     * Constructor.
     * @param rows The number of rows is the length of the longest diagonal.
     * @param cols The number of columns ie diagonals.
     * @param ld The distance between the first element of each row.
     * @param ptr  Be sure this is preallocated memory with a destruction plan.
     * @param indices Each values is the index of the corresponding row.
     */
    BandedMat(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> ptr, const Vec<int32_t> &indices);

public:

    const Vec<int32_t> _indices;

    /**const
    * @brief Constructs a banded matrix by wrapping an existing dense matrix buffer.
    *
    * @param windowTo Existing matrix (device or host-backed) whose memory and dimensions
    *                 are adopted without deep copying.
    * @param indices  Diagonal index vector defining which diagonal each row corresponds to.
    *
    * @note The data pointer is shared; no allocation or element copying is performed.
    */
    BandedMat(const Mat<T> &windowTo, const Vec<int32_t> &indices);

    /**
    * @brief Allocates a new banded matrix on device memory.
    *
    * @param denseSqMatDim          Dimension of the original square matrix (number of columns).
    * @param indices       Vector of size @p numDiagonals specifying diagonal offsets.
    *
    * @return A newly allocated @c BandedMat with zero-initialized data.
    */
    static BandedMat create(size_t denseSqMatDim, const Vec<int32_t> &indices);

    /**
     *
     * @param denseSqMatDim The height and width of the dense square matrix.
     * @param numDiagonals The number of diagonals.
     * @param ld The number of elements between the first element of each diagonal.  Diagonals are stored consecutively.
     * @param data The values in the diagonals.
     * @param indices The indices of each diagonal.  The offset from the primary diagonal.
     * @param indsStride The stride of the indices data.
     * @return A banded matrix.  Memory management must be handled externally for banded matrices created here.
     */
    static BandedMat create(size_t denseSqMatDim, size_t numDiagonals, size_t ld, T* data, int32_t* indices, size_t indsStride);

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

    void generateEigenStedc(Handle &hand, SquareMat<T> eVecs, Vec<T> &eVals);

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
                    Singleton<T> alpha = GPUScalar<T>::get(1), const Singleton<T> beta = GPUScalar<T>::get(0), bool
                    transpose = false) const;

    void bandedMult(const Mat<T> &other, Mat<T> &result, Handle *handle, Singleton<T> alpha, Singleton<T> beta,
                    bool transpose) const;


    /**
     * Creates a matrix with lots of 0s stored off the diagonals of ineterst.
     * @param dense Where the matrix with all the zeroes is stored.  It should be _rows x _rows.
     * @param hand
     */
    void getDense(SquareMat<T> dense, Handle &hand) const;

    /**
     * This method allocates its own ememory.
     * @param hand
     * @return A dense version of this matrix.
     */
    SquareMat<T> getDense(Handle &hand) const;
};


#endif //BICGSTAB_BANDEDMAT_H
