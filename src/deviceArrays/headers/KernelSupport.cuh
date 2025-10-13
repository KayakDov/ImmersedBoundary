#ifndef BICGSTAB_KERNELSUPPORT_CUH
#define BICGSTAB_KERNELSUPPORT_CUH

#include <cstddef>
#include <cstdint>

/**
 * @brief Helper class to map a row/column in a banded matrix to the corresponding element
 *        in a dense square matrix.
 *
 * DenseInd is intended for use inside CUDA device code. Given a banded matrix row index
 * and a column index, along with the banded matrix's diagonal indices, it computes:
 * - The diagonal offset (`d`)
 * - The row and column of the corresponding element in the dense matrix.
 *
 * Provides bounds checking and flat indexing suitable for linear memory access.
 */
class DenseInd {
public:
    const int32_t d;    ///< Diagonal index for the current banded row. 0 = main diagonal, >0 = upper, <0 = lower
    const int32_t row;  ///< Row index in the dense matrix corresponding to this banded row
    const int32_t col;  ///< Column index in the dense matrix corresponding to this banded row

    /**
     * @brief Constructor for DenseInd.
     * @param bandedRow Row index in the banded matrix
     * @param bandedCol Column index in the banded matrix
     * @param indices Pointer to the array of diagonal indices for the banded matrix
     *
     * Computes the corresponding row and column in the dense square matrix.
     * @note This function is intended for device code (__device__).
     */
    __device__ DenseInd(size_t bandedRow, size_t bandedCol, const int32_t* indices):
        d(indices[bandedRow]),
        row(static_cast<int32_t>(d > 0 ? bandedCol : bandedCol - d)),
        col(static_cast<int32_t>(d > 0 ? bandedCol + d : bandedCol))
    {}

    /**
     * @brief Checks if the dense matrix indices are out of bounds.
     * @param max Size of the square dense matrix (number of rows/columns)
     * @return True if either row or column is outside [0, max-1], false otherwise
     */
    __device__ bool outOfBounds(size_t max) const {
        return row < 0 || row >= max || col < 0 || col >= max;
    }

    /**
     * @brief Returns the flattened index for linear memory access in column-major order.
     * @param denseLd Leading dimension (stride) of the dense matrix
     * @return Flattened index suitable for accessing dense matrix memory
     */
    __device__ size_t flat(const size_t denseLd) const {
        return col * denseLd + row;
    }
};

#endif //BICGSTAB_KERNELSUPPORT_CUH
