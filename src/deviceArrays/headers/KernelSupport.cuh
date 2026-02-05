#ifndef BICGSTAB_KERNELSUPPORT_CUH
#define BICGSTAB_KERNELSUPPORT_CUH

#include <cstddef>
#include <cstdint>

#include "Support/GridDim.hpp"

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
class DenseInd : public GridInd2d{
public:
    const int32_t d;    ///< Diagonal index for the current banded row. 0 = main diagonal, >0 = upper, <0 = lower

    /**
     * @brief Constructor for DenseInd.
     * @param bandedRow Row index in the banded matrix
     * @param bandedCol Column index in the banded matrix
     * @param indices Pointer to the array of diagonal indices for the banded matrix
     *
     * Computes the corresponding row and column in the dense square matrix.
     * @note This function is intended for device code (__device__).
     */
    __device__ DenseInd(const GridInd2d bandedInd, const int32_t* indices):
        GridInd2d(
            static_cast<int32_t>(indices[bandedInd.col] > 0 ? bandedInd.row : bandedInd.row - indices[bandedInd.col]),
            static_cast<int32_t>(indices[bandedInd.col] > 0 ? bandedInd.row + indices[bandedInd.col] : bandedInd.row)),
        d(indices[bandedInd.col])
    {}

    /**
     * @brief Checks if the dense matrix indices are out of bounds.
     * @param max Size of the square dense matrix (number of rows/columns)
     * @return True if either row or column is outside [0, max-1], false otherwise
     */
    __device__ bool outOfBounds(size_t max) const {
        return row >= max || col >= max;
    }

};


/**
 * @brief Global device method to compute the positive modulo (remainder).
 *
 * This function handles negative dividends, ensuring the result is always
 * between 0 (inclusive) and n (exclusive).
 *
 * @tparam T Integer type.
 * @param a The dividend (the number).
 * @param n The divisor (the modulus). Must be positive.
 * @return The positive remainder.
 */
template <typename T>
__host__ __device__ T modPos(T a, T n) {
    T result = a % n;
    if (result < 0) result += n;
    return result;
}

#endif //BICGSTAB_KERNELSUPPORT_CUH
