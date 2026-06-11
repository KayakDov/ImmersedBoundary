//
// Created by usr on 06/11/26.
//

#ifndef BICGSTAB_TRIDIAGONAL_H
#define BICGSTAB_TRIDIAGONAL_H

#include "BandedMat.h"

/**
 * @brief Represents a square tridiagonal matrix layout.
 * * Column mapping strategy:
 * - Column 0: Main/Primary diagonal (offset 0)
 * - Column 1: Subdiagonal (offset -1)
 * - Column 2: Superdiagonal (offset 1)
 */
template<typename T>
class TriDiagonal : public BandedMat<T> {
public:
    inline static const AdjacencyIndPair prevNext{1, 1};
    inline static const AdjacencyInd primary{0, 0};
    /**
     * @brief Allocates an uninitialized tridiagonal matrix directly on device memory.
     * @param denseSqMatDim Matrix dimension count (rows/columns).
     * @param hand The active CUDA stream/context handle.
     */
    TriDiagonal(size_t denseSqMatDim, Handle &hand);

    /**
     * @brief Wraps an existing allocated matrix asset as a Tridiagonal system.
     * @param copyFrom Input Matrix buffer with exactly 3 columns.
     * @param hand The active CUDA stream/context handle.
     */
    TriDiagonal(const Mat<T> &copyFrom, Handle &hand);

private:
    /**
     * @brief Lazily constructs and shares the unified coordinate indices vector.
     */
    static const Vec<int32_t>& getSharedIndices(Handle &hand);
};

#endif //BICGSTAB_TRIDIAGONAL_H