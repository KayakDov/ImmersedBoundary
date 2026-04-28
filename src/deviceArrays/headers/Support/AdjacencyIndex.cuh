#pragma once

#include <vector>

/**
 * @brief Represents a single diagonal's metadata within a banded matrix storage format.
 * * Maps a specific data column in the banded buffer to its mathematical diagonal offset
 * in the original square matrix.
 */
class AdjacencyInd {
public:
    /**
     * @brief The 0-based column index in the banded matrix storage array.
     */
    const size_t col;

    /**
     * @brief The mathematical diagonal offset.
     * * Positive values represent upper diagonals; negative values represent lower diagonals;
     * 0 represents the main diagonal.
     */
    const int32_t diag;

    /**
     * @brief Constructs a new Adjacency Index mapping.
     * @param col The column in the banded buffer.
     * @param diag The diagonal index.
     */
    __device__ __host__ AdjacencyInd(const size_t col, const int32_t diag) : col(col), diag(diag) {
    }
};

/**
 * @brief Helper class for managing symmetric or paired diagonals in a banded operator.
 * * Provides utility methods to derive "left" (lower) and "right" (upper) diagonal
 * descriptors based on a central mapping.
 */
class AdjacencyIndPair {
    const size_t firstColOfTwoConsecutive, nextDiagOffset;
public:
    /**
     * @brief Constructs an Adjacency Index Pair.
     * @param firstCol The base column index.  The right diagonal will be stored at this index plus 1, so be sure not to
     * put anything else there.
     * @param nextDiagOffset The base diagonal offset.  This should be positive.
     */
    __host__ AdjacencyIndPair(const size_t firstCol, const size_t nextDiagOffset) :
        firstColOfTwoConsecutive(firstCol),
        nextDiagOffset(nextDiagOffset) {}

    /**
     * @brief Returns the "Left" (mirrored) version of this diagonal.
     * @return An AdjacencyInd with the same column but a negated diagonal offset.
     */
    __host__ __device__ [[nodiscard]] AdjacencyInd getLeft() const {
        return {this->firstColOfTwoConsecutive, -static_cast<int32_t>(this->nextDiagOffset)};
    }

    /**
     * @brief Returns the "Right" version of this diagonal.
     * @return An AdjacencyInd with the next sequential column and the original diagonal offset.
     */
    __host__ __device__ [[nodiscard]] AdjacencyInd getRight() const {
        return {this->firstColOfTwoConsecutive + 1, static_cast<int32_t>(this->nextDiagOffset)};
    }

    /**
     * @brief Accesses either the left or right diagonal descriptor using a boolean flag.
     * @param isRight If true, returns the right descriptor; otherwise, returns the left.
     * @return The corresponding AdjacencyInd.
     */
    __host__ __device__ AdjacencyInd operator[](bool isRight) {
        return isRight ? getRight() : getLeft();
    }
};

template <typename T>
class Vec;
/**
* How the adjacent grid cells are stored in the laplacian. *
 */
class AdjacencyPatern {//TODO:resolve 3d issues.
public:

    const bool is3d;
    const AdjacencyInd here;
    const AdjacencyIndPair upDown, leftRight, frontBack;
    /**
     *
     * @param dim The dimensions of the grid.
     */
    __host__ AdjacencyPatern(GridDim dim):
        here(0, 0),
        upDown(1, 1),
        leftRight(3, dim.rows * dim.layers),
        frontBack(5, dim.rows),
        is3d(dim.numDims() == 3)
    {

    };

    __host__ void loadMapRowToDiag(Vec<int32_t> &diags, cudaStream_t stream) const;

    __host__ static void loadMapRowToDiag(Vec<int32_t> &diags, std::vector<AdjacencyInd> &indices, cudaStream_t stream);
};