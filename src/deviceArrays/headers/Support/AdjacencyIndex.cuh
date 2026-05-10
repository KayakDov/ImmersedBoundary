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
    const size_t colInBanded;

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
    __device__ __host__ AdjacencyInd(const size_t col, const int32_t diag) : colInBanded(col), diag(diag) {
    }

    /**
     * Provides the corespondoing index in a banded matrix.
     * @param denseRow The row of the index in the dense matrix format.
     * @return Indices for a dense matrix.
     */
    __device__ GridInd2d bandedInd(size_t denseRow) const {
        if (diag < 0) return {denseRow + diag, colInBanded};
        return {denseRow, colInBanded};
    }

    /**
     * Provides the corespondoing index in a dense matrix.
     * @param denseRow The row of the index in the dense matrix format.
     * @return Indices for a dense matrix.
     */
    __device__ GridInd2d denseInd(size_t denseRow) const {

        return {denseRow, denseRow + diag};
    }
};




/**
 * @brief Helper class for managing symmetric or paired diagonals in a banded operator.
 * * Provides utility methods to derive "left" (lower) and "right" (upper) diagonal
 * descriptors based on a central mapping.
 */
class AdjacencyIndPair {

public:
    AdjacencyInd left, right;
    /**
     * @brief Constructs an Adjacency Index Pair.
     * @param firstCol The base column index.  The right diagonal will be stored at this index plus 1, so be sure not to
     * put anything else there.
     * @param posDiagonalOffset The base diagonal offset.  This should be positive.
     */
    __host__ AdjacencyIndPair(const size_t firstCol, const size_t posDiag) :
        left(firstCol, -static_cast<int32_t>(posDiag)),
        right(firstCol + 1, posDiag){}

    /**
     * @brief Accesses either the left or right diagonal descriptor using a boolean flag.
     * @param isRight If true, returns the right descriptor; otherwise, returns the left.
     * @return The corresponding AdjacencyInd.
     */
    __host__ __device__ AdjacencyInd operator[](bool isRight) {
        return isRight ? right : left;
    }
};

template <typename T>
class Vec;
/**
* How the adjacent grid cells are stored in the laplacian. *
 */
class AdjacencyPatern : public XYZ<AdjacencyIndPair>{
public:

    const bool is3d;
    const AdjacencyInd here;

    /**
     *
     * @param dim The dimensions of the grid.
     */
    __host__ AdjacencyPatern(GridDim dim):
        XYZ<AdjacencyIndPair>(
            {3, dim[GridInd3d(0, 1, 0)]},
            {1, dim[GridInd3d(1, 0, 0)]},
            {5, dim[GridInd3d(0, 0, 1)]}
        ),
        here(0, 0),
        is3d(dim.numDims() == 3)
    {

    };

    __host__ void loadMapRowToDiag(Vec<int32_t> &diags, cudaStream_t stream) const;

    __host__ static void loadMapRowToDiag(Vec<int32_t> &diags, std::vector<AdjacencyInd> &indices, cudaStream_t stream);
};