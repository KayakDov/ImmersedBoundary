
#ifndef CUDABANDED_SPARSE_CUH
#define CUDABANDED_SPARSE_CUH
#include <vector>

#include "SimpleArray.h"

//TODO:Check to see if SELL format would be a better fit.


using SpMatDescrPtr = std::shared_ptr<std::remove_pointer<cusparseSpMatDescr_t>::type>;

template <typename Real = double, typename Int = uint32_t>
class SparseCSC {

    SpMatDescrPtr descriptor;

public:
    const size_t rows, cols;
    SimpleArray<Real> values;
    SimpleArray<Int> columnOffsets;//std::shared_ptr<SparseCSC<Real, Int>> B;
    SimpleArray<Int> rowPointers;

    SparseCSC(size_t rows, size_t cols, const SimpleArray<Real> &values, const SimpleArray<Int> &columnOffsets,
              const SimpleArray<Int> &rowPointers);

    static SparseCSC<Real, Int> create(size_t nnz, size_t rows, size_t cols, cudaStream_t stream);

    static SparseCSC<Real, Int> create(size_t rows, SimpleArray<Real> values, SimpleArray<Int> rowPointers,
                                SimpleArray<Int> colOffsets, cudaStream_t stream);

    size_t nnz();

    /**
     * The numver of elements of size T needed for the workspace for mult;
     * @param vec The dense vector being multiplied by.
     * @param result The product will be placed here.
     * @param multProduct A scalar that scales the product.
     * @param multThis  A scalar that scales whatever is in the result before the product is added to itt
     * @param transposeThis Should this be transposed.
     * @param h The handle
     * @return The numver of T type elements in the workspace that mult will need.
     */
    size_t multWorkspaceSize(const SimpleArray<Real> &vec, SimpleArray<Real> &result,
                             const Singleton<Real> &multProduct, const Singleton<Real> &multThis,
                             bool transposeThis, Handle &h) const;


    /**
     * This version alocated memory.  If you'd like to prealocate the memory, call a different function.
    * @param vec The dense vector being multiplied by.
     * @param result The product will be added to whatever is here.
     * @param multProduct A scalar that scales the product.
     * @param preMultResult  A scalar that scales whatever is in the result before the product is added to itt
     * @param transposeThis Should this be transposed.
     * @param workSpace
     * @param h The handle
     */
    void mult(const SimpleArray<Real> &vec, SimpleArray<Real> &result, const Singleton<Real> &multProduct,
    const Singleton<Real> &preMultResult, bool transposeThis, SimpleArray<Real> &workSpace, Handle &h) const;

    void getDense(Mat<Real> &dest, Handle &h) const;
};
/**
 * @brief A high-performance utility for assembling sparse matrices in Compressed Sparse Column (CSC) format.
 * * @tparam Real The floating-point type (float or double).
 * @tparam Int The integer type used for indexing (typically uint32_t or size_t).
 * * @note **Performance Requirement:** To ensure $O(1)$ insertion time, elements MUST be added
 * strictly in increasing order of column index (@p col), and then increasing order of
 * row index (@p row) within each column.
 * * Failure to follow this ordering will result in an invalid CSC structure or significantly
 * degraded performance if the internal logic attempts to sort or shift data.
 * * @code
 * SparseMatrixBuilder builder(numCols);
 * for (Int c = 0; c < numCols; ++c) {
 * for (Int r : neighboringRows) {
 * builder.add(value, r, c); // Correct: Columns 0..N, Rows strictly increasing within C
 * }
 * }
 * @endcode
 */
template <typename Real = double, typename Int = uint32_t>
class SparseMatrixBuilder {
    const size_t cols;
public:
    std::vector<Int> columnOffsets; // Size: cols + 1
    std::vector<Int> rowIndices;    // Row indices for non-zero values (size: nnz)
    std::vector<Real> values;       // Non-zero values (size: nnz)
    /**
     * @brief Constructs a builder for a matrix with a fixed number of columns.
     * @param numCols The total number of columns in the resulting matrix.
     */
    SparseMatrixBuilder(size_t numCols);

    /**
     * @brief Adds a non-zero element to the sparse matrix.
     * * @param val The numerical value to store.
     * @param row The row index (must be >= previous row index in the current column).
     * @param col The column index (must be >= current column index).
     * * @warning If @p col is greater than the current active column, the builder
     * automatically finalizes the offsets for all skipped columns.
     */
    void add(Real val, Int row, Int col);

    /**
     * @brief Finalizes the matrix and transfers data to GPU memory.
     * @param rows The total number of rows in the resulting matrix.
     * @param stream The CUDA stream to use for asynchronous data transfer.
     * @return A SparseCSC object ready for use with cuSPARSE.
     */
    SparseCSC<Real, Int> get(size_t rows, cudaStream_t stream) const;

private:
    Int last_col_added = 0;
};
#endif //CUDABANDED_SPARSE_CUH
