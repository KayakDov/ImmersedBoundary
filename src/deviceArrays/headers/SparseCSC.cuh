
#ifndef CUDABANDED_SPARSE_CUH
#define CUDABANDED_SPARSE_CUH
#include <vector>

#include "SimpleArray.h"
#include "SparseCSR.h"
#include "Streamable.h"

//TODO:Check to see if SELL format would be a better fit.


using SpMatDescrPtr = std::shared_ptr<std::remove_pointer<cusparseSpMatDescr_t>::type>;

template <typename Real, typename Int>
class SparseCSC : public SparseMat<Real, Int> {

    //Reminder, in CSC offsets are for columns and inds are for rows.
protected:
    virtual void setDescriptor() override;

public:

    SparseCSC(size_t rows, size_t cols, SimpleArray<Real> &values, SimpleArray<Int> &colOffsets, SimpleArray<Int> &rowInds);

    static SparseCSC create(size_t nnz, size_t rows, size_t cols, cudaStream_t stream);

    static SparseCSC create(size_t rows, SimpleArray<Real> values, SimpleArray<Int> colOffsets, SimpleArray<Int> rowInds);

    std::shared_ptr<SparseMat<Real, Int>> createWithPointer(SimpleArray<Real> vals,
        SimpleArray<Int> offsets, SimpleArray<Int> inds) const override;
};
/**
 * @brief A high-performance utility for assembling sparse matrices in Compressed Sparse Column (CSC) format.
 * * @tparam Real The floating-point type (float or double).
 * @tparam Int The integer type used for indexing (typically int32_t).
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
template <typename Real, typename Int>
class CSCBuilder {
    const size_t cols;
public:
    std::vector<Int> columnOffsets; // Size: cols + 1
    std::vector<Int> rowIndices;    // Row indices for non-zero values (size: nnz)
    std::vector<Real> values;       // Non-zero values (size: nnz)
    /**
     * @brief Constructs a builder for a matrix with a fixed number of columns.
     * @param numCols The total number of columns in the resulting matrix.
     */
    CSCBuilder(size_t numCols);

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

};

/**
 * @brief Helper wrapper to print SparseCSC contents using GpuOut.
 */
template <typename Real, typename Int>
struct SparseCSCOut {
    const SparseCSC<Real, Int>& mat;
    Handle& handle;

    SparseCSCOut(const SparseCSC<Real, Int>& m, Handle& h)
        : mat(m), handle(h) {}
};

template <typename Real, typename Int>
std::ostream& operator<<(std::ostream& os, const SparseCSCOut<Real, Int>& out) {
    const auto& mat = out.mat;
    Handle& h = out.handle;

    os << "SparseCSC Debug Output" << std::endl;
    os << "  Dimensions: " << mat.rows << " x " << mat.cols << std::endl;
    os << "  nnz: " << mat.nnz() << std::endl;

    os << "Values:" << std::endl;
    os << GpuOut<Real>(mat.values, h) << std::endl;

    os << "Row Pointers:" << std::endl;
    os << GpuOut(mat.rowInds, h) << std::endl;

    os << "Column Offsets:" << std::endl;
    os << GpuOut(mat.columnOffsets, h) << std::endl;

    os << "dense:" << std::endl;
    auto dense = Mat<Real>::create(mat.rows, mat.cols);
    mat.getDense(dense, h);
    os << GpuOut(dense, h) << std::endl;

    return os;
}



#endif //CUDABANDED_SPARSE_CUH
