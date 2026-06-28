/**
 * @file SparseCSR.h
 * @brief Defines CSR sparse-matrix storage and operations.
 * @ingroup sparse_matrices
 *
 * @details
 * Sparse-matrix classes build on the device array layer and expose storage-format-specific operations while preserving explicit CUDA handle and stream control.
 */

#ifndef CUDABANDED_SPARSECSR_H
#define CUDABANDED_SPARSECSR_H
#include "SparseMat.h"

template <typename Real, typename Int>
class SparseCSR : public SparseMat<Real, Int> {
    //Reminder, in CSR offsets are for rows and inds are for cols.
protected:
    void setDescriptor() override;

public:

    SparseCSR(size_t rows, size_t cols, SimpleArray<Real>& vals, SimpleArray<Int>& rowOffsets, SimpleArray<Int>& colInds);

    static SparseCSR create(size_t nnz, size_t rows, size_t cols, cudaStream_t stream);

    static SparseCSR create(size_t cols, SimpleArray<Real> values, SimpleArray<Int> rowOffsets, SimpleArray<Int> colInds);

    std::unique_ptr<SparseMat<Real, Int>> createWithPointer(SimpleArray<Real> vals, SimpleArray<Int> offsets, SimpleArray<Int> inds) const override;
};


#endif //CUDABANDED_SPARSECSR_H
