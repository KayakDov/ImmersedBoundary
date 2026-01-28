//
// Created by usr on 1/28/26.
//

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
};


#endif //CUDABANDED_SPARSECSR_H
