//
// Created by usr on 1/28/26.
//

#include "../headers/SparseCSR.h"

template<typename Real, typename Int>
void SparseCSR<Real, Int>::setDescriptor() {
    cusparseSpMatDescr_t descr;

    cudaDataType valueType = cuValueType<Real>();

    CHECK_SPARSE_ERROR(cusparseCreateCsr(
        &descr,
        this->rows, this->cols, this->nnz(),
        this->offsets.data(), // int* (size cols + 1)
        this->inds.data(), // int* (size nnz)
        this->values.data(), // float* or double* (size nnz)
        cuIndexType<Int>(), // Offset type
        cuIndexType<Int>(), //maybe use a smaller data type       // Index type
        CUSPARSE_INDEX_BASE_ZERO, // The 0-based indexing you requested
        valueType
    ));

    this->descriptor = SpMatDescrPtr(descr, [](cusparseSpMatDescr_t p) { if (p) cusparseDestroySpMat(p); });
}

template<typename Real, typename Int>
SparseCSR<Real, Int>::SparseCSR(size_t rows, size_t cols, SimpleArray<Real>& vals, SimpleArray<Int>& rowOffsets, SimpleArray<Int>& colInds):
    SparseMat<Real, Int>(rows, cols, vals, rowOffsets, colInds){

    setDescriptor();
}

template<typename Real, typename Int>
SparseCSR<Real, Int> SparseCSR<Real, Int>::create(size_t nnz, size_t rows, size_t cols, cudaStream_t stream) {
    auto vals = SimpleArray<Real>::create(nnz, stream);
    auto rowOffsets = SimpleArray<Int>::create(rows + 1, stream);//warning, aligning these arrays to be contiguouse in memory seems to cause problems.
    auto colInds = SimpleArray<Int>::create(nnz, stream);
    return {rows, cols, vals, rowOffsets, colInds};
}

template<typename Real, typename Int>
SparseCSR<Real, Int> SparseCSR<Real, Int>::create(size_t cols, SimpleArray<Real> values, SimpleArray<Int> rowOffsets, SimpleArray<Int> colInds) {
    if (colInds.size() != values.size())throw std::invalid_argument(
        "SparseCSC Dimensional Mismatch: rowPointers.size() (" + std::to_string(colInds.size()) +
        ") does not match values.size() (" + std::to_string(values.size()) + ")."
    );
    return {rowOffsets.size() - 1, cols, values, rowOffsets, colInds};
}

template class SparseCSR<float, int32_t>;
template class SparseCSR<double, int32_t>;
template class SparseCSR<float, int64_t>;
template class SparseCSR<double, int64_t>;