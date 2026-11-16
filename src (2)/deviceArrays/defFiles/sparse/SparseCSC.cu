
#include "../../headers/sparse/SparseCSC.cuh"
# include <string>

template<typename Real, typename Int>
std::unique_ptr<SparseMat<Real, Int>> SparseCSC<Real, Int>::createWithPointer(SimpleArray<Real> vals, SimpleArray<Int> offsets, SimpleArray<Int> inds) const {
    return std::make_unique<SparseCSC<Real, Int>>(SparseCSC<Real, Int>::create(this->rows, vals, offsets, inds));
}

template<typename Real, typename Int>
void SparseCSC<Real, Int>::setDescriptor() {
    cusparseSpMatDescr_t descr;

    cudaDataType valueType = cuValueType<Real>();

    CHECK_SPARSE_ERROR(cusparseCreateCsc(
        &descr,
        this->rows, this->cols, this->nnz(),
        this->offsets.data(),    // int* (size cols + 1)
        this->inds.data(),    // int* (size nnz)
        this->values.data(),        // float* or double* (size nnz)
        cuIndexType<Int>(),       // Offset type
        cuIndexType<Int>(),//maybe use a smaller data type       // Index type
        CUSPARSE_INDEX_BASE_ZERO,  // 0-based indexing
        valueType
    ));

    this->descriptor = SpMatDescrPtr(descr, [](cusparseSpMatDescr_t p) {if (p) cusparseDestroySpMat(p);});
}

template<typename Real, typename Int>
SparseCSC<Real, Int>::SparseCSC(size_t rows, size_t cols, SimpleArray<Real>& values, SimpleArray<Int>& colOffsets,  SimpleArray<Int>& rowInds):
    SparseMat<Real, Int>(rows, cols, values, colOffsets, rowInds) {

    setDescriptor();
}

template<typename Real, typename Int>
SparseCSC<Real, Int> SparseCSC<Real, Int>::create(size_t nnz, size_t rows, size_t cols, Handle& handle) {
    auto vals = SimpleArray<Real>::create(nnz, handle);
    auto colOffsets = SimpleArray<Int>::create(cols + 1, handle);//warning, aligning these arrays to be contiguous in memory seems to cause problems.
    auto rowInds = SimpleArray<Int>::create(nnz, handle);
    return {rows, cols, vals, colOffsets, rowInds};
}

template<typename Real, typename Int>
SparseCSC<Real, Int> SparseCSC<Real, Int>::create(size_t rows, SimpleArray<Real> values, SimpleArray<Int> colOffsets, SimpleArray<Int> rowInds) {
    if (rowInds.size() != values.size())throw std::invalid_argument(
        "SparseCSC Dimensional Mismatch: rowPointers.size() (" + std::to_string(rowInds.size()) +
        ") does not match values.size() (" + std::to_string(values.size()) + ")."
    );
    return {rows, colOffsets.size() - 1, values, colOffsets, rowInds};
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Real, typename Int>
CSCBuilder<Real, Int>::CSCBuilder(size_t cols): cols(cols) {
    columnOffsets.assign(cols + 1, 0);
}

template<typename Real, typename Int>
void CSCBuilder<Real, Int>::add(Real val, Int row, Int col) {
    size_t i = columnOffsets[col];
    while(i < columnOffsets[col + 1] && row > rowIndices[i]) i++;
    if (row == rowIndices[i]) values[i] += val;
    else values.insert(val, i);
    rowIndices.insert(row, i);
    for (;i < columnOffsets.size(); ++i) ++columnOffsets[i];
}

template<typename Real, typename Int>
SparseCSC<Real, Int> CSCBuilder<Real, Int>::get(size_t rows, cudaStream_t stream) const {
    auto sparse = SparseCSC<Real, Int>::create(values.size(), rows, cols, stream);
    sparse.values.set(values.data(), stream);
    sparse.colOffsets.set(columnOffsets.data(), stream);
    sparse.rowsPointers.set(rowIndices.data(), stream);
    return sparse;
}


template class SparseCSC<float, int32_t>;
template class SparseCSC<double, int32_t>;
template class SparseCSC<float, int64_t>;
template class SparseCSC<double, int64_t>;
