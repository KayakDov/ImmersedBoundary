//
// Created by usr on 2/3/26.
//

#include "../headers/SparseCOO.h"

#include "SparseCSR.h"

// row pointers is offsets,
// col pointers is inds

template<typename Real, typename Int>
SparseCOO<Real, Int>::SparseCOO(size_t rows, size_t cols, SimpleArray<Real> &values, SimpleArray<Int> &rowPointers, SimpleArray<Int> &colPointers) :
    SparseMat<Real, Int>(rows, cols, values, rowPointers, colPointers) {
    setDescriptor();
}

template<typename Real, typename Int>
void SparseCOO<Real, Int>::setDescriptor() {
    cusparseSpMatDescr_t descr;

    const cudaDataType valueType = cuValueType<Real>();

    CHECK_SPARSE_ERROR(cusparseCreateCoo(
        &descr,
        this->rows, this->cols, this->nnz(),
        this->offsets.data(), // int* (size nnz)
        this->inds.data(), // int* (size cols + 1)
        this->values.data(), // float* or double* (size nnz)
        cuIndexType<Int>(), // Offset type
        CUSPARSE_INDEX_BASE_ZERO, // The 0-based indexing you requested
        valueType
    ));

    this->descriptor = SpMatDescrPtr(descr, [](cusparseSpMatDescr_t p) { if (p) cusparseDestroySpMat(p); });
}

template<typename Real, typename Int>
std::shared_ptr<SparseMat<Real, Int>> SparseCOO<Real, Int>::createWithPointer(SimpleArray<Real> vals, SimpleArray<Int> rowPointers, SimpleArray<Int> colPointers) const {
    return std::make_shared<SparseCOO<Real, Int>>(SparseCOO<Real, Int>::create(this->rows, this->cols, vals, rowPointers, colPointers));
}


template<typename Real, typename Int>
SparseCOO<Real, Int> SparseCOO<Real, Int>::create(size_t nnz, size_t rows, size_t cols, cudaStream_t stream) {
    auto vals = SimpleArray<Real>::create(nnz, stream);
    auto rowsP = SimpleArray<Int>::create(nnz, stream);
    auto colsP = SimpleArray<Int>::create(nnz, stream);
    return {rows, cols, vals, rowsP, colsP};
}

template<typename Real, typename Int>
SparseCOO<Real, Int> SparseCOO<Real, Int>::create(size_t rows, size_t cols, SimpleArray<Real> values, SimpleArray<Int> rowPointers, SimpleArray<Int> colPointers) {
    return SparseCOO<Real, Int>(
        rows,
        cols,
        values,
        rowPointers,
        colPointers
    );
}

template<typename Real, typename Int> // Class template parameters
template<typename T, typename>        // Member template parameters (matches the enable_if)
SparseCSR<Real, Int> SparseCOO<Real, Int>::getCSR(SimpleArray<Int>& offsets, SimpleArray<int32_t> nnzAllocated, std::unique_ptr<SimpleArray<Real>>& buffer, Handle& hand) {

    nnzAllocated.setValsToIndecies(hand);
    size_t bufferSizeInBytes[1];

    //TODO:  The address of pBuffer must be multiple of 128 bytes. If not, CUSPARSE_STATUS_INVALID_VALUE is returned.  This might be a limit on the tupe in buffer.  We'll see if it works without this.

    CHECK_SPARSE_ERROR(cusparseXcoosort_bufferSizeExt(
        hand,
        this->rows,
        this->cols,
        this->nnz(),
        this->offsets.data(),
        this->inds.data(),
        bufferSizeInBytes
    ));


    if (!buffer || *bufferSizeInBytes > buffer->size() * sizeof(Real)) {
        size_t neededElements = (1.5 * *bufferSizeInBytes + sizeof(Real) - 1) / sizeof(Real);
        buffer = std::make_unique<SimpleArray<Real>>(SimpleArray<Real>::create(std::max(neededElements, this->nnz()), hand));
    }

    CHECK_SPARSE_ERROR(cusparseXcoosortByRow(hand,
        this->rows,
        this->cols,
        this->nnz(),
        this->offsets.data(),
        this->inds.data(),
        nnzAllocated.data(),
        buffer->data()
    ));

    this->values.permute(nnzAllocated, *buffer, hand);
    this->values.set(*buffer, hand);

    CHECK_SPARSE_ERROR(cusparseXcoo2csr(
            hand,
            this->offsets.data(),      // COO row indices
            this->nnz(),
            this->rows,
            offsets.data(),      // CSR rowOffsets
            CUSPARSE_INDEX_BASE_ZERO
    ));

    return SparseCSR<Real, Int>::create(this-> cols, this->values, offsets, this->inds);
}
//TODO: write getCSC

template class SparseCOO<float, int32_t>;
template class SparseCOO<double, int32_t>;
template class SparseCOO<float, int64_t>;
template class SparseCOO<double, int64_t>;

// Explicit instantiation for the getCSR member template
template SparseCSR<double, int32_t> SparseCOO<double, int32_t>::getCSR<int32_t, void>(
    SimpleArray<int32_t>&,
    SimpleArray<int32_t>,
    std::unique_ptr<SimpleArray<double>>&,
    Handle&
);

template SparseCSR<float, int32_t> SparseCOO<float, int32_t>::getCSR<int32_t, void>(
    SimpleArray<int32_t>&,
    SimpleArray<int32_t>,
    std::unique_ptr<SimpleArray<float>>&,
    Handle&
);
