//
// Created by usr on 1/4/26.
//

#include <iostream>

#include "deviceArrays/headers/SparseCSC.cuh"
#include "Streamable.h"

cusparseOperation_t cuTranspose(bool trans) {
    return trans? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
}

template<typename Real, typename Int>
SparseCSC<Real, Int>::SparseCSC(size_t rows, size_t cols, const SimpleArray<Real>& values, const SimpleArray<Int>& columnOffsets,
    const SimpleArray<Int>& rowPointers): values(values), columnOffsets(columnOffsets), rowPointers(rowPointers), rows(rows), cols(cols) {
    cusparseSpMatDescr_t descr;

    cudaDataType valueType = cuValueType<Real>();

    // Create the descriptor
    CHECK_SPARSE_ERROR(cusparseCreateCsc(
        &descr,
        rows, cols, nnz(),
        this->columnOffsets.data(),    // int* (size cols + 1)
        this->rowPointers.data(),    // int* (size nnz)
        this->values.data(),        // float* or double* (size nnz)
        cuIndexType<Int>(),       // Offset type
        cuIndexType<Int>(),//maybe use a smaller data type       // Index type
        CUSPARSE_INDEX_BASE_ZERO,  // The 0-based indexing you requested
        valueType
    ));

    descriptor = SpMatDescrPtr(descr, [](cusparseSpMatDescr_t p) {if (p) cusparseDestroySpMat(p);});
}

template<typename Real, typename Int>
SparseCSC<Real, Int> SparseCSC<Real, Int>::create(size_t nnz, size_t rows, size_t cols, cudaStream_t stream) {
    auto vals = SimpleArray<Real>::create(nnz, stream);
    auto colOffsets = SimpleArray<Int>::create(cols + 1, stream);//warning, aligning these arrays to be contiguouse in memory seems to cause problems.
    auto rowPointers = SimpleArray<Int>::create(nnz, stream);
    return {rows, cols, vals, colOffsets, rowPointers};
}

template<typename Real, typename Int>
SparseCSC<Real, Int> SparseCSC<Real, Int>::create(size_t rows, SimpleArray<Real> values, SimpleArray<Int> rowPointers, SimpleArray<Int> colOffsets, cudaStream_t stream) {
    if (rowPointers.size() != values.size())throw std::invalid_argument(
        "SparseCSC Dimensional Mismatch: rowPointers.size() (" + std::to_string(rowPointers.size()) +
        ") does not match values.size() (" + std::to_string(values.size()) + ")."
    );
    return {rows, colOffsets.size() - 1, values, colOffsets, rowPointers};
}



template<typename Real, typename Int>
size_t SparseCSC<Real, Int>::nnz() const{
    return values.size();
}

template <typename Real, typename Int>
size_t SparseCSC<Real, Int>::multWorkspaceSize(const SimpleArray<Real>& vec, SimpleArray<Real>& result, const Singleton<Real>& multProduct, const Singleton<Real>& preMultResult, const bool transposeThis, Handle& h) const {
    size_t bytesRequired = 0;
    cudaDataType valueType = cuValueType<Real>();
    cusparseOperation_t op = cuTranspose(transposeThis);

    CHECK_SPARSE_ERROR(cusparseSpMV_bufferSize(
        h, op,
        multProduct.data(), descriptor.get(), vec.getDescr(),
        preMultResult.data(), result.getDescr(),
        valueType, CUSPARSE_SPMV_ALG_DEFAULT, &bytesRequired));

    return (bytesRequired + sizeof(Real) - 1) / sizeof(Real);
}

template <typename Real, typename Int>
void SparseCSC<Real, Int>::mult(const SimpleArray<Real>& vec, SimpleArray<Real>& result,
                        const Singleton<Real>& multProduct, const Singleton<Real>& preMultResult,
                        bool transposeMat, SimpleArray<Real>& workSpace, Handle& h) const{

    cudaDataType valueType = cuValueType<Real>();
    cusparseOperation_t op = cuTranspose(transposeMat);

    CHECK_SPARSE_ERROR(cusparseSpMV(
        h, op,
        multProduct.toKernel1d(), descriptor.get(), vec.getDescr(),
        preMultResult.toKernel1d(), result.getDescr(),
        valueType, CUSPARSE_SPMV_ALG_DEFAULT, workSpace.data()));
}

// template<typename Real, typename Int>
// void SparseCSC<Real, Int>::mult(const SimpleArray<Real> &vec, SimpleArray<Real> &result, const Singleton<Real> &multProduct, const Singleton<Real> &preMultResult, bool transposeThis, Handle &h) const {
//
//     int wsSize = multWorkspaceSize(vec, result, multProduct, preMultResult, transposeThis,h);
//     auto workSpace = SimpleArray<Real>::create(wsSize, h);
//     mult(vec, result, multProduct, preMultResult, transposeThis, workSpace, h);
// }

template<typename Real, typename Int>
void SparseCSC<Real, Int>::getDense(Mat<Real>& dest, Handle& h) const {
    size_t bytesRequired = 0;
    cudaDataType valueType = cuValueType<Real>();

    // 1. Get buffer size required for conversion
    CHECK_SPARSE_ERROR(cusparseSparseToDense_bufferSize(
        h, descriptor.get(), dest.getDescr(),
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bytesRequired));

    auto workSpace = SimpleArray<Real>::create((bytesRequired + sizeof(Real) - 1) / sizeof(Real), h);

    // 2. Execute conversion
    CHECK_SPARSE_ERROR(cusparseSparseToDense(
        h, descriptor.get(), dest.getDescr(),
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT, workSpace.data()));
}

template<typename Real, typename Int>
SparseMatrixBuilder<Real, Int>::SparseMatrixBuilder(size_t cols): cols(cols) {
    columnOffsets.assign(cols + 1, 0);
}

template<typename Real, typename Int>
void SparseMatrixBuilder<Real, Int>::add(Real val, Int row, Int col) {
    size_t i = columnOffsets[col];
    while(i < columnOffsets[col + 1] && row > rowIndices[i]) i++;
    if (row == rowIndices[i]) values[i] += val;
    else values.insert(val, i);
    rowIndices.insert(row, i);
    for (;i < columnOffsets.size(); ++i) ++columnOffsets[i];
}

template<typename Real, typename Int>
SparseCSC<Real, Int> SparseMatrixBuilder<Real, Int>::get(size_t rows, cudaStream_t stream) const {
    auto sparse = SparseCSC<Real, Int>::create(values.size(), rows, cols, stream);
    sparse.values.set(values.data(), stream);
    sparse.columnOffsets.set(columnOffsets.data(), stream);
    sparse.rowsPointers.set(rowIndices.data(), stream);
    return sparse;
}


template class SparseCSC<float, int32_t>;
template class SparseCSC<double, int32_t>;
template class SparseCSC<float, int64_t>;
template class SparseCSC<double, int64_t>;



// void testSparseMult() {
//     Handle hand;
//     size_t rows = 4, cols = 5;
//
//     // auto dense = Mat<double>::create(4,4);
//     // std::vector<double> hostDense = {0,0,0,0,   0,1,1,0,   0,1,1,0,   0,0,0,0};
//     // dense.set(hostDense.data(), hand);
//
//     auto sparse = SparseCSC<double>::create(4, rows, cols, hand);
//     std::vector<double> hostValsDense = {1, 1, 1, 1};
//     std::vector<int32_t> hostColOffsets = {0, 0, 2, 4, 4, 4, 4};
//     std::vector<int32_t> hostRowInds = {1, 2, 1, 2};
//
//     sparse.values.set(hostValsDense.data(), hand);
//     sparse.columnOffsets.set(hostColOffsets.data(), hand);
//     sparse.rowsPointers.set(hostRowInds.data(), hand);
//
//     auto x = SimpleArray<double>::create(cols, hand);
//     std::vector<double> hostVec = {1, 2, 3, 4};
//     x.set(hostVec.data(), hand);
//
//
//     auto result = SimpleArray<double>::create(rows, hand);
//
//     // dense.mult(vec, result, &hand, &Singleton<double>::ONE, &Singleton<double>::ZERO, false);
//
//     // (const SimpleArray<Real>& vec, SimpleArray<Real>& result,
//     //                     const Singleton<Real>& multProduct, const Singleton<Real>& multThis,
//     //                     bool transposeThis, SimpleArray<Real>& workSpace, Handle& h)
//
//     std::cout << "x = \n" << GpuOut<double>(x, hand) << std::endl;
//     sparse.mult(x, result, Singleton<double>::ONE, Singleton<double>::ZERO, false, hand);
//
//     // std::cout << GpuOut<double>(dense, hand) << std::endl;
//
//     std::cout << "result = \n" << GpuOut<double>(result, hand) << std::endl;
// }