#include "deviceArrays/headers/SparseMat.h"

cusparseOperation_t cuTranspose(bool trans) {
    return trans? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
}

template <typename Real, typename Int>
size_t SparseMat<Real, Int>::multWorkspaceSize(const SimpleArray<Real>& vec, SimpleArray<Real>& result, const Singleton<Real>& multProduct, const Singleton<Real>& preMultResult, const bool transposeThis, Handle& h) const {
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
void SparseMat<Real, Int>::mult(const SimpleArray<Real>& vec, SimpleArray<Real>& result,
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

template<typename Real, typename Int>
void SparseMat<Real, Int>::getDense(Mat<Real>& dest, Handle& h) const {
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
SparseMat<Real, Int>::SparseMat(size_t rows, size_t cols, SimpleArray<Real>& values, SimpleArray<Int> &offsets, SimpleArray<Int> &inds):
    rows(rows), cols(cols), values(values), offsets(offsets), inds(inds) {
}

template<typename Real, typename Int>
size_t SparseMat<Real, Int>::nnz() const{
    return values.size();
}

template class SparseMat<float, int32_t>;
template class SparseMat<double, int32_t>;
template class SparseMat<float, int64_t>;
template class SparseMat<double, int64_t>;