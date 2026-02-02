
#ifndef CUDABANDED_SPARSEMAT_H
#define CUDABANDED_SPARSEMAT_H
#include <cusparse.h>
#include <memory>

#include "SimpleArray.h"

using SpMatDescrPtr = std::shared_ptr<std::remove_pointer<cusparseSpMatDescr_t>::type>;

template <typename Real, typename Int>
class SparseMat {


protected:
    SpMatDescrPtr descriptor;

    /**
     * Should be called in dependent class constructors.
     */
    virtual void setDescriptor() =0;

public:
    const size_t rows, cols;
    SimpleArray<Real> values;
    SimpleArray<Int> offsets, inds;

    SparseMat(size_t rows, size_t cols, SimpleArray<Real> &values, SimpleArray<Int> &offsets, SimpleArray<Int> &inds);

    [[nodiscard]] size_t nnz() const;

    void changeStorageType(SparseMat<Real, Int> &dest, Handle &hand, std::shared_ptr<SimpleArray<Real>> &buffer) const;

    /**
     * The numver of elements of size T needed for the workspace for mult;
     * @param vec The dense vector being multiplied by.
     * @param result The product will be placed here.
     * @param multProduct A scalar that scales the product.
     * @param preMultResult  A scalar that scales whatever is in the result before the product is added to itt
     * @param transposeThis Should this be transposed.
     * @param h The handle
     * @return The numver of T type elements in the workspace that mult will need.
     */
    virtual size_t multWorkspaceSize(const SimpleArray<Real> &vec, SimpleArray<Real> &result,
                             const Singleton<Real> &multProduct, const Singleton<Real> &preMultResult,
                             bool transposeThis, Handle &h) const;


    /**
     * This version alocated memory.  If you'd like to prealocate the memory, call a different function.
    * @param vec The dense vector being multiplied by.
     * @param result The product will be added to whatever is here.
     * @param multProduct A scalar that scales the product.
     * @param preMultResult  A scalar that scales whatever is in the result before the product is added to itt
     * @param transposeMat Should this be transposed.
     * @param workSpace
     * @param h The handle
     */
    void mult(const SimpleArray<Real> &vec, SimpleArray<Real> &result, const Singleton<Real> &multProduct, const Singleton<Real> &preMultResult, bool transposeMat, SimpleArray<Real> &workSpace, Handle &h) const;

    /**
     * Creates a dense version of this matrix.
     * @param dest Where the dense matrix will be put.
     * @param h the handle.
     */
    void getDense(Mat<Real> &dest, Handle &h) const;

    void set(Int* offsets, Int* inds, Real* vals, Handle& hand);

    virtual std::shared_ptr<SparseMat<Real, Int>> createWithPointer(
        SimpleArray<Real> vals,
        SimpleArray<Int> offsets,
        SimpleArray<Int> inds
    ) const = 0;
};
#endif //CUDABANDED_SPARSEMAT_H