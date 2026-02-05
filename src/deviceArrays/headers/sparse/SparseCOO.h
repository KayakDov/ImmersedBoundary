//
// Created by usr on 2/3/26.
//

#ifndef CUDABANDED_SPARSECOO_H
#define CUDABANDED_SPARSECOO_H
#include <vector>

#include "SparseCSR.h"
#include "SparseMat.h"
#include "solvers/BiCGSTAB.cuh"

template <typename Real, typename Int>
class SparseCOO: public SparseMat<Real, Int> {

    SparseCOO(size_t rows, size_t cols, SimpleArray<Real> &values, SimpleArray<Int> &rowPointers, SimpleArray<Int> &colPointers);

protected:
    void setDescriptor() override;

public:
    std::unique_ptr<SparseMat<Real, Int>> createWithPointer(SimpleArray<Real> vals, SimpleArray<Int> rowPointers,
        SimpleArray<Int> colPointers) const override;

    static SparseCOO create(size_t nnz, size_t rows, size_t cols, cudaStream_t stream);

    static SparseCOO create(size_t rows, size_t cols, SimpleArray<Real> values, SimpleArray<Int> rowPointers, SimpleArray<Int> colPointers);


    /**
     * @brief Converts the current COO matrix to CSR format and stores it in the destination.
     * * This method sorts the internal COO data by row and then transforms the row indices
     * into CSR row offsets. It also physically reorders the column indices and values
     * using a permutation vector to maintain data integrity.
     * * @param dst           Destination SparseCSR container where the results are stored.
     * @param nnzAllocated  An auxiliary array used to compute and store the permutation vector.
     * Must be initialized to identity (0, 1, ..., nnz-1).
     * @param buffer        A unique_ptr to a workspace buffer used by cuSPARSE.
     * @note The buffer pointer must point to the **beginning** of an
     * allocated memory section to satisfy cuSPARSE's 128-byte alignment
     * requirements.
     * @param hand          The Handle managing the CUDA stream and cuSPARSE context.
     * * @warning This method relies on cuSPARSE legacy sort routines which typically expect
     * 32-bit signed integers for indices. If the template parameter @p Int is
     * set to @p int64_t, this method may trigger a CUSPARSE_STATUS_INVALID_VALUE
     * or result in undefined behavior.
     */
    template<typename T = Int, typename = std::enable_if_t<std::is_same_v<T, int32_t>>>
    SparseCSR<Real, Int> getCSR(SimpleArray<Int> &offsets, SimpleArray<int32_t> nnzAllocated, std::unique_ptr<SimpleArray<Real>> &buffer, Handle &hand);
};

inline void test() {
    Handle hand;

    auto coo = SparseCOO<double, int32_t>::create(4, 6, 6, hand);

    std::vector<int32_t> rowsHost = {2,3,2,3};
    std::vector<int32_t> colsHost = {2,2,3,3};
    std::vector<double>  valsHost = {1,2,3,4};
    coo.set(rowsHost.data(), colsHost.data(), valsHost.data(), hand);

    auto dense = SquareMat<double>::create(6);
    coo.getDense(dense, hand);
    std::cout << GpuOut<double>(dense, hand) << std::endl;


    auto permBuffer = SimpleArray<int32_t>::create(coo.nnz(), hand);


    std::unique_ptr<SimpleArray<double>> buffer = nullptr;//std::make_unique<SimpleArray<double>>(SimpleArray<double>::create(1000, hand));

    auto offsets = SimpleArray<int32_t>::create(coo.rows + 1, hand);
    auto csr = coo.getCSR(offsets, permBuffer, buffer, hand);

    std::cout << "csr offsets " << GpuOut<int32_t>(csr.offsets, hand) << std::endl;
    std::cout << "csr row inds " << GpuOut<int32_t>(csr.inds, hand) << std::endl;
    std::cout << "csr vals " << GpuOut<double>(csr.values, hand) << std::endl;
    std::cout << "permutations  " << GpuOut<int32_t>(permBuffer, hand) << std::endl;


    csr.getDense(dense, hand);
    std::cout << GpuOut<double>(dense, hand) << std::endl;
}


#endif //CUDABANDED_SPARSECOO_H
