//
// Created by usr on 12/24/25.
//

#include "poisson/Laplacian.cuh"

#include <vector>

#include "deviceArrays/headers/Support/Streamable.h"
#include "math/Real3dDevice.hpp"

#include <memory>


/**
 * @brief Efficiently creates unique objects based on 1D Laplacian boundary conditions.
 *
 * This utility iterates through the X, Y, and Z dimensions of a Laplacian setup.
 * If the boundary conditions (dimensions, spacing, and types) for two dimensions
 * are identical, the function reuses the existing object by sharing ownership
 * via std::shared_ptr. This prevents redundant memory allocations and unnecessary
 * re-computation of Laplacian matrices or eigenvalue vectors.
 *
 * @tparam T          The floating-point type (e.g., float or double).
 * @tparam ResultType The type of object to be created (e.g., Mat<T> or Vec<T>).
 * @tparam Factory    A callable type (lambda or function) that returns ResultType by value.
 *
 * @param[in]  conds    A reference to an array of 3 LaplacianConditions (X, Y, Z).
 * @param[out] outputs  An array of 3 shared_ptrs to be populated with the results.
 * @param[in]  factory  A factory function: `ResultType factory(const LaplacianConditions<T>&)`.
 * * @note Because ResultType is wrapped in a shared_ptr, ResultType does not need
 * to be assignable, which is critical for classes with const data members.
 */
template <typename T, typename ResultType, typename Factory>
void createUnique(const LaplcianConditions<T> (&conds)[3], std::shared_ptr<const ResultType> (&outputs)[3], Factory factory){
    for (size_t i = 0; i < 3; ++i) {
        bool matched = false;
        for (size_t j = 0; j < i; ++j) {
            if (conds[i] == conds[j]) {
                outputs[i] = outputs[j];
                matched = true;
                break;
            }
        }
        if (!matched) {
            outputs[i] = std::make_shared<ResultType>(factory(conds[i]));
        }
    }
}

template<typename T>
LaplcianConditions<T>::LaplcianConditions(const size_t length, const BoundaryConditionPair<T> &boundary) :
    dim(length), boundary(boundary) {}

template<typename T>
std::unique_ptr<BandedMat<T>> & Laplacian1d<T>::operator[](size_t dim) {
    switch (dim) {
        case 0: return Lx;
        case 1: return Ly;
        case 2: return Lz;
        default: throw std::out_of_range("Invalid dimension: must be 0, 1, or 2");
    }
}

template<typename T>
Laplacian<T>::Laplacian(const GridDim& dim, const Real3d& delta, const BoundaryConfig<T>& boundary) :
    dim(dim),
    delta(delta),
    boundary(boundary),
    adjacencies(dim) {
}

template <typename T>
T invSq(T x) {
    return 1/(x*x);
}

void AdjacencyPatern::loadMapRowToDiag(Vec<int32_t>& diags, const cudaStream_t stream) const{
    loadMapRowToDiag(diags, {here, up, down, left, right, front, back}, stream);
}

void AdjacencyPatern::loadMapRowToDiag(Vec<int32_t> &diags, std::vector<AdjacencyInd> indices, cudaStream_t stream) {
    std::vector<int32_t> diagsCpu(diags.size(), 0);
    for (AdjacencyInd ind : indices) diagsCpu[ind.col] = ind.diag;
    diags.set(diagsCpu.data(), stream);
}

template<typename T>
void Laplacian<T>::setOperation(cudaStream_t stream) {
    size_t numDiags =  dim.layers > 1 ? numDiagonals3d : numDiagonals2d;
    Vec<int32_t> preAllocatedForIndices = Vec<int32_t>::create(numDiags, stream);
    Mat<T> rhsAndL = Mat<T>::create(dim.size(), numDiags + 1);
    Mat<T> preAllocatedForL = rhsAndL.subMat(0,0,dim.size(), numDiags);
    SimpleArray<T> rhsModifier = rhsAndL.col(numDiags);

    setOperation(stream, preAllocatedForL, preAllocatedForIndices, rhsModifier);
}

template<typename T>
void Laplacian<T>::setOperation(cudaStream_t stream, Mat<T> &preAllocatedForL, Vec<int32_t> &preAllocatedForIndices, Vec<T>& rhsModifier) {

    KernelPrep kp = this->dim.kernelPrep();
    buildLaplacianKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAllocatedForL.toKernel2d(),
        this->dim, this->boundary,
        this->adjacencies,
        rhsModifier.toKernel1d()
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    this->adjacencies.loadMapRowToDiag(preAllocatedForIndices, stream);

    bandedL = std::make_unique<BandedMat<T>>(preAllocatedForL, preAllocatedForIndices);
    rhsBC = std::make_unique<Vec<T>>(rhsModifier);
}

template<typename T>
void Laplacian1d<T>::set(cudaStream_t stream, Mat<T> &preAllocatedForL_i, Vec<int32_t> &preAllocatedForIndices, size_t dim) {

    AdjacencyInd prev(1, -1), next(2, 1), primary(0, 0);

    KernelPrep kp(dim);

    buildL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAllocatedForL_i.toKernel2d(),
        this->boundary(dim, true), this->boundary(dim, false),
        primary, prev, next
    );

    CHECK_CUDA_ERROR(cudaGetLastError());

    AdjacencyPatern::loadMapRowToDiag(preAllocatedForIndices, {primary, prev, next}, stream);

    this->operator[](dim) = std::make_unique<BandedMat<T>>(preAllocatedForL_i, preAllocatedForIndices);
}

template<typename T>
void Laplacian1d<T>::set(cudaStream_t stream, Mat<T>* preAllocatedForL_iX3, Vec<int32_t> &preAllocatedForIndices) {

    AdjacencyInd prev(1, -1), next(2, 1), primary(0, 0);

    KernelPrep kp(std::max(preAllocatedForL_iX3[0].rows, std::max(preAllocatedForL_iX3[1].rows, preAllocatedForL_iX3[2].rows)));

    buildAllL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        preAllocatedForL_iX3[0].toKernel2d(), preAllocatedForL_iX3[1].toKernel2d(), preAllocatedForL_iX3[2].toKernel2d(),
        this->boundary,
        primary, prev, next
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    AdjacencyPatern::loadMapRowToDiag(preAllocatedForIndices, {primary, prev, next}, stream);

    for (size_t i = 0; i < 3; i++)
        this->operator[](i) = std::make_unique<BandedMat<T>>(preAllocatedForL_iX3[i], preAllocatedForIndices);
}

template<typename T>
void Laplacian1d<T>::set(cudaStream_t stream, size_t n, size_t dim) {
    Vec<int32_t> preAllocatedForIndices = Vec<int32_t>::create(3, stream);
    Mat<T> preAllocatedForL_i = Mat<T>::create(n, 3);
    set(stream, preAllocatedForL_i, preAllocatedForIndices, dim);
}

template<typename T>
void Laplacian1d<T>::set(cudaStream_t stream, const GridDim& dim) {
    Vec<int32_t> preAllocatedForIndices = Vec<int32_t>::create(3, stream);

    LaplcianConditions<T> laplcianConds[3] = {
        LaplcianConditions<T>(dim.cols, boundary.left, boundary.right),
        LaplcianConditions<T>(dim.rows, boundary.top, boundary.bottom),
        LaplcianConditions<T>(dim.layers, boundary.front, boundary.back)
    };

    std::shared_ptr<Mat<T>> preAllocatedForL_iX3[3];

    createUnique(laplcianConds, preAllocatedForL_iX3, [](const auto& c) {
        return Mat<T>::create(c.dim, 3);
    });

    Mat<T> mats[3] = {*preAllocatedForL_iX3[0], *preAllocatedForL_iX3[1], *preAllocatedForL_iX3[2]};
    set(stream, mats, preAllocatedForIndices);
}

template<typename T>
LaplacianEigenVec<T>::LaplacianEigenVec(
    const SquareMat<T>& eVecX,
    const SquareMat<T>& eVecY,
    const SquareMat<T>& eVecZ
) : eVecX(eVecX), eVecY(eVecY), eVecZ(eVecZ) {}

template<typename T>
BandedMat<T> & Laplacian<T>::banded(cudaStream_t stream) {
    if (bandedL == nullptr) setOperation(stream);
    return *bandedL;
}

template<typename T>
void Laplacian<T>::setRhsBC(cudaStream_t stream) {
    Vec<T> rhsModifier = SimpleArray<T>::create(dim.size(), stream);
    setRhsBC(stream, rhsModifier);
}

template<typename T>
void Laplacian<T>::setRhsBC(cudaStream_t stream, Vec<T>& rhsModifier) {
    KernelPrep kp(std::max(dim.rows, dim.layers), std::max(dim.layers, dim.cols));
    buildRhsBCKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(dim, boundary, rhsModifier.toKernel1d());
    CHECK_CUDA_ERROR(cudaGetLastError());
    rhsBC = std::make_unique<Vec<T>>(rhsModifier);
}


template class Laplacian<float>;
template class Laplacian<double>;
