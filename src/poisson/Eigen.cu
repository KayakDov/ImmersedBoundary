//
// Created by usr on 6/11/26.
//

#include "Eigen.cuh"
#include "BoundaryConfig.cuh"
#include "deviceArrays/headers/sparse/BandedMat.h"
#include "deviceArrays/headers/SquareMat.h"
#include "kronecker/KroneckerTriplet.h"
#include "math/XYZ.cuh"
#include "solvers/Event.h"


template<typename T>
void Eigen<T>::generateEigen(Handle& hand, SquareMat<T> eVecs, Vec<T> eVals, const VariableSegment<T> &axisSegment) {

    auto buffer = Mat<T>::create(eVecs._rows, 3 + eVals.size());
    auto rawBanded = buffer.subMat(0,0,eVals.size(), 3);
    AdjacencyInd primary(0, 0);
    AdjacencyIndPair superSub(1, 1);
    KernelPrep kp(3, eVecs._rows);
    buildL1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(rawBanded, axisSegment, primary, superSub);
    CHECK_CUDA_ERROR(cudaGetLastError());
    auto indices = SimpleArray<int32_t>::create(3, hand);

    std::vector<int32_t> indicesHost(3, 0);
    indicesHost[primary.colInBanded] = primary.diag;
    indicesHost[superSub.left.colInBanded] = superSub.left.diag;
    indicesHost[superSub.right.colInBanded] = superSub.right.diag;
    indices.set(indicesHost.data(), hand);

    BandedMat<T> banded(rawBanded, indices);

    auto dense = buffer.sqSubMat(0, 3, eVals.size());
    banded.getDense(dense, hand);

    dense.eigen(eVals, &eVecs, hand);
}

/**
* True if i is the first index that these conditions appear at.
* @param i The index to be checked.
* @return the value of the index repeated, or -1 if this is the first appearence.
*/
template <typename Real>
int repeat(const BoundaryConfig<Real>& boundary, int i) {
    for (size_t j = 0; j < i; ++j) if (boundary[j] == boundary[i]) return j;
    return -1;

}

/**
 * @brief Efficiently creates unique objects based on 1D Laplacian boundary conditions.
 *
 * This utility iterates through the X, Y, and Z dimensions of a Laplacian setup.
 * If the boundary conditions (dimensions, spacing, and types) for two dimensions
 * are identical, the function reuses the existing object by sharing ownership
 * via std::shared_ptr. This prevents redundant memory allocations and unnecessary
 * re-computation of Laplacian matrices or eigenvalue vectors.
 *
 * @tparam ResultType The type of object to be created (e.g., Mat<T> or Vec<T>).

 * @param[out] outputs  An array of 3 shared_ptrs to be populated with the results.
 * @param[in]  factory  A factory function: `ResultType factory(const LaplacianConditions<T>&)`.
 * * @note Because ResultType is wrapped in a shared_ptr, ResultType does not need
 * to be assignable, which is critical for classes with const data members.
 */
template <typename Real, typename ResultType>
void createUnique(const BoundaryConfig<ResultType>& boundaryConfig, std::shared_ptr<ResultType> (&outputs)[3], std::function<ResultType(const UniformSegment<Real>&)> factory){
    size_t numDim = boundaryConfig.dim().numDims();
    for (size_t i = 0; i <  numDim; ++i) {
        int repeatInd = repeat(boundaryConfig, i);
        if (repeatInd == -1) outputs[i] = std::make_shared<ResultType>(factory(boundaryConfig[i]));
        else outputs[i] = outputs[repeatInd];
    }
    if (numDim == 2) outputs[2] = nullptr;
}

/**
* Generates, including memory allocation, eigen values and vectors.  The matrices pointed to, that are retruned,
* hold the values in the last column, and the vectors in the first nxn cells.
* @param hands3 Used to create the different vectors in parrallel.  The number of handles should be equal to the number of dimentisons.
* @param events The number of events should be equal to the number of dimesnions.
* @param preAllocatedForL_iX3
* @return pointers to matrices containing the eigen values and vectors.
*/
template<typename Real>
void Eigen<Real>::generateEigen(const BoundaryConfig<Real> boundary, const Handle *hands3, Event *events, std::shared_ptr<Mat<Real>> (&preAllocatedForL_iX3)[3]) {

createUnique<Mat<Real>>(preAllocatedForL_iX3, [](const UniformSegment<Real>& c) {
    return Mat<Real>::create(c.numNodes, c.numNodes + 1);
});

boundary[0].generateEigen(hands3[0], *(preAllocatedForL_iX3[0]));

for (size_t i = 1; i < boundary.dim().numDims(); ++i)
    if (boundary.repeat(i) < 0){
        boundary[i].generateEigen(hands3[i], *(preAllocatedForL_iX3[i]));
        events[i - 1].record(hands3[i]);
        events[i - 1].hold(hands3[0]);
    }
}


template<typename T>
Eigen<T>::Eigen(const XYZ<Vec<T>> &vals, const XYZ<SquareMat<T>> &vecs) :
    vals(vals), vecs(vecs) {}


template<typename T>
Eigen<T> Eigen<T>::make(const BoundaryConfig<T>& boundary, Handle* hands3, Event* events) {

    bool is3d = boundary.dim().numDims() == 3;
    std::shared_ptr<Mat<T>> eigen[3];
    generateEigen(boundary, hands3, events, eigen);
    XYZ<Vec<T>> vals(eigen[0]->lastCol(), eigen[1]->lastCol(), is3d ? eigen[2]->lastCol() : SimpleArray<T>::empty());
    XYZ<SquareMat<T>> vecs(
        eigen[0]->sqSubMatFirstBiggest(),
        eigen[1]->sqSubMatFirstBiggest(),
        is3d ? eigen[2]->sqSubMatFirstBiggest() : SquareMat<T>::empty()//GPUConst<T>::get(0).matrix(1).sqSubMat(0,0,1)
    );
    return Eigen<T>(vals, vecs);
}

template<typename T>
GridDim Eigen<T>::dim() const {
    return GridDim(vals.y.size(), vals.x.size(), vals.z.size());
}

template class Eigen<float>;
template class Eigen<double>;