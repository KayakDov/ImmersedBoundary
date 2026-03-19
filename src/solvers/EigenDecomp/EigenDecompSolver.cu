#include "EigenDecompSolver.h"

#include "../Event.h"



// ============================================================================
//                                    Kernels
// ============================================================================

template<typename T>
__host__ __device__ constexpr T PI = static_cast<T>(3.14159265358979323846);

/**
 * @brief Computes the eigenvector matrix of the 1D discrete Laplacian operator.
 *
 * This kernel fills a 2D matrix whose entries correspond to the analytical
 * eigenvectors of the standard second-order finite-difference Laplacian
 * with Dirichlet boundary conditions on a uniform grid.
 *
 * Each entry is given by:
 * \f[
 * V_{ij} = \sqrt{2\,\text{den}} \; \sin\!\left((i+1)(j+1)\pi\,\text{den}\right)
 * \f]
 * where \f$\text{den} = \frac{1}{N+1}\f$ and \f$N\f$ is the number of rows.
 *
 * Thread mapping:
 * - Each thread computes a single matrix entry \f$V_{ij}\f$.
 * - The 2D index is obtained via GridInd2d.
 *
 * @tparam T Floating-point type (e.g., float or double).
 *
 * @param[out] eVecs Device 2D array storing the eigenvector matrix.
 * @param[in]  den   Precomputed normalization factor \f$1/(N+1)\f$.
 *
 * @note
 * - Assumes Dirichlet boundary conditions.
 * - The resulting matrix is orthonormal up to floating-point error.
 * - `den` is invariant across threads and should be precomputed on the host.
 */
template<typename T>
__global__ void eigenMatLKernel(DeviceData2d<T> eVecs, const T den) {
    if (const GridInd2d ind; ind < eVecs)
        eVecs[ind] = std::sqrt(2 * den) * std::sin((ind.col + 1) * (ind.row + 1) * PI<T> * den);

}

/**
 * @brief Computes eigenvalues of the 1D discrete Laplacian using precomputed constants.
 *
 * This kernel evaluates the analytical eigenvalues of the standard second-order
 * finite-difference Laplacian with Dirichlet boundary conditions on a uniform grid.
 *
 * Each eigenvalue is given by:
 * \f[
 * \lambda_k = -\frac{4}{\Delta^2}
 * \sin^2\!\left(\frac{(k+1)\pi}{2(N+1)}\right)
 * \f]
 *
 * In this implementation, constant factors are precomputed on the host:
 * - \p minFourOverDeltaSq = \f$-4 / \Delta^2\f$
 * - \p piOverTwoNPlus1   = \f$\pi / (2(N+1))\f$
 *
 * This reduces per-thread arithmetic and avoids redundant divisions.
 *
 * Thread mapping:
 * - Each thread computes one eigenvalue \f$\lambda_k\f$ using linear indexing.
 *
 * @tparam T Floating-point type (e.g., float or double).
 *
 * @param[out] eVals Device 1D array storing eigenvalues.
 * @param[in]  minFourOverDeltaSq Precomputed factor \f$-4 / \Delta^2\f$.
 * @param[in]  piOverTwoNPlus1 Precomputed factor \f$\pi / (2(N+1))\f$.
 *
 * @note
 * - Assumes Dirichlet boundary conditions.
 * - Eigenvalues are strictly non-positive.
 * - This formulation minimizes per-thread arithmetic and improves instruction efficiency.
 */
template<typename T>
__global__ void eigenValLKernel(DeviceData1d<T> eVals, const T minFourOverDeltaSq, T const piOverTwoNPlus1) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < eVals.cols) {
        T s = std::sin((idx + 1) * piOverTwoNPlus1);
        eVals[idx] = s * s * minFourOverDeltaSq;
    }
}


// ============================================================================
//                         EigenDecompSolver<T> Methods
// ============================================================================


template<typename T>
void EigenDecompSolver<T>::eigenVecsL(size_t i, cudaStream_t stream) {
    KernelPrep kpVec = eVecs[i].kernelPrep();
    eigenMatLKernel<T><<<kpVec.numBlocks, kpVec.threadsPerBlock, 0, stream>>>(
        eVecs[i].toKernel2d(), 1/ static_cast<T>(eVecs[i]._rows + 1));
}

template<typename T>
void EigenDecompSolver<T>::eigenValsL(size_t i, const double delta, cudaStream_t stream) {

    KernelPrep kpVal = eVals[i].kernelPrep();
    eigenValLKernel<T><<<kpVal.numBlocks, kpVal.threadsPerBlock, 0, stream>>>(
        eVals[i].toKernel1d(),
        -4/(delta*delta),
        PI<T> / (2 * (eVals[i]._cols + 1))
    );
}

template<typename T>
void EigenDecompSolver<T>::eigenL(size_t i, const Real3d delta, cudaStream_t stream) {

    int32_t samePtrAs = -1, sameDimAs = -1;

    for (size_t j = 0; j < i; j++) {
        if (eVecs[i].ptr() == eVecs[j].ptr()) samePtrAs = j;
        if (dim[i] == dim[j]) sameDimAs = j;
    }

    if (samePtrAs < 0) {
        // if (sameDimAs < 0) {
            eigenVecsL(i, stream);
            eigenValsL(i, delta[i], stream);
        // }//TODO: find a way to restore this without creating a race condition, or remove code for sameAs.
        // else {//this creates a race condition
        //     auto scale = sizeOfB.get(i);
        //     scale.set(delta[sameDimAs] * delta[sameDimAs] / delta[i] / delta[i], stream);
        //     this->eVals[i].add(eVals[sameDimAs], scale, Singleton<T>::ZERO, stream);
        // }
    }
}

template<typename T>
void EigenDecompSolver<T>::appendMatAndVec(Mat<T>& src) {
    eVals.emplace_back(src.col(src._cols - 1));
    eVecs.emplace_back(src.sqSubMat(0, 0, src._rows));
}

template<typename T>
EigenDecompSolver<T>::EigenDecompSolver(std::vector<Mat<T> > eMatsAndVecs, SimpleArray<T> &sizeOfB) :
    dim(
        eMatsAndVecs[1]._rows,
        eMatsAndVecs[0]._rows,
        eMatsAndVecs.size() == 3 ? eMatsAndVecs[2]._rows : 1
    ),
    sizeOfB(sizeOfB) {

    eVals.reserve(eMatsAndVecs.size());
    eVecs.reserve(eMatsAndVecs.size());

    appendMatAndVec(eMatsAndVecs[0]);

    if (eMatsAndVecs[0].ptr() == eMatsAndVecs[1].ptr()) appendMatAndVec(eMatsAndVecs[0]);
    else appendMatAndVec(eMatsAndVecs[1]);

    if (eMatsAndVecs.size() == 3) {
        if (eMatsAndVecs[0].ptr() == eMatsAndVecs[2].ptr()) appendMatAndVec(eMatsAndVecs[0]);
        else if (eMatsAndVecs[1].ptr() == eMatsAndVecs[2].ptr()) appendMatAndVec(eMatsAndVecs[1]);
        else appendMatAndVec(eMatsAndVecs[2]);
    }
}

template<typename T>
EigenDecompSolver<T>::EigenDecompSolver(const GridDim& dim, const Real3d& delta, SimpleArray<T> sizeOfB):
    dim(dim),
    sizeOfB(sizeOfB.subArray(0, dim.size())) {
    auto xMat = Mat<T>::create(dim.cols, dim.cols + 1);
    appendMatAndVec(xMat);

    auto yMat = dim.rows == dim.cols && delta.x == delta.y ? xMat : Mat<T>::create(dim.rows, dim.rows + 1);
    appendMatAndVec(yMat);

    if (dim.numDims() == 3) {
        auto zMat = dim.rows == dim.layers && delta.y == delta.z ? yMat : (dim.cols == dim.layers && delta.x == delta.z ? xMat : Mat<T>::create(dim.layers, dim.layers + 1));
        appendMatAndVec(zMat);
    }
}

template<typename T>
EigenDecompSolver<T>::EigenDecompSolver(const GridDim& dim, const Real3d& delta, Handle& hand):
EigenDecompSolver(dim, delta, SimpleArray<T>::create(dim.size(), hand))
{}

template<typename T>
SquareMat<T> EigenDecompSolver<T>::inverseL(Handle &hand) const {
    auto id = SquareMat<T>::create(this->dim.size());
    id.setToIdentity(hand);
    auto result = SquareMat<T>::create(id._rows);
    for (size_t i = 0; i < result._rows; ++i) {
        auto src = id.col(i);
        auto dst = result.col(i);
        solve(dst, src, hand);
    }
    return result;
}

template class EigenDecompSolver<double>;
template class EigenDecompSolver<float>;

