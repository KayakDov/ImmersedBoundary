#include "EigenDecompThomas.cuh"

#include <stdexcept>

#include "deviceArrays/headers/Support/Streamable.h"
#include "poisson/BoundaryConfig.cuh"

/**
 * @brief The Thomas algorithm (TDMA) for one tridiagonal system.
 *
 * Solves \f[ a_i x_{i-1} + b_i x_i + c_i x_{i+1} = r_i \f], where seg supplies the off-diagonal and
 * boundary coefficients and modifyMainDiag is added to the main diagonal (the sum of the eigenvalues
 * of the diagonalized axes, minus the Helmholtz shift).
 *
 * Works in rhs and x alone:
 * - superPrime is another name for rhs. Each modified super-diagonal coefficient is written over the
 *   right-hand-side entry at the same index, just after that entry is used to compute x there.
 * - x holds the modified right-hand side during elimination and is overwritten in place by back
 *   substitution.
 *
 * Requires x.cols >= 2.
 *
 * @tparam Real Floating-point precision (float or double).
 * @param rhs The right-hand side. Overwritten with the modified super-diagonal coefficients.
 * @param x Receives the solution; also holds the modified right-hand side during elimination. Must
 * not be the same memory as rhs.
 * @param seg The interior and boundary values of the tridiagonal matrix.
 * @param modifyMainDiag Added to the main diagonal.
 * @param singular True for the one singular system (all Neumann, zero eigenvalue): x[0] is pinned to 0.
 */
template<typename Real, typename SegmentT>
__device__ void solveThomas3dLap(
    DeviceData1d<Real>& rhs,
    DeviceData1d<Real>& x,
    const SegmentT& seg,
    const Real modifyMainDiag,
    const bool singular
    ) {

    DeviceData1d<Real>& superPrime = rhs;
    Real rowData[3];
    Real* row = rowData + 1;

    if (singular) {
        x[0] = 0;
        superPrime[0] = 0;
    } else {
        row[0] = modifyMainDiag;
        seg.start.setL(row[0], row[1]);
        x[0] = rhs[0] / row[0];
        superPrime[0] = row[1] / row[0];
    }

    const size_t n = x.cols - 1;
    for (size_t i = 1; i < n; i++) {
        row[0] = modifyMainDiag;
        seg.setInteriorL(row[0], row[-1], row[1], i);
        const Real denom = 1 / (row[0] - row[-1] * superPrime[i - 1]);
        x[i] = (rhs[i] - row[-1] * x[i - 1]) * denom;
        superPrime[i] = row[1] * denom;
    }
    row[0] = modifyMainDiag;
    seg.end.setL(row[0], row[-1]);
    const Real denom = 1 / (row[0] - row[-1] * superPrime[n - 1]);
    x[n] = (rhs[n] - row[-1] * x[n - 1]) * denom;

    for (size_t i = n; i-- > 0; )
        x[i] -= superPrime[i] * x[i + 1];
}

/**
 * @brief CUDA kernel that solves the independent tridiagonal systems along Z (layers).
 *
 * One thread per (row, col) pair, that is per (y, x) pair. threadIdx.x runs along rows, the fastest
 * index in memory, so at each step along z adjacent threads touch adjacent addresses. Each system's
 * main diagonal is shifted by the X and Y eigenvalues of its (x, y) pair.
 *
 * @tparam Real Floating-point precision (float or double).
 * @param x Output tensor (u-tilde); also the modified right-hand side during elimination.
 * @param b Input tensor (f-tilde). Overwritten with the modified super-diagonal coefficients.
 * @param eValsX Eigenvalues of the X-direction Laplacian.
 * @param eValsY Eigenvalues of the Y-direction Laplacian.
 * @param seg The boundary conditions for the z dimension.
 * @param isSingular True if the Laplacian is singular.
 * @param helmholtzShift The Helmholtz shift sigma.
 */
template<typename Real, typename SegmentT>
__global__ void solveThomas3dLaplacianKernel(
    DeviceData3d<Real> x,
    DeviceData3d<Real> b,
    DeviceData1d<Real> eValsX,
    DeviceData1d<Real> eValsY,
    SegmentT seg,
    bool isSingular,
    const Real helmholtzShift
) {
    GridInd3d system(idx(), idy(), 0);
    if (system.row >= x.rows || system.col >= x.cols) return;

    DeviceData1d<Real> depthX(x.layers, x, system, 0, 0, 1);
    DeviceData1d<Real> depthB(b.layers, b, system, 0, 0, 1);

    solveThomas3dLap(
        depthB,
        depthX,
        seg,
        eValsX[system.col] + eValsY[system.row] - helmholtzShift,
        isSingular && system.row == 0 && system.col == 0
    );
}

template<typename T, typename SegmentT>
void EigenDecompThomas<T, SegmentT>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {

    this->lapEigen.vecsInv.mult2d(b, this->sizeOfB, x, hand);

    solveTridiagonalSystems(this->sizeOfB, x, hand);

    this->lapEigen.vecs.mult2d(x, x, this->sizeOfB, hand);
}

template<typename T, typename SegmentT>
void EigenDecompThomas<T, SegmentT>::solveTridiagonalSystems(SimpleArray<T> &rhs, SimpleArray<T> &dst, Handle &hand) const {
    if (dst.data() == rhs.data())
        throw std::invalid_argument("EigenDecompThomas: the tridiagonal solve's output must be separate memory from its right-hand side, which it also uses as its workspace.");

    KernelPrep kpVec(this->dim.rows, this->dim.cols);
    solveThomas3dLaplacianKernel<T><<<kpVec.numBlocks, kpVec.threadsPerBlock, 0, hand>>>(
        dst.tensor(this->dim.rows, this->dim.layers).toKernel3d(),
        rhs.tensor(this->dim.rows, this->dim.layers).toKernel3d(),
        this->lapEigen.vals.x.toKernel1d(),
        this->lapEigen.vals.y.toKernel1d(),
        boundaryZ.forDevice(),
        this->isSingular,
        this->helmholtzShift
    );
}

template<typename T, typename SegmentT>
void EigenDecompThomas<T, SegmentT>::requireTwoLayers() const {
    if (this->dim.layers < 2)
        throw std::invalid_argument("EigenDecompThomas needs at least 2 layers: it solves a tridiagonal system along z. Use EigenDecomp2d for a single layer.");
}

template<typename T, typename SegmentT>
EigenDecompThomas<T, SegmentT>::EigenDecompThomas(const Eigen<T>& eigen, const AxisSegmentHost<SegmentT>& boundZ, SimpleArray<T> sizeOfB, bool isSingular, T helmholtzShift):
    EigenDecomp3d<T>(eigen, sizeOfB, isSingular, helmholtzShift),
    boundaryZ(boundZ) {
    requireTwoLayers();
}

template<typename T, typename SegmentT>
template<typename SegX, typename SegY>
EigenDecompThomas<T, SegmentT>::EigenDecompThomas(const BoundaryConfigHost<T, SegX, SegY, SegmentT>& boundary, Handles &hands, SimpleArray<T> sizeOfB, T helmholtzShift):
    EigenDecomp3d<T>(boundary.forDevice(), hands, sizeOfB, helmholtzShift),
    boundaryZ(boundary.z) {
    requireTwoLayers();
}

template<typename T, typename SegmentT>
template<typename SegX, typename SegY>
EigenDecompThomas<T, SegmentT>::EigenDecompThomas(const BoundaryConfigHost<T, SegX, SegY, SegmentT>& boundary, Handles &hands, T helmholtzShift):
    EigenDecomp3d<T>(boundary.forDevice(), hands, helmholtzShift),
    boundaryZ(boundary.z) {
    requireTwoLayers();
}

// ==============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// ==============================================================================

// 1. Explicitly instantiate the class for all required Z segment types
template class EigenDecompThomas<double, FluxLaplacian<double>>;
template class EigenDecompThomas<double, UniformSegment<double>>;
template class EigenDecompThomas<double, VariableSegment<double>>;

template class EigenDecompThomas<float, FluxLaplacian<float>>;
template class EigenDecompThomas<float, UniformSegment<float>>;
template class EigenDecompThomas<float, VariableSegment<float>>;

// 2. Macro for the templated constructors
// SegX and SegY are deduced from the BoundaryConfigHost parameter, so they are
// not repeated in the constructor name.
#define INSTANTIATE_EIGEN_DECOMP_THOMAS_CONSTRUCTORS(Real, SegX, SegY, SegZ) \
template EigenDecompThomas<Real, SegZ>::EigenDecompThomas(               \
    const BoundaryConfigHost<Real, SegX, SegY, SegZ>& boundary,          \
    Handles& hands,                                                      \
    Real helmholtzShift                                                  \
);                                                                       \
template EigenDecompThomas<Real, SegZ>::EigenDecompThomas(               \
    const BoundaryConfigHost<Real, SegX, SegY, SegZ>& boundary,          \
    Handles& hands,                                                      \
    SimpleArray<Real> sizeOfB,                                           \
    Real helmholtzShift                                                  \
);

// 3. Apply permutations
APPLY_TO_ALL_SEGMENT_COMBOS(double, INSTANTIATE_EIGEN_DECOMP_THOMAS_CONSTRUCTORS)
APPLY_TO_ALL_SEGMENT_COMBOS(float, INSTANTIATE_EIGEN_DECOMP_THOMAS_CONSTRUCTORS)

