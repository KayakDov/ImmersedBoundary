

#include "EigenDecompThomas.cuh"

#include "deviceArrays/headers/Support/Streamable.h"
#include "poisson/BoundaryConfig.cuh"

/**
 * @brief Core implementation of the Thomas Algorithm (TDMA) for a 1D tridiagonal system.
 * This device function performs the forward elimination and back-substitution required
 * to solve a tridiagonal system of the form:
 * \f[ a_i x_{i-1} + b_i x_i + c_i x_{i+1} = r_i \f]
 *
 * @tparam Real Floating-point precision (float or double).
 * @param rhs Input Right-Hand Side vector.
 * @param x Output Solution vector.
 * @param superPrimeBuffer Workspace for the modified super-diagonal coefficients.
 * @param rhsPrimeBuffer Workspace for the modified intermediate RHS values.
 * @param seg The interior and boundary values of the tridiagonal matrix.
 */
template<typename Real, typename SegmentT>
__device__ void solveThomas3dLap(
    DeviceData1d<Real> rhs,
    DeviceData1d<Real> x,
    DeviceData1d<Real>& superPrimeBuffer,
    DeviceData1d<Real>& rhsPrimeBuffer,
    const SegmentT& seg,
    const Real modifyMainDiag,
    const bool singular
    ) {

    Real rowData[3];
    Real* row = rowData + 1;

    if (singular) {
        superPrimeBuffer[0] = 0;
        rhsPrimeBuffer[0] = 0;
    }else {
        row[0] = modifyMainDiag;
        seg.start.setL(row[0], row[1]);
        superPrimeBuffer[0] = row[1] / row[0];
        rhsPrimeBuffer[0] = rhs[0] / row[0];
    }

    size_t n = x.cols - 1;
    for (size_t i = 1; i < n; i++) {
        row[0] = modifyMainDiag;
        seg.setInteriorL(row[0], row[-1], row[1], i);
        Real denom = 1 / (row[0] - row[-1] * superPrimeBuffer[i - 1]);
        superPrimeBuffer[i] = row[1] * denom;
        rhsPrimeBuffer[i] = (rhs[i] - row[-1] * rhsPrimeBuffer[i - 1]) * denom;
    }
    row[0] = modifyMainDiag;
    seg.end.setL(row[0], row[-1]);
    Real denom = 1 / (row[0] - row[-1] * superPrimeBuffer[n - 1]);
    rhsPrimeBuffer[n] = (rhs[n] - row[-1] * rhsPrimeBuffer[n - 1]) * denom;

    x[n] = rhsPrimeBuffer[n];
    for (int32_t i = n - 1; i >= 0; --i)
        x[i] = rhsPrimeBuffer[i] - superPrimeBuffer[i] * x[i + 1];
}

/**
 * @brief CUDA kernel that solves independent tridiagonal systems along the z-axis (layers).
 * This kernel maps each (x, y) pair of the 3D grid, that is each row and layer, to a single
 * CUDA thread. Each thread solves a 1D tridiagonal system along the Z-dimension. This is the
 * "Semi-Direct" step where the main diagonal is modified by the eigenvalues of the Y and Z
 * operators to solve the 3D Helmholtz/Poisson equation in the partially transformed eigen-space.
 *
 * @tparam Real Floating-point precision (float or double).
 * @param x Output 3D tensor for the solution.
 * @param b Input 3D tensor for the RHS (f-tilde).
 * @param eValsX Vector containing the eigenvalues of the X-direction Laplacian.
 * @param eValsY Vector containing the eigenvalues of the Y-direction Laplacian.
 * @param superPrime 3D workspace tensor for modified super-diagonals.
 * @param bPrime 3D workspace tensor for modified intermediate RHS.
 * @param seg The boundary conditions for the z dimension.
 */
template<typename Real, typename SegmentT>
__global__ void solveThomas3dLaplacianKernel(//TODO: for the buffers, should I be using local shared memory instead of global memory?
    DeviceData3d<Real> x,
    DeviceData3d<Real> b,
    DeviceData1d<Real> eValsX,
    DeviceData1d<Real> eValsY,
    DeviceData3d<Real> superPrime,
    DeviceData3d<Real> bPrime,
    SegmentT seg,
    bool isSingular,
    const Real helmholtzShift
) {//width is rows and height is cols.
    GridInd3d system(idx(), idy(), 0);
    if (system.row >= x.rows || system.col >= x.cols) return;

    DeviceData1d<Real> depthX(x.layers, x, system, 0, 0, 1);
    DeviceData1d<Real> depthB(b.layers, b, system, 0, 0, 1);
    DeviceData1d<Real> depthSuperPrime(superPrime.layers, superPrime, system, 0, 0, 1);
    DeviceData1d<Real> depthRHSPrime(bPrime.layers, bPrime, system, 0, 0, 1);

    solveThomas3dLap(
        depthB,
        depthX,
        depthSuperPrime,
        depthRHSPrime,
        seg,
        eValsX[system.col] + eValsY[system.row] - helmholtzShift,
        isSingular && system.row == 0 && system.col == 0
    );
}

template<typename T, typename SegmentT>
void EigenDecompThomas<T, SegmentT>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {

    this->lapEigen.vecsInv.mult2d(b, this->sizeOfB, x, hand);

    this->multLEigenValInverse(this->sizeOfB, x, hand);

    this->lapEigen.vecs.mult2d(x, x, this->sizeOfB, hand);
}


template<typename T, typename SegmentT>
void EigenDecompThomas<T, SegmentT>::multLEigenValInverse(const SimpleArray<T> &src, SimpleArray<T> &dst, Handle &hand) const {
    KernelPrep kpVec( this->dim.rows, this->dim.cols);
    solveThomas3dLaplacianKernel<T><<<kpVec.numBlocks, kpVec.threadsPerBlock, 0, hand>>>(
        dst.tensor(this->dim.rows, this->dim.layers).toKernel3d(),
        src.tensor(this->dim.rows, this->dim.layers).toKernel3d(),
        this->lapEigen.vals.x.toKernel1d(),
        this->lapEigen.vals.y.toKernel1d(),
        workSpaceSuperPrime.toKernel3d(),
        workSpaceRHSPrime.toKernel3d(),
        boundaryZ.forDevice(),
        this->isSingular,
        this->helmholtzShift
    );
}
template<typename T, typename SegmentT>
EigenDecompThomas<T, SegmentT>::EigenDecompThomas(const Eigen<T>& eigen, const AxisSegmentHost<SegmentT>& boundZ, Mat<T> &sizeOfBX3, bool isSingular, T helmholtzShift):
    EigenDecomp3d<T>(
        eigen,
        sizeOfBX3.col(0),
        isSingular,
        helmholtzShift
    ),
    workSpaceSuperPrime(sizeOfBX3.col(1).tensor(eigen.dim().rows, eigen.dim().layers)),
    workSpaceRHSPrime(sizeOfBX3.col(2).tensor(eigen.dim().rows, eigen.dim().layers)),
    boundaryZ(boundZ)
{
}


template<typename T, typename SegmentT>
template<typename SegX, typename SegY>
    EigenDecompThomas<T, SegmentT>::EigenDecompThomas(const BoundaryConfigHost<T, SegX, SegY, SegmentT>& boundary, Handles &hands, Mat<T> sizeOfBX3, T helmholtzShift):
        EigenDecomp3d<T>(boundary.forDevice(), hands, sizeOfBX3.col(0), helmholtzShift),
        workSpaceSuperPrime(sizeOfBX3.col(1).tensor(boundary.forDevice().dim().rows, boundary.forDevice().dim().layers)),
        workSpaceRHSPrime(sizeOfBX3.col(2).tensor(boundary.forDevice().dim().rows, boundary.forDevice().dim().layers)),
        boundaryZ(boundary.z)
{}

template<typename T, typename SegmentT>
template<typename SegX, typename SegY>
    EigenDecompThomas<T, SegmentT>::EigenDecompThomas(const BoundaryConfigHost<T, SegX, SegY, SegmentT>& boundary, Handles &hands, T helmholtzShift):
        EigenDecompThomas(boundary, hands, Mat<T>::create(boundary.forDevice().dim().size(), 3, hands[0]), helmholtzShift)
{
}
// ==============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// ==============================================================================

// 1. Explicitly instantiate the class for all required segment types
template class EigenDecompThomas<double, FluxLaplacian<double>>;
template class EigenDecompThomas<double, UniformSegment<double>>;
template class EigenDecompThomas<double, VariableSegment<double>>;

template class EigenDecompThomas<float, FluxLaplacian<float>>;
template class EigenDecompThomas<float, UniformSegment<float>>;
template class EigenDecompThomas<float, VariableSegment<float>>;

// 2. Macro for the Thomas constructor
// SegY and SegZ are deduced from the BoundaryConfigHost parameter, so they are
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
    Mat<Real> sizeOfBX3,                                                 \
    Real helmholtzShift                                                  \
);

// 3. Apply permutations
APPLY_TO_ALL_SEGMENT_COMBOS(double, INSTANTIATE_EIGEN_DECOMP_THOMAS_CONSTRUCTORS)
APPLY_TO_ALL_SEGMENT_COMBOS(float, INSTANTIATE_EIGEN_DECOMP_THOMAS_CONSTRUCTORS)

