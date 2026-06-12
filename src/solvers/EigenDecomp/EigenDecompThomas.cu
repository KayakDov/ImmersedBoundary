

#include "EigenDecompThomas.cuh"


/**
 * The dridiagonal first and last rows.
 * @tparam Real
 */
template<typename Real>
struct TriDiagWithBoundary {
    Real firstRowPrimary, firstRowSecondary, lastRowPrimary, lastRowSecondary, interiorPrimary, interiorSecondary;
    __device__ TriDiagWithBoundary(
        const UniformSegment<Real>& boundary,
        Real eigenContribution,
        Real invDeltaSq
    ):
        firstRowPrimary(eigenContribution),
        firstRowSecondary(0),
        lastRowPrimary(eigenContribution),
        lastRowSecondary(0),
        interiorPrimary(-2* invDeltaSq + eigenContribution),
        interiorSecondary(invDeltaSq) {
        boundary.start.setL(firstRowPrimary, firstRowSecondary);
        boundary.end.setL(lastRowPrimary, lastRowSecondary);
    }
};

/**
 * @brief Core implementation of the Thomas Algorithm (TDMA) for a 1D tridiagonal system.
 * * This device function performs the forward elimination and back-substitution required
 * to solve a tridiagonal system of the form:
 * \f[ a_i x_{i-1} + b_i x_i + c_i x_{i+1} = r_i \f]
 * In this specific implementation, the sub-diagonal and super-diagonal coefficients are
 * assumed to be uniform (secondaryDiag), and the main diagonal is uniform (diagonal).
 *
 * @tparam Real Floating-point precision (float or double).
 * @param rhs Input Right-Hand Side vector.
 * @param x Output Solution vector.
 * @param superPrimeBuffer Workspace for the modified super-diagonal coefficients.
 * @param rhsPrimeBuffer Workspace for the modified intermediate RHS values.
 * @param triDiag The interior and boundary values of the tridaigonal matrix.
 */
template<typename Real>
__device__ void solveThomas3dLap(
    DeviceData1d<Real> rhs,
    DeviceData1d<Real> x,
    DeviceData1d<Real>& superPrimeBuffer,
    DeviceData1d<Real>& rhsPrimeBuffer,
    TriDiagWithBoundary<Real>& triDiag
    ) {

    superPrimeBuffer[0] = triDiag.firstRowSecondary / triDiag.firstRowPrimary;
    rhsPrimeBuffer[0] = rhs[0] / triDiag.firstRowPrimary;

    size_t n = x.cols - 1;
    for (size_t i = 1; i < n; i++) {
        Real denom = 1 / (triDiag.interiorPrimary - triDiag.interiorSecondary * superPrimeBuffer[i - 1]);
        superPrimeBuffer[i] = triDiag.interiorSecondary * denom;
        rhsPrimeBuffer[i] = (rhs[i] - triDiag.interiorSecondary * rhsPrimeBuffer[i - 1]) * denom;
    }
    Real denom = 1 / (triDiag.lastRowPrimary - triDiag.lastRowSecondary * superPrimeBuffer[n - 1]);
    rhsPrimeBuffer[n] = (rhs[n] - triDiag.lastRowSecondary * rhsPrimeBuffer[n - 1]) * denom;

    x[n] = rhsPrimeBuffer[n];
    for (int32_t i = n - 1; i >= 0; --i)
        x[i] = rhsPrimeBuffer[i] - superPrimeBuffer[i] * x[i + 1];
}

/**
 * @brief CUDA kernel that solves independent tridiagonal systems along the Z-axis (depth).
 * * This kernel maps each (x, y) coordinate of the 3D grid to a single CUDA thread. Each
 * thread solves a 1D tridiagonal system along the X-dimension. This is the "Semi-Direct"
 * step where the main diagonal is modified by the eigenvalues of the X and Y operators
 * to solve the 3D Helmholtz/Poisson equation in the partially transformed eigen-space.
 *
 *
 * @tparam Real Floating-point precision (float or double).
 * @param x Output 3D tensor for the solution.
 * @param b Input 3D tensor for the RHS (f-tilde).
 * @param eValsY Vector containing the eigenvalues of the Y-direction Laplacian.
 * @param eValsZ Vector containing the eigenvalues of the Z-direction Laplacian.
 * @param superPrime 3D workspace tensor for modified super-diagonals.
 * @param bPrime 3D workspace tensor for modified intermediate RHS.
 * @param boundsX The boundary conditions for the x dimension.
 */
template<typename Real>
__global__ void solveThomas3dLaplacianKernel(//TODO: for the buffers, should I be using local shared memory instead of global memory?
    DeviceData3d<Real> x,
    DeviceData3d<Real> b,
    DeviceData1d<Real> eValsY,
    DeviceData1d<Real> eValsZ,
    DeviceData3d<Real> superPrime,
    DeviceData3d<Real> bPrime,
    UniformSegment<Real> boundsX,
    bool isSingular
) {//width is layers and height is rows
    GridInd3d system(idy(), 0, idx());
    if (system.row >= x.rows || system.layer >= x.layers) return;

    DeviceData1d<Real> colX(x.cols, x, system, 0, 1, 0);//TODO: remove all the extra variables for speed improvement.
    DeviceData1d<Real> colB(b.cols, b, system, 0, 1, 0);
    DeviceData1d<Real> colSuperPrime(superPrime.cols, superPrime, system, 0, 1, 0);
    DeviceData1d<Real> colRHSPrime(bPrime.cols, bPrime, system, 0, 1, 0);
    Real deltaSquaredInv = boundsX.start.inverseDeltaSquared;

    TriDiagWithBoundary<Real> triDiag(
        boundsX,
        eValsY[system.row] + eValsZ[system.layer],
        deltaSquaredInv
    );

    Real origB = colB[0];

    if (isSingular && system.row == 0 && system.layer == 0) {
        triDiag.firstRowPrimary = static_cast<Real>(1.0);
        triDiag.firstRowSecondary = static_cast<Real>(0.0);
        colB[0] = static_cast<Real>(0.0);
    }

    solveThomas3dLap(
        colB,
        colX,
        colSuperPrime,
        colRHSPrime,
        triDiag
    );

    if (isSingular && system.row == 0 && system.layer == 0) colB[0] = origB;

}

template<typename T>
void EigenDecompThomas<T>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {

    this->eigen.vecs.multCols(b, x, true, hand);
    this->eigen.vecs.multDepths(x, this->sizeOfB, true, hand);

    this->multLEigenValInverse(this->sizeOfB, x, hand);

    this->eigen.vecs.multCols(x, this->sizeOfB, false, hand);
    this->eigen.vecs.multDepths(this->sizeOfB, x, false, hand);
}


template<typename T>
void EigenDecompThomas<T>::multLEigenValInverse(const SimpleArray<T> &src, SimpleArray<T> &dst, Handle &hand) const {
    KernelPrep kpVec( this->dim.layers, this->dim.rows);
    solveThomas3dLaplacianKernel<T><<<kpVec.numBlocks, kpVec.threadsPerBlock, 0, hand>>>(
        dst.tensor(this->dim.rows, this->dim.layers).toKernel3d(),
        src.tensor(this->dim.rows, this->dim.layers).toKernel3d(),
        this->eigen.vals.y.toKernel1d(),
        this->eigen.vals.z.toKernel1d(),
        workSpaceSuperPrime.toKernel3d(),
        workSpaceRHSPrime.toKernel3d(),
        boundaryX,
        this->isSingular
    );
}
//EigenDecomp3d(const poisson::Eigen<T> &eigen, SimpleArray<T>& sizeOfB, Vec<T>& size1IfSingular, bool isSingular);
template<typename T>
EigenDecompThomas<T>::EigenDecompThomas(const Eigen<T>& eigen, const UniformSegment<T>& boundX, Mat<T> &sizeOfBX3, bool isSingular):
    EigenDecomp3d<T>(
        eigen,
        sizeOfBX3.col(0),
        isSingular
    ),
    workSpaceSuperPrime(sizeOfBX3.col(1).tensor(eigen.dim().rows, eigen.dim().layers)),
    workSpaceRHSPrime(sizeOfBX3.col(2).tensor(eigen.dim().rows, eigen.dim().layers)),
    boundaryX(boundX)
{
}

template<typename T>
template<typename BoundaryConfigT>
EigenDecompThomas<T>::EigenDecompThomas(const BoundaryConfigT &boundary, Handle *hand3, Event *event2, Mat<T> sizeOfBX3):
    EigenDecomp3d<T>(boundary, hand3, event2, sizeOfBX3.col(0)),
    workSpaceSuperPrime(sizeOfBX3.col(1).tensor(boundary.dim().rows, boundary.dim().layers)),
    workSpaceRHSPrime(sizeOfBX3.col(2).tensor(boundary.dim().rows, boundary.dim().layers)),
    boundaryX(boundary.x)
{}

template<typename T>
template<typename BoundaryConfigT>
EigenDecompThomas<T>::EigenDecompThomas(const BoundaryConfigT &boundary, Handle *hand3, Event *event2):
    EigenDecompThomas(boundary, hand3, event2, Mat<T>::create(boundary.dim().size(), 3))
{
}


template class EigenDecompThomas<double>;
template class EigenDecompThomas<float>;