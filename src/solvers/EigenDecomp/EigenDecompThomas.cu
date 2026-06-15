

#include "EigenDecompThomas.cuh"

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
 * @param seg The interior and boundary values of the tridaigonal matrix.
 */
template<typename Real, typename SegmentT>
__device__ void solveThomas3dLap(
    DeviceData1d<Real> rhs,
    DeviceData1d<Real> x,
    DeviceData1d<Real>& superPrimeBuffer,
    DeviceData1d<Real>& rhsPrimeBuffer,
    const SegmentT& seg,
    const bool singular
    ) {

    Real rowData[3];
    Real* row = rowData + 1;

    if (singular) {
        superPrimeBuffer[0] = 0;
        rhsPrimeBuffer[0] = 0;
    }else {
        seg.start.setL(row[0], row[1]);
        superPrimeBuffer[0] = row[1] / row[0];
        rhsPrimeBuffer[0] = rhs[0] / row[0];
    }

    size_t n = x.cols - 1;
    for (size_t i = 1; i < n; i++) {
        seg.setInteriorL(row[0], row[-1], row[1], i);
        Real denom = 1 / (row[0] - row[-1] * superPrimeBuffer[i - 1]);
        superPrimeBuffer[i] = row[1] * denom;
        rhsPrimeBuffer[i] = (rhs[i] - row[-1] * rhsPrimeBuffer[i - 1]) * denom;
    }
    seg.end.setL(row[0], row[-1]);
    Real denom = 1 / (row[0] - row[-1] * superPrimeBuffer[n - 1]);
    rhsPrimeBuffer[n] = (rhs[n] - row[-1] * rhsPrimeBuffer[n - 1]) * denom;

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
 * @param seg The boundary conditions for the x dimension.
 */
template<typename Real, typename SegmentT>
__global__ void solveThomas3dLaplacianKernel(//TODO: for the buffers, should I be using local shared memory instead of global memory?
    DeviceData3d<Real> x,
    DeviceData3d<Real> b,
    DeviceData1d<Real> eValsY,
    DeviceData1d<Real> eValsZ,
    DeviceData3d<Real> superPrime,
    DeviceData3d<Real> bPrime,
    SegmentT seg,
    bool isSingular
) {//width is layers and height is rows
    GridInd3d system(idy(), 0, idx());
    if (system.row >= x.rows || system.layer >= x.layers) return;

    DeviceData1d<Real> colX(x.cols, x, system, 0, 1, 0);//TODO: remove all the extra variables for speed improvement.
    DeviceData1d<Real> colB(b.cols, b, system, 0, 1, 0);
    DeviceData1d<Real> colSuperPrime(superPrime.cols, superPrime, system, 0, 1, 0);
    DeviceData1d<Real> colRHSPrime(bPrime.cols, bPrime, system, 0, 1, 0);

    solveThomas3dLap(
        colB,
        colX,
        colSuperPrime,
        colRHSPrime,
        seg,
        isSingular && system.row == 0 && system.layer == 0
    );
}

template<typename T, typename SegmentT>
void EigenDecompThomas<T, SegmentT>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {

    this->eigen.vecs.multCols(b, x, true, hand);
    this->eigen.vecs.multDepths(x, this->sizeOfB, true, hand);

    this->multLEigenValInverse(this->sizeOfB, x, hand);

    this->eigen.vecs.multCols(x, this->sizeOfB, false, hand);
    this->eigen.vecs.multDepths(this->sizeOfB, x, false, hand);
}


template<typename T, typename SegmentT>
void EigenDecompThomas<T, SegmentT>::multLEigenValInverse(const SimpleArray<T> &src, SimpleArray<T> &dst, Handle &hand) const {
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
template<typename T, typename SegmentT>
EigenDecompThomas<T, SegmentT>::EigenDecompThomas(const Eigen<T>& eigen, const SegmentT& boundX, Mat<T> &sizeOfBX3, bool isSingular):
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

template<typename T, typename SegmentT>
template<typename BoundaryConfigT>
EigenDecompThomas<T, SegmentT>::EigenDecompThomas(const BoundaryConfigT &boundary, Handle *hand3, Event *event2, Mat<T> sizeOfBX3):
    EigenDecomp3d<T>(boundary, hand3, event2, sizeOfBX3.col(0)),
    workSpaceSuperPrime(sizeOfBX3.col(1).tensor(boundary.dim().rows, boundary.dim().layers)),
    workSpaceRHSPrime(sizeOfBX3.col(2).tensor(boundary.dim().rows, boundary.dim().layers)),
    boundaryX(boundary.x)
{}

template<typename T, typename SegmentT>
template<typename BoundaryConfigT>
EigenDecompThomas<T, SegmentT>::EigenDecompThomas(const BoundaryConfigT &boundary, Handle *hand3, Event *event2):
    EigenDecompThomas(boundary, hand3, event2, Mat<T>::create(boundary.dim().size(), 3))
{
}
// ==============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// ==============================================================================

// 1. Instantiate the class itself for all precision and Segment types
template class EigenDecompThomas<float, UniformSegment<float>>;
template class EigenDecompThomas<double, UniformSegment<double>>;
template class EigenDecompThomas<float, VariableSegment<float>>;
template class EigenDecompThomas<double, VariableSegment<double>>;


// 2. Define macro to instantiate the templated constructors
// Note: We use <Real, SegX> for the class, and deduce the BoundaryConfigT from the args.
#define INSTANTIATE_EIGEN_THOMAS_BOUNDARY(Real, SegX, SegY, SegZ) \
template EigenDecompThomas<Real, SegX>::EigenDecompThomas( \
    const BoundaryConfig<Real, SegX, SegY, SegZ>&, Handle*, Event*); \
template EigenDecompThomas<Real, SegX>::EigenDecompThomas( \
    const BoundaryConfig<Real, SegX, SegY, SegZ>&, Handle*, Event*, Mat<Real>);


// 3. Generate all combinations of Uniform and Variable segments
#define INSTANTIATE_EIGEN_ALL(Real) \
INSTANTIATE_EIGEN_THOMAS_BOUNDARY(Real, UniformSegment<Real>,  UniformSegment<Real>,  UniformSegment<Real>)  \
INSTANTIATE_EIGEN_THOMAS_BOUNDARY(Real, UniformSegment<Real>,  UniformSegment<Real>,  VariableSegment<Real>) \
INSTANTIATE_EIGEN_THOMAS_BOUNDARY(Real, UniformSegment<Real>,  VariableSegment<Real>, UniformSegment<Real>)  \
INSTANTIATE_EIGEN_THOMAS_BOUNDARY(Real, UniformSegment<Real>,  VariableSegment<Real>, VariableSegment<Real>) \
INSTANTIATE_EIGEN_THOMAS_BOUNDARY(Real, VariableSegment<Real>, UniformSegment<Real>,  UniformSegment<Real>)  \
INSTANTIATE_EIGEN_THOMAS_BOUNDARY(Real, VariableSegment<Real>, UniformSegment<Real>,  VariableSegment<Real>) \
INSTANTIATE_EIGEN_THOMAS_BOUNDARY(Real, VariableSegment<Real>, VariableSegment<Real>, UniformSegment<Real>)  \
INSTANTIATE_EIGEN_THOMAS_BOUNDARY(Real, VariableSegment<Real>, VariableSegment<Real>, VariableSegment<Real>)

INSTANTIATE_EIGEN_ALL(float)
INSTANTIATE_EIGEN_ALL(double)