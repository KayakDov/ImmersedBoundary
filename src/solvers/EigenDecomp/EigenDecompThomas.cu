

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
 * @param superPrime Workspace for the modified super-diagonal coefficients.
 * @param rhsPrime Workspace for the modified intermediate RHS values.
 * @param diagonal The value of the main diagonal elements.
 * @param secondaryDiag The value of the sub and super diagonal elements.
 */
template<typename Real>
__device__ void solveThomas3dLap(DeviceData1d<Real> rhs, DeviceData1d<Real> x, DeviceData1d<Real>& superPrime, DeviceData1d<Real>& rhsPrime, Real diagonal, Real secondaryDiag) {

    superPrime[0] = secondaryDiag / diagonal;
    rhsPrime[0] = rhs[0] / diagonal;
    for (size_t i = 1; i < x.cols; i++) {
        Real denom = 1 / (diagonal - secondaryDiag * superPrime[i - 1]);
        superPrime[i] = secondaryDiag * denom;
        rhsPrime[i] = (rhs[i] - secondaryDiag * rhsPrime[i - 1]) * denom;
    }
    size_t n = x.cols - 1;
    x[n] = rhsPrime[n];
    for (int32_t col = n - 1; col >= 0; --col)
        x[col] = rhsPrime[col] - superPrime[col] * x[col + 1];
}

/**
 * @brief CUDA kernel that solves independent tridiagonal systems along the Z-axis (depth).
 * * This kernel maps each (x, y) coordinate of the 3D grid to a single CUDA thread. Each
 * thread solves a 1D tridiagonal system along the Z-dimension. This is the "Semi-Direct"
 * step where the main diagonal is modified by the eigenvalues of the X and Y operators
 * to solve the 3D Helmholtz/Poisson equation in the partially transformed eigen-space.
 *
 *
 * @tparam Real Floating-point precision (float or double).
 * @param x Output 3D tensor for the solution.
 * @param b Input 3D tensor for the RHS (f-tilde).
 * @param eValsX Vector containing the eigenvalues of the X-direction Laplacian.
 * @param eValsY Vector containing the eigenvalues of the Y-direction Laplacian.
 * @param superPrime 3D workspace tensor for modified super-diagonals.
 * @param bPrime 3D workspace tensor for modified intermediate RHS.
 * @param deltaZSquaredInv The precomputed value of \f$ 1/\Delta z^2 \f$.
 */
template<typename Real>
__global__ void solveThomas3dLaplacianDepthsKernel(DeviceData3d<Real> x, DeviceData3d<Real> b, DeviceData1d<Real> eValsX, DeviceData1d<Real> eValsY, DeviceData3d<Real> superPrime, DeviceData3d<Real> bPrime, Real deltaZSquaredInv) {
    GridInd3d system(idy(), idx(), 0);
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
        -2*deltaZSquaredInv + eValsX[system.row] + eValsY[system.col],
        deltaZSquaredInv
    );
}


template<typename T>
void EigenDecompThomas<T>::multEZ(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const {
}

template<typename T>
void EigenDecompThomas<T>::setUTilde(const Tensor<T> &src, Tensor<T> &dst, Handle &hand) const {
    KernelPrep kpVec(this->dim.cols, this->dim.rows);
    solveThomas3dLaplacianDepthsKernel<T><<<kpVec.numBlocks, kpVec.threadsPerBlock, 0, hand>>>(
        dst.toKernel3d(),
        src.toKernel3d(),
        this->eVals[0].toKernel1d(),
        this->eVals[1].toKernel1d(),
        workSpaceSuperPrime.toKernel3d(),
        workSpaceRHSPrime.toKernel3d(),
        1/deltaZ/deltaZ
    );
}


//Mat<T> &rowsXRowsP1, Mat<T> &colsXColsP1, Mat<T> &depthsXDepthsP1, SimpleArray<T> &sizeOfB, Handle *hand3, Real3d delta, Event *event
template<typename T>
EigenDecompThomas<T>::EigenDecompThomas(Mat<T> &rowsXRowsP1, Mat<T> &colsXColsP1, Mat<T> &depthsXDepthsP1, Mat<T> &sizeOfBX3, Handle *hand3, const Real3d& delta, Event *event):
    EigenDecomp3d<T>(
        rowsXRowsP1,
        colsXColsP1,
        depthsXDepthsP1,
        sizeOfBX3.col(0),
        hand3,
        delta,
        event
    ),
    workSpaceSuperPrime(sizeOfBX3.col(1).tensor(rowsXRowsP1._rows, depthsXDepthsP1._rows)),
    workSpaceRHSPrime(sizeOfBX3.col(2).tensor(rowsXRowsP1._rows, depthsXDepthsP1._rows)),
    deltaZ(delta.z)
{
}

template<typename T>
EigenDecompThomas<T>::EigenDecompThomas(const GridDim& dim, Handle *hand3, const Real3d& delta, Mat<T> sizeOfBX3, Event *event):
    EigenDecomp3d<T>(dim, hand3, delta, sizeOfBX3.col(0), event),
    workSpaceSuperPrime(sizeOfBX3.col(1).tensor(dim.rows, dim.layers)),
    workSpaceRHSPrime(sizeOfBX3.col(2).tensor(dim.rows, dim.layers)),
    deltaZ(delta.z)
{}

template<typename T>
EigenDecompThomas<T>::EigenDecompThomas(const GridDim& dim, Handle *hand3, const Real3d& delta, Event *event):
    EigenDecompThomas(dim, hand3, delta, Mat<T>::create(dim.size(), 3), event) {
}

template class EigenDecompThomas<double>;
template class EigenDecompThomas<float>;