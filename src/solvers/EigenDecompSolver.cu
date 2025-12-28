#include "solvers/EigenDecompSolver.h"

#include "Event.h"

template<typename T>
__device__ constexpr T PI = static_cast<T>(3.14159265358979323846);

// ============================================================================
//                                    Kernels
// ============================================================================

template<typename T>
__global__ void eigenMatLKernel(DeviceData2d<T> eVecs) {
    if (const GridInd2d ind; ind < eVecs) {
        eVecs[ind] = std::sqrt(2 / static_cast<T>(eVecs.rows + 1)) *
                     std::sin((ind.col + 1) * (ind.row + 1) * PI<T> / static_cast<T>(eVecs.rows + 1));
    }
}

template<typename T>
__global__ void eigenValLKernel(DeviceData1d<T> eVals, T delta) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < eVals.cols) {
        T s = std::sin((idx + 1) * PI<T> / (2 * (eVals.cols + 1)));
        eVals[idx] = -4 * s * s / (delta * delta);
    }
}

template<typename T>
__global__ void setUTildeKernel3d(DeviceData3d<T> uTilde,
                                const DeviceData2d<T> eVals,
                                const DeviceData3d<T> fTilde) {
    if (GridInd3d ind; ind < uTilde)
        uTilde[ind] = fTilde[ind] / (eVals(ind.col, 0) + eVals(ind.row, 1) + eVals(ind.layer, 2));

}

template<typename T>
__global__ void setUTildeKernel2d(DeviceData2d<T> uTilde,
                                const DeviceData2d<T> eVals,
                                const DeviceData2d<T> fTilde) {
    if (GridInd2d ind; ind < uTilde)
        uTilde[ind] = fTilde[ind] / (eVals(ind.col, 0) + eVals(ind.row, 1));

}

// ============================================================================
//                         EigenDecompSolver<T> Methods
// ============================================================================

template<typename T>
void EigenDecompSolver<T>::eigenL(size_t i, double delta, cudaStream_t stream) {
    KernelPrep kpVec = eVecs[i].kernelPrep();
    eigenMatLKernel<T><<<kpVec.numBlocks, kpVec.threadsPerBlock, 0, stream>>>(
        eVecs[i].toKernel2d());

    size_t n = eVecs[i]._cols;
    KernelPrep kpVal(n);
    eigenValLKernel<T><<<kpVal.numBlocks, kpVal.threadsPerBlock, 0, stream>>>(
        eVals.col(i).subVec(0, n, 1).toKernel1d(),
        delta
        );
}

template<typename T>
void EigenDecompSolver3d<T>::setUTilde(const Tensor<T> &f,
                                     Tensor<T> &u,
                                     Handle &hand) const{
    KernelPrep kp = f.kernelPrep();
    setUTildeKernel3d<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        u.toKernel3d(),
        this->eVals.toKernel2d(),
        f.toKernel3d());
}
template<typename T>
void EigenDecompSolver2d<T>::setUTilde(const Mat<T> &f, Mat<T> &u, Handle &hand) const{
    KernelPrep kp = f.kernelPrep();
    setUTildeKernel2d<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        u.toKernel2d(),
        this->eVals.toKernel2d(),
        f.toKernel2d());
}


template<typename T>
void EigenDecompSolver3d<T>::multE(size_t i,
                                 bool transposeEigen,
                                 bool transpose,
                                 const Mat<T> &a1,
                                 Mat<T> &dst1,
                                 size_t stride,
                                 Handle &hand,
                                 size_t batchCount) const{
    Mat<T>::batchMult(
        transpose ? a1 : this->eVecs[i], transpose ? stride : 0,
        transpose ? this->eVecs[i] : a1, transpose ? 0 : stride,
        dst1, stride,
        transpose ? false : transposeEigen,
        transpose ? transposeEigen : false,
        hand, batchCount
    );
}

template<typename T>
void EigenDecompSolver3d<T>::multEX(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const{
    multE(0, transposeE, true, src, dst, src._rows, hand, this->dim.layers);
}

template<typename T>
void EigenDecompSolver3d<T>::multEY(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const{
    multE(1, transposeE, false, src, dst, src._rows, hand, this->dim.layers);
}

template<typename T>
void EigenDecompSolver3d<T>::multEZ(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const{
    multE(2, transposeE, true, src, dst, this->dim.layers * this->dim.rows, hand, this->dim.cols);
}

template<typename T>
void EigenDecompSolver3d<T>::multiplyEF(Handle &hand, Tensor<T> &src, Tensor<T> &dst, bool transposeE) const{
    auto xF = src.layerRowCol(0), dx1 = dst.layerRowCol(0),
            yF = dst.layerRowCol(0), dyF = src.layerRowCol(0),
            zS = src.layerColDepth(0), dzS = dst.layerColDepth(0);

    if (transposeE) {
        multEX(xF, dx1, hand, transposeE);
        multEY(yF, dyF, hand, transposeE);
        multEZ(zS, dzS, hand, transposeE);
    } else {
        multEZ(zS, dzS, hand, transposeE);
        multEY(yF, dyF, hand, transposeE);
        multEX(xF, dx1, hand, transposeE);
    }
}

template<typename T>
EigenDecompSolver<T>::EigenDecompSolver(std::vector<SquareMat<T>> eMats,
                                        Mat<T> &maxDimX2Or3):
    dim(eMats[1]._rows, eMats[0]._cols, maxDimX2Or3._cols == 3? eMats[2]._rows: 0),
    eVecs(eMats),
    eVals(maxDimX2Or3) {

}

template<typename T>
EigenDecompSolver2d<T>::EigenDecompSolver2d(SquareMat<T> &rowsXRows, SquareMat<T> &colsXCols, Mat<T> &maxDimX2, std::array<Handle, 2> &hand2, Real2d delta)
    : EigenDecompSolver<T>({colsXCols, rowsXRows}, maxDimX2) {

    Event doneEigen;
    this->eigenL(1, delta.y, hand2[1]);
    doneEigen.record(hand2[1]);
    this->eigenL(0, delta.x, hand2[0]);
    doneEigen.wait(hand2[0]);
}

template<typename T>
EigenDecompSolver3d<T>::EigenDecompSolver3d(SquareMat<T> &rowsXRows, SquareMat<T> &colsXCols, SquareMat<T> &depthsXDepths, Mat<T> &maxDimX3, std::array<Handle, 3> &hand3, Real3d delta):
    EigenDecompSolver<T>({colsXCols, rowsXRows, depthsXDepths}, maxDimX3){
    Event doneEigen[2]{};

    this->eigenL(0, delta.x, hand3[1]);
    doneEigen[0].record(hand3[1]);

    this->eigenL(1, delta.y, hand3[2]);
    doneEigen[1].record(hand3[2]);

    this->eigenL(2, delta.z, hand3[0]);
    doneEigen[0].wait(hand3[0]);
    doneEigen[1].wait(hand3[0]);
}


template<typename T>
void EigenDecompSolver3d<T>::solve(Vec<T> &x, Vec<T> &b, Handle &hand) const {
    auto bT = b.tensor(this->dim.rows, this->dim.layers);
    auto xT = x.tensor(this->dim.rows, this->dim.layers);

    this->multiplyEF(hand, bT, xT, true);

    this->setUTilde(xT, bT, hand);

    this->multiplyEF(hand, bT, xT, false);
}

template<typename T>
void EigenDecompSolver2d<T>::solve(Vec<T> &x, Vec<T> &b, Handle &hand) const {//TODO:There's some redundant code here.
    auto bM = b.matrix(this->dim.rows);
    auto xM = x.matrix(this->dim.rows);

    bM.mult(this->eVecs[0], &xM, &hand, &Singleton<T>::ONE, &Singleton<T>::ZERO, false, false);
    this->eVecs[1].mult(xM, &bM, &hand, &Singleton<T>::ONE, &Singleton<T>::ZERO, true, false);

    setUTilde(bM, xM, hand);

    this->eVecs[1].mult(xM, &bM, &hand, &Singleton<T>::ONE, &Singleton<T>::ZERO, false, false);
    bM.mult(this->eVecs[0], &xM, &hand, &Singleton<T>::ONE, &Singleton<T>::ZERO, false, true);
}

template class EigenDecompSolver<double>;
template class EigenDecompSolver<float>;
template class EigenDecompSolver3d<double>;
template class EigenDecompSolver3d<float>;
template class EigenDecompSolver2d<double>;
template class EigenDecompSolver2d<float>;

// My b vector is (1,2,3,4,5,6)^T.  Let's walk through my code step by step, you tell me what you think my values should be, and I'll take a look at the debugger and
