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
void EigenDecompSolver<T>::eigenVecsL(size_t i, cudaStream_t stream) {
    KernelPrep kpVec = eVecs[i].kernelPrep();
    eigenMatLKernel<T><<<kpVec.numBlocks, kpVec.threadsPerBlock, 0, stream>>>(
        eVecs[i].toKernel2d());
}

template<typename T>
void EigenDecompSolver<T>::eigenValsL(size_t i, const double delta, cudaStream_t stream) {
    size_t n = eVecs[i]._cols;
    KernelPrep kpVal(n);
    eigenValLKernel<T><<<kpVal.numBlocks, kpVal.threadsPerBlock, 0, stream>>>(
        eVals.col(i).subVec(0, n, 1).toKernel1d(),
        delta
    );
}

template<typename T>
void EigenDecompSolver<T>::eigenL(size_t i, const Real3d delta, cudaStream_t stream) {

    int32_t sameAs = -1;

    for (size_t j = 0; j < i; j++) if (eVecs[i].ptr() == eVecs[j].ptr()) {
            sameAs = j;
            break;
        }

    if (sameAs < 0) {
        eigenVecsL(i, stream);
        eigenValsL(i, delta[i], stream);
    } else if (delta[i] == delta[sameAs]) eVals.col(i).set(eVals.col(sameAs), stream);
    else  {
        auto scale = sizeOfB.get(i);
        scale.set(delta[sameAs] * delta[sameAs] / delta[i] / delta[i], stream);
        this->eVals.col(i).add(eVals.col(sameAs), scale, Singleton<T>::ZERO, stream);
    }
}

template<typename T>
void EigenDecompSolver3d<T>::setUTilde(const Tensor<T> &f,
                                       Tensor<T> &u,
                                       Handle &hand) const {
    KernelPrep kp = f.kernelPrep();
    setUTildeKernel3d<T><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        u.toKernel3d(),
        this->eVals.toKernel2d(),
        f.toKernel3d());
}

template<typename T>
void EigenDecompSolver2d<T>::setUTilde(const Mat<T> &f, Mat<T> &u, Handle &hand) const {
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
                                   size_t batchCount) const {
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
void EigenDecompSolver3d<T>::multEX(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const {
    multE(0, transposeE, true, src, dst, src._rows, hand, this->dim.layers);
}

template<typename T>
void EigenDecompSolver3d<T>::multEY(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const {
    multE(1, transposeE, false, src, dst, src._rows, hand, this->dim.layers);
}

template<typename T>
void EigenDecompSolver3d<T>::multEZ(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const {
    multE(2, transposeE, true, src, dst, this->dim.layers * this->dim.rows, hand, this->dim.cols);
}

template<typename T>
void EigenDecompSolver3d<T>::multiplyEF(Handle &hand, const Tensor<T> &src, Tensor<T> &dst, bool transposeE) const {
    const auto xF = src.layerRowCol(0);
    auto dx1 = dst.layerRowCol(0);
    auto yF = dst.layerRowCol(0);
    auto sizeOfBT = this->sizeOfB.tensor(this->dim.rows, this->dim.layers);
    auto dyF = sizeOfBT.layerRowCol(0);
    auto zS = sizeOfBT.layerColDepth(0);
    auto dzS = dst.layerColDepth(0);

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
EigenDecompSolver<T>::EigenDecompSolver(std::vector<SquareMat<T> > eMats,
                                        Mat<T> &maxDimX2Or3,
                                        SimpleArray<T> &sizeOfB) : dim(eMats[1]._rows, eMats[0]._cols,
                                                                       maxDimX2Or3._cols == 3 ? eMats[2]._rows : 1),
                                                                   eVecs(eMats),
                                                                   eVals(maxDimX2Or3),
                                                                   sizeOfB(sizeOfB) {
}

template<typename T>
EigenDecompSolver2d<T>::EigenDecompSolver2d(SquareMat<T> &rowsXRows, SquareMat<T> &colsXCols, Mat<T> &maxDimX2,
                                            SimpleArray<T> &sizeOfB, Handle* hand2, const Real2d delta, Event& event)
    : EigenDecompSolver<T>({colsXCols, rowsXRows}, maxDimX2, sizeOfB) {
    this->eigenL(1, delta, hand2[1]);
    event.record(hand2[1]);

    this->eigenL(0, delta, hand2[0]);
    event.hold(hand2[0]);
}

template<typename T>
EigenDecompSolver3d<T>::EigenDecompSolver3d(
    SquareMat<T> &rowsXRows,
    SquareMat<T> &colsXCols,
    SquareMat<T> &depthsXDepths,
    Mat<T> &maxDimX3,
    SimpleArray<T> &sizeOfB,
    Handle* hand3,
    Real3d delta,
    Event* event
) :
    EigenDecompSolver<T>({colsXCols, rowsXRows, depthsXDepths}, maxDimX3, sizeOfB) {


    this->eigenL(0, delta, hand3[1]);
    event[0].record(hand3[1]);

    this->eigenL(1, delta, hand3[2]);
    event[1].record(hand3[2]);
    event[1].hold(hand3[0]);

    this->eigenL(2, delta, hand3[0]);
    event[0].hold(hand3[0]);


}

//TODO: make sure this class is efficiently reuing memory if rowsXrows = colsXcols or the like.
template<typename T>
void EigenDecompSolver3d<T>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {
    const auto bT = b.tensor(this->dim.rows, this->dim.layers);
    auto bWorkSpaceT = this->sizeOfB.tensor(this->dim.rows, this->dim.layers);
    auto xT = x.tensor(this->dim.rows, this->dim.layers);

    this->multiplyEF(hand, bT, xT, true);

    this->setUTilde(xT, bWorkSpaceT, hand);

    this->multiplyEF(hand, bWorkSpaceT, xT, false);
}

template<typename T>
void EigenDecompSolver2d<T>::solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const {

    const auto bM = b.matrix(this->dim.rows);
    auto soBM = this->sizeOfB.matrix(this->dim.rows);
    auto xM = x.matrix(this->dim.rows);

    bM.mult(this->eVecs[0], &xM, &hand, false, false);
    this->eVecs[1].mult(xM, &soBM, &hand, true, false);

    setUTilde(soBM, xM, hand);

    this->eVecs[1].mult(xM, &soBM, &hand, false, false);
    soBM.mult(this->eVecs[0], &xM, &hand, false, true);
}

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
template class EigenDecompSolver3d<double>;
template class EigenDecompSolver3d<float>;
template class EigenDecompSolver2d<double>;
template class EigenDecompSolver2d<float>;
