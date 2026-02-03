#include "ImerssedEquation.h"

#include "Streamable.h"
#include "solvers/EigenDecompSolver.h"
#include "solvers/Event.h"
#include <span>

//todo: merge base Data, ImmerssedEq, and the solver.

FileMeta::FileMeta(std::ifstream &xFile, std::ifstream &bFile) : xFile(xFile), bFile(bFile) {
    {
        bFile.read((char *) &bRows, 8);
        bFile.read((char *) &bCols, 8);
        bFile.read((char *) &nnz, 8);

        uint64_t n;
        xFile.read((char *) &n, 8);

        fSize = bRows;
        pSize = bCols;
    }
}

template<typename Real, typename Int>
void BaseData<Real, Int>::checkNNZB(size_t nnzB) const {
    if (nnzB > maxSparseVals.size()) {
        throw std::invalid_argument(
            "BaseData::setB - NNZ Overflow: Requested nnzB (" + std::to_string(nnzB) +
            ") exceeds maxB capacity (" + std::to_string(maxSparseVals.size()) + ")."
        );
    }
}

template<typename Real, typename Int>
BaseData<Real, Int>::BaseData(const FileMeta &meta, const GridDim &dim, double dT, Handle& hand) :
    maxSparseInds(SimpleArray<Int>::create(meta.nnz, hand)),
    maxSparseOffsets(SimpleArray<Int>::create(meta.fSize + 1, hand)),
    delta(1.0 / dim.cols, 1.0 / dim.rows, 1.0 / dim.layers),
    dim(dim),
    dT(Singleton<Real>::create(3/(2 * dT), hand))
{
    auto p = gridVecs.col(0);
    meta.xFile >> GpuIn<Real>(p, hand, false, true);
    auto f = lagrangeVecs.col(0);
    meta.xFile >> GpuIn<Real>(f, hand, false, true);

    meta.bFile >> GpuIn<Real>(maxSparseVals, hand, false, true);
}

template<typename Real, typename Int>
BaseData<Real, Int>::BaseData(
    SimpleArray<Int> maxSparseInds,
    SimpleArray<Int> maxSparseOffsets,
    const GridDim &dim,
    const Real3d &delta,
    Singleton<Real> dT
) :
    maxSparseInds(maxSparseInds),
    maxSparseOffsets(maxSparseOffsets),
    dim(dim),
    delta(delta),
    dT(dT)
{
}

template<typename Real, typename Int>
BaseData<Real, Int>::BaseData(const GridDim &dim, size_t fSize, size_t nnzMax, const Real3d &delta, Real *f, Real *p, double dT, Handle& hand) :
    BaseData(
        SimpleArray<Int>::create(nnzMax, hand),
        SimpleArray<Int>::create(fSize + 1, hand),
        dim,
        delta,
        Singleton<Real>::create(3/(2 * dT), hand)
    ) {
    this->lagrangeVec(LagrangeInd::f).set(f, hand);
    this->gridVec(GridInd::p).set(p, hand);
}

template<typename Real, typename Int>
void BaseData<Real, Int>::setSparse(
    std::shared_ptr<SparseMat<Real, Int>>& sparse,
    size_t nnz,
    Int *offsets,
    Int *inds,
    Real *vals,
    Handle& hand
) {//TODO: deal with B = null case.
    checkNNZB(nnz);
    sparse = sparse->createWithPointer(
            maxSparseVals.subArray(0, nnz),
            maxSparseOffsets,
            maxSparseInds.subArray(0, nnz)
        );
    sparse->set(offsets, inds, vals, hand);
}

template<typename Real, typename Int>
void BaseData<Real, Int>::printDenseB(Handle& hand) const {
    auto denseB = Mat<Real>::create(B->rows, B->cols);
    B->getDense(denseB, hand);
    std::cout << GpuOut<Real>(denseB, hand) << std::endl;
}

template<typename Real, typename Int>
SimpleArray<Real> BaseData<Real, Int>::lagrangeVec(LagrangeInd ind) const{
    return lagrangeVecs.col(static_cast<size_t>(ind));
}

template<typename Real, typename Int>
SimpleArray<Real> BaseData<Real, Int>::gridVec(GridInd ind) const{
    return gridVecs.col(static_cast<size_t>(ind));
}



template<typename Real>
std::shared_ptr<EigenDecompSolver<Real>> createEDS(
    const GridDim &dim,
    SimpleArray<Real> sizeOfP,
    Handle *hand,
    Real3d delta = Real3d(1, 1, 1))
{
    auto maxDimX2Or3 = Mat<Real>::create(dim.maxDim(), dim.numDims());
    auto rowsXrows = SquareMat<Real>::create(dim.rows);
    auto colsXcols = dim.cols != dim.rows ? SquareMat<Real>::create(dim.cols) : rowsXrows;

    if (dim.numDims() == 3) {
        //(SquareMat<T> &rowsXRows, SquareMat<T> &colsXCols, SquareMat<T> &depthsXDepths, Mat<T> &maxDimX3, SimpleArray<T>& sizeOfB, Handle* hand3, Real3d delta = Real3d(1, 1, 1));
        auto layersXLayers = dim.layers != dim.rows ? SquareMat<Real>::create(dim.layers) : rowsXrows;
        return std::make_shared<EigenDecompSolver3d<Real>>(rowsXrows, colsXcols, layersXLayers, maxDimX2Or3, sizeOfP, hand, delta);
    } else {
        return std::make_shared<EigenDecompSolver2d<Real>>(rowsXrows, colsXcols, maxDimX2Or3, sizeOfP, hand,
                                             Real2d(delta.x, delta.y));
    }
}

template<typename Real, typename Int>//TODO: chnage this so it can mult B or R.
void ImmersedEq<Real, Int>::multB(const SimpleArray<Real> &vec, SimpleArray<Real> &result, const Singleton<Real> &multProduct, const Singleton<Real> &preMultResult, bool transposeB) const {

    const size_t multBufferSizeNeeded = baseData.B->multWorkspaceSize(vec, result, multProduct, preMultResult, transposeB, hand5[0]);
    if (!sparseMultBuffer.get() || multBufferSizeNeeded > sparseMultBuffer->size())
        sparseMultBuffer = std::make_shared<SimpleArray<Real> >(SimpleArray<Real>::create(1.5 * multBufferSizeNeeded, hand5[0]));

    baseData.B->mult(vec, result, multProduct, preMultResult, transposeB, *sparseMultBuffer, hand5[0]);
}

template<typename Real, typename Int>
ImmersedEq<Real, Int>::ImmersedEq(BaseData<Real, Int> baseData, double tolerance, size_t maxBCGIterations):
    baseData(baseData),
    tolerance(tolerance),
    maxIterations(maxBCGIterations){
}

template<typename Real, typename Int>
ImmersedEq<Real, Int>::ImmersedEq(const GridDim &dim, size_t fSize, size_t nnzMaxSparse, Real *p, Real *f, const Real3d &delta, double dT, double tolerance, size_t maxBCGIterations) :
    baseData(dim, fSize, nnzMaxSparse, delta, f, p, dT, hand5[0]),
    tolerance(tolerance),
    maxIterations(maxBCGIterations)
    {
}

template<typename Real, typename Int> //(I+2L^-1BT*B) * x = b, or equivilently, x = (I+2L^-1BT*B)^-1 b
void ImmersedEq<Real, Int>::LHSTimes(const SimpleArray<Real> &x, SimpleArray<Real> &result, const Singleton<Real> &multLinearOperationOutput, const Singleton<Real> &preMultResult) const {

    if (preMultResult.data() == Singleton<Real>::ZERO.data()) result.fill(0, hand5[4]);
    else result.mult(preMultResult, &hand5[4]);
    lhsTimes.record(hand5[4]);

    auto Bx = baseData.lagrangeVec(LagrangeInd::LHS);
    auto BTBx = baseData.gridVec(GridInd::LHS1);
    auto invLBTBx = baseData.gridVec(GridInd::LHS2);

    multB(x, Bx, Singleton<Real>::ONE, Singleton<Real>::ZERO, false);// f <- B * x
    multB(Bx, BTBx, Singleton<Real>::TWO, Singleton<Real>::ZERO, true);// p <- B^T * (B * x)
    eds->solve(invLBTBx, BTBx, hand5[0]); // workspace2 <- L^-1 * B^T * (B * x)

    invLBTBx.add(x, &Singleton<Real>::ONE, &hand5[0]);
    auto& invLxBTBxPlusX = invLBTBx;

    lhsTimes.hold(hand5[0]);
    result.add(invLxBTBxPlusX, &multLinearOperationOutput, &hand5[0]); //result <- result + preMultResult * x * preMultX
}



template<typename Real, typename Int>
SquareMat<Real> ImmersedEq<Real, Int>::LHSMat() {
    auto id = SquareMat<Real>::create(baseData.p->size());
    id.setToIdentity(hand5[0]);

    auto result = SquareMat<Real>::create(baseData.p->size());
    for (size_t i = 0; i < baseData.p->size(); ++i) {
        auto col = result.col(i);
        LHSTimes(id.col(i), static_cast<SimpleArray<Real> &>(col), Singleton<Real>::ONE, Singleton<Real>::ZERO);
    }
    return result;
}

template<typename Real, typename Int>//TODO: rewrite this method so that it takes indices for p and F, and remove p and f as pointers all together.
SimpleArray<Real> &ImmersedEq<Real, Int>::RHS(const bool reset) {

    if (reset) {
        auto f = baseData.f;

        auto pSize = baseData.gridVec(GridInd::RHS);

        pSize.set(*baseData.p, hand5[0]);

        multB(*f, pSize, Singleton<Real>::TWO, Singleton<Real>::ONE, true);
        //p <- BT*f+p

        eds->solve(RHSSpace, pSize, hand5[0]);
    }
    return RHSSpace;
}

template<typename Real, typename Int>
SimpleArray<Real> &ImmersedEq<Real, Int>::solve(const bool multiStream) {
    //TODO: should the initial guess be random, or the RHS of the equation?

    baseData.result.set(RHSSpace, hand5[0]);
    // baseData.result.fillRandom(&hand5[0]);

    ImmersedEqSolver<Real, Int> solver(*this, allocatedRHSHeightX7, allocated9, events11, tolerance, maxIterations);

    if (multiStream) solver.solveUnconditionedMultiStream(baseData.result);
    else solver.solveUnpreconditioned(baseData.result);

    return baseData.result;
}


template<typename Real, typename Int>
void ImmersedEq<Real, Int>::solve(
    Real *result,
    const size_t nnzB,
    Int *offsetsB,
    Int *indsB,
    Real *valuesB,
    const bool multiStream
) {

    auto resultDevice = solve(nnzB, offsetsB, indsB, valuesB, multiStream);
    resultDevice.get(result, hand5[0]);
}


/**
 * @brief Computes the discrete divergence (\nabla \cdot u*) on a staggered MAC grid.
 *
 * This kernel calculates the divergence at the cell centers (Eulerian grid) using
 * the intermediate velocity components stored on the cell faces. This represents
 * the "volume error" or source term for the Pressure Poisson equation in the
 * SIMPLE-based Immersed Boundary Method.
 *
 * @tparam Real Floating point type (float or double).
 *
 * @param u           The x-velocity component grid (staggered).
 * @param v           The y-velocity component grid (staggered).
 * @param w           The z-velocity component grid (staggered).
 * @param dst         Output scalar grid (cell centers) where divergence is stored.
 * @param deltaCols   The grid spacing in the x-direction (dx).
 * @param deltaRows   The grid spacing in the y-direction (dy).
 * @param deltaLayers The grid spacing in the z-direction (dz).
 *
 * @note **Grid Dimension Requirements:**
 * To ensure every cell center in @p dst has a bounding pair of faces:
 * - @p u must have dimensions (dst.cols + 1, dst.rows, dst.layers).
 * - @p v must have dimensions (dst.cols, dst.rows + 1, dst.layers).
 * - @p w must have dimensions (dst.cols, dst.rows, dst.layers + 1).
 *
 * @details
 * The calculation follows the second-order central difference for staggered grids:
 * div = (u[i+1,j,k] - u[i,j,k])/dx + (v[i,j+1,k] - v[i,j,k])/dy + (w[i,j,k+1] - w[i,j,k])/dz
 */
template <typename Real>
__global__ void divergenceKernel3d(
    DeviceData3d<Real> u,
    DeviceData3d<Real> v,
    DeviceData3d<Real> w,
    DeviceData3d<Real> dst,
    const double deltaCols,
    const double deltaRows,
    const double deltaLayers,
    const Real* scalar
) {
    if (GridInd3d ind; ind < dst)
        dst[ind] = -(*scalar) * (
            (u(ind, 0, 1, 0) - u[ind])/deltaCols +
            (v(ind, 1, 0, 0) - v[ind])/deltaRows +
            (w(ind, 0, 0, 1) - w[ind])/deltaLayers);

}

/**
 * @brief Computes the discrete divergence (∇·u*) on a 2D staggered MAC grid.
 *
 * @tparam T Floating point type (float or double).
 * @param u The x-velocity component grid.
 * @param v The y-velocity component grid.
 * @param dst Output scalar grid (cell centers) for divergence results.
 * @param deltaCols Grid spacing in x (dx).
 * @param deltaRows Grid spacing in y (dy).
 *
 * @note **Requirement:** @p u and @p v must have 1 more element in their
 * respective staggered dimension than @p dst.
 */
template <typename Real>
__global__ void divergenceKernel2d(DeviceData2d<Real> u, DeviceData2d<Real> v, DeviceData2d<Real> dst, const double deltaCols, const double deltaRows, const Real* scalar) {

    if(GridInd2d ind;ind < dst)
        dst[ind] = *scalar * (
            (u(ind, 0, 1) - u[ind])/deltaCols +
            (v(ind, 1, 0) - v[ind])/deltaRows
        );
}


template<typename Real, typename Int>
void BaseData<Real, Int>::setRHSPPrime(Handle &hand) {

    auto u = velocities.subArray(0, dim.rows *(dim.cols + 1) * dim.layers);
    auto v = velocities.subArray(u.size(),(dim.rows + 1) * dim.cols * dim.layers);
    auto w = velocities.subArray(u.size() + v.size(), velocities.size() - u.size() - v.size());

    auto RHSPPrime = gridVec(GridInd::RHSPPrime);
    const KernelPrep kp = RHSPPrime.kernelPrep();

    if (dim.layers > 1) divergenceKernel3d<Real><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
            u.tensor(dim.rows, dim.layers).toKernel3d(),
            v.tensor(dim.rows, dim.layers).toKernel3d(),
            w.tensor(dim.rows, dim.layers).toKernel3d(),
            RHSPPrime.tensor(dim.rows, dim.layers).toKernel3d(),
            delta.x, delta.y, delta.z,
            dT.data()
        );
    else divergenceKernel2d<Real><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
            u.matrix(dim.rows).toKernel2d(),
            v.matrix(dim.rows).toKernel2d(),
            RHSPPrime.matrix(dim.rows).toKernel2d(),
            delta.x, delta.y,
            dT.data()
        );

}

template<typename Real, typename Int>
void ImmersedEq<Real, Int>::setRHSFPrime(Handle &hand) {

    auto RHSF = baseData.lagrangeVec(LagrangeInd::RHSFPrime);

    size_t minBufferSize = baseData.R->multWorkspaceSize(baseData.velocities, RHSF, baseData.dT, Singleton<Real>::ZERO, true, hand);//TODO:this code seems redundant with the multB code.
    if (!sparseMultBuffer || sparseMultBuffer->size() > minBufferSize) sparseMultBuffer = std::make_shared<SimpleArray<Real>>(SimpleArray<Real>::create(1.5 * minBufferSize, hand, true));

    baseData.R->mult(baseData.velocities, RHSF, baseData.dT, Singleton<Real>::ZERO, true, *sparseMultBuffer, hand);

    RHSF.subtract(baseData.lagrangeVec(LagrangeInd::UGamma), &baseData.dT, sparseMultBuffer->get(0), &hand);
}

template<typename Real, typename Int>
void ImmersedEq<Real, Int>::solve(
    Real* resultP,
    Real* resultF,
    size_t nnzB,
    Int *rowOffsetsB,
    Int *colIndsB,
    Real *valuesB,
    size_t nnzR,
    Int *colOffsetsR,
    Int *rowIndsR,
    Real *valuesR,
    Real *UGamma,
    Real* uStar,
    bool multiStream) {

    baseData.setSparse(baseData.R, nnzR, colOffsetsR, rowIndsR, valuesR, hand5[0]);
    baseData.velocities.set(uStar, hand5[0]);
    baseData.setRHSPPrime(hand5[0]);

    events11[0].record(hand5[0]);
    events11[0].hold(hand5[1]);
    baseData.lagrangeVec(LagrangeInd::UGamma).set(UGamma, hand5[1]);
    setRHSFPrime(hand5[1]);
    events11[1].record(hand5[1]);
    events11[1].hold(hand5[0]);

    baseData.p = std::make_shared<SimpleArray<Real>>(baseData.gridVec(GridInd::RHSPPrime));
    baseData.f = std::make_shared<SimpleArray<Real>>(baseData.lagrangeVec(LagrangeInd::RHSFPrime));

    solve(resultP, nnzB, rowOffsetsB, colIndsB, valuesB, multiStream);

    baseData.p = std::make_shared<SimpleArray<Real>>(baseData.gridVec(GridInd::p));
    baseData.f = std::make_shared<SimpleArray<Real>>(baseData.lagrangeVec(LagrangeInd::f));//2 (B p' - RHSf')

    auto fResultDevice = baseData.lagrangeVec(LagrangeInd::fPrime);
    multB(baseData.gridVec(GridInd::pPrime), fResultDevice, Singleton<Real>::TWO, Singleton<Real>::ZERO, false);

    fResultDevice.add(baseData.lagrangeVec(LagrangeInd::RHSFPrime), &Singleton<Real>::MINUS_ONE, &hand5[0]);
    fResultDevice.get(resultF, hand5[0]);
}

template<typename Real, typename Int>
SimpleArray<Real> ImmersedEq<Real, Int>::solve(
    size_t nnzB,
    Int *offsetsB, //
    Int *indsB, //
    Real *valuesB,
    const bool multithreadBCG
) {

    baseData.setSparse(baseData.B, nnzB, offsetsB, indsB, valuesB, hand5[0]);

    RHS(true);

    return solve(multithreadBCG);
}

template<typename Real, typename Int>
ImmersedEqSolver<Real, Int>::ImmersedEqSolver(
    ImmersedEq<Real, Int>& imEq,
    Mat<Real> &allocatedRHSHeightX7,
    Vec<Real> &allocated9,
    Event* events11,
    Real tolerance,
    size_t maxIterations
)
    : BiCGSTAB<Real>(imEq.RHS(false), imEq.hand5.get(), events11, &allocatedRHSHeightX7, &allocated9, tolerance, maxIterations),
      imEq(imEq) {
}

template<typename Real, typename Int>
void ImmersedEqSolver<Real, Int>::mult(Vec<Real> &vec, Vec<Real> &product, Singleton<Real> multProduct,
                                       Singleton<Real> preMultResult) const {
    SimpleArray<Real> vecSA(vec), productSA(product);

    return imEq.LHSTimes(vecSA, productSA, multProduct, preMultResult);
}

// If your SparseCSC usually uses uint32_t and uint64_t:
template class BaseData<float, int32_t>;
template class BaseData<double, int32_t>;
template class BaseData<float, int64_t>;
template class BaseData<double, int64_t>;

template class ImmersedEq<float, int32_t>;
template class ImmersedEq<double, int32_t>;
template class ImmersedEq<float, int64_t>;
template class ImmersedEq<double, int64_t>;

template class ImmersedEqSolver<float, int32_t>;
template class ImmersedEqSolver<double, int32_t>;
template class ImmersedEqSolver<float, int64_t>;
template class ImmersedEqSolver<double, int64_t>;