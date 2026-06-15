#include "ImerssedEquation.h"

#include "solvers/EigenDecomp/EigenDecompSolver.h"
#include "solvers/Event.h"

#include "solvers/EigenDecomp/EigenDecomp2d.h"
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"
#include <string>

//TODO: Split this into a class that deals with memory allocation, and a class that does everything else.

template<typename Real, typename Int>
void ImmersedEq<Real, Int>::checkNNZ(size_t nnz) const {
    if (nnz > maxSparseVals.size()) {
        throw std::invalid_argument(
            "ImmersedEq::setSparse - NNZ Overflow: Requested nnzB (" + std::to_string(nnz) +
            ") exceeds maxB capacity (" + std::to_string(maxSparseVals.size()) + ")."
        );
    }
}


template<typename Real, typename Int>
void ImmersedEq<Real, Int>::setSparse(
    std::unique_ptr<SparseMat<Real, Int>>& sparse,
    size_t nnz,
    Int *offsets,
    Int *inds,
    Real *vals,
    Handle& hand
) {
    checkNNZ(nnz);
    sparse = sparse->createWithPointer(
            maxSparseVals.subArray(0, nnz),
            maxSparseOffsets,
            maxSparseInds.subArray(0, nnz)
        );
    sparse->set(offsets, inds, vals, hand);
}

template<typename Real, typename Int>
SimpleArray<Real> ImmersedEq<Real, Int>::lagrangeVec(LagrangeInd ind) const{
    return lagrangeVecs.col(static_cast<size_t>(ind));
}

template<typename Real, typename Int>
SimpleArray<Real> ImmersedEq<Real, Int>::gridVec(GridInd ind) const{
    return gridVecs.col(static_cast<size_t>(ind));
}

template<typename Real, typename  BoundaryConfigT>
std::shared_ptr<EigenDecompSolver<Real>> createEDS(const BoundaryConfigT &bounary, Handle *hand, Event* event) {

    if (bounary.dim().numDims() == 3) return std::make_shared<EigenDecomp3d<Real>>(bounary, hand, event);
    return std::make_shared<EigenDecomp2d<Real>>(bounary, hand, event[0]);

}

template<typename Real, typename Int>
void ImmersedEq<Real, Int>::multSparse(const std::unique_ptr<SparseMat<Real, Int>>& mat, const SimpleArray<Real> &vec, SimpleArray<Real> &result, const Singleton<Real> &multProduct, const Singleton<Real> &preMultResult, bool transposeB) const {

    const size_t multBufferSizeNeeded = mat->multWorkspaceSize(vec, result, multProduct, preMultResult, transposeB, hand5[0]);
    if (!sparseMultBuffer || multBufferSizeNeeded > sparseMultBuffer->size())
        sparseMultBuffer = std::make_unique<SimpleArray<Real> >(SimpleArray<Real>::create(1.5 * multBufferSizeNeeded, hand5[0]));

    mat->mult(vec, result, multProduct, preMultResult, transposeB, *sparseMultBuffer, hand5[0]);
}

template<typename Real, typename Int>
template<typename BoundaryConfigT>
ImmersedEq<Real, Int>::ImmersedEq(
    BoundaryConfigT boundary,
    SimpleArray<Int> maxSparseInds,
    SimpleArray<Int> maxSparseOffsets,
    Singleton<Real> dT, Real tolerance,
    size_t maxBCGIterations
):
    dim(boundary.dim()),
    delta(boundary.delta()),
    eds(createEDS<Real>(boundary, hand5, events12)),
    maxSparseInds(maxSparseInds),
    maxSparseOffsets(maxSparseOffsets),
    dT(dT),
    solver(*this, gridVecs, SimpleArray<Real>::create(9, hand5[0]), events12, tolerance, maxBCGIterations)
{

    poisson::boundaryCorrection(boundary, gridVec(GridInd::boundRHSAdj), hand5[0]);
}

template<typename Real, typename Int>
template<typename BoundaryConfigT>
ImmersedEq<Real, Int>::ImmersedEq(
    const BoundaryConfigT &boundary,
    size_t fSize,
    size_t nnzMax,
    Real *p,
    Real *f,
    double dT,
    Real tolerance,
    size_t maxBCGIterations
) :
    delta(boundary.delta()),
    dim(boundary.dim()),
    eds(createEDS<Real>(boundary, hand5, events12)),
    maxSparseInds(SimpleArray<Int>::create(nnzMax + fSize + 1, hand5[0]).subArray(0, nnzMax)),
    maxSparseOffsets(maxSparseInds.subArray(nnzMax, fSize + 1)),
    dT(Singleton<Real>::create(3/(2 * dT), hand5[0])),
    solver(*this, gridVecs, SimpleArray<Real>::create(9, hand5[0]), events12, tolerance, maxBCGIterations)
{
    this->lagrangeVec(LagrangeInd::f).set(f, hand5[0]);
    this->gridVec(GridInd::p).set(p, hand5[0]);

    poisson::boundaryCorrection(boundary, gridVec(GridInd::boundRHSAdj), hand5[0]);
}

template<typename Real, typename Int> //(I+2L^-1BT*B) * x = b, or equivilently, x = (I+2L^-1BT*B)^-1 b
void ImmersedEq<Real, Int>::LHSTimes(const SimpleArray<Real> &x, SimpleArray<Real> &result, const Singleton<Real> &multLinearOperationOutput, const Singleton<Real> &preMultResult) {

    if (preMultResult.data() == GPUScalar<Real>::get(0).data()) result.fill(0, hand5[4]);
    else {
        lhsTimes.record(hand5[0]);
        lhsTimes.hold(hand5[4]);
        result.mult(preMultResult, hand5 + 4);
    }
    lhsTimes.record(hand5[4]);

    auto Bx = lagrangeVec(LagrangeInd::LHS_Bx);
    auto BTBx = gridVec(GridInd::LHS_BTBx);
    auto invLBTBx = gridVec(GridInd::LHS_invLBTBx);

    multSparse(B, x, Bx, GPUScalar<Real>::get(1), GPUScalar<Real>::get(0), false);// f <- B * x
    multSparse(B, Bx, BTBx, GPUScalar<Real>::get(2), GPUScalar<Real>::get(0), true);// p <- B^T * (B * x)
    eds->solve(invLBTBx, BTBx, hand5[0]); // workspace2 <- L^-1 * B^T * (B * x)

    invLBTBx.add(x, &GPUScalar<Real>::get(1), hand5);
    auto& invLxBTBxPlusX = invLBTBx;

    lhsTimes.hold(hand5[0]);
    result.add(invLxBTBxPlusX, &multLinearOperationOutput, hand5); //result <- result + preMultResult * x * preMultX
}

template<typename Real, typename Int>
SquareMat<Real> ImmersedEq<Real, Int>::LHSMat() {
    auto id = SquareMat<Real>::create(dim.size());
    id.setToIdentity(hand5[0]);

    auto result = SquareMat<Real>::create(dim.size());
    for (size_t i = 0; i < dim.size(); ++i) {
        auto col = result.col(i);
        LHSTimes(id.col(i), col, GPUScalar<Real>::get(1), GPUScalar<Real>::get(0));
    }
    return result;
}

template<typename Real, typename Int>
SimpleArray<Real> ImmersedEq<Real, Int>::getRHS(SimpleArray<Real>& p, SimpleArray<Real>& f, SimpleArray<Real>& rhsAdjustment, SparseCSR<Real, Int>& B) {
    gridVec(GridInd::p).set(p, hand5[0]);
    lagrangeVec(LagrangeInd::f).set(f, hand5[0]);
    gridVec(GridInd::boundRHSAdj).set(rhsAdjustment, hand5[0]);
    this->B = std::make_unique<SparseCSR<Real, Int>>(B);

    setRHS(false);
    return gridVec(GridInd::RHS);
}

template<typename Real, typename Int>
void ImmersedEq<Real, Int>::setRHS(bool prime) {

    auto p = gridVec(prime ? GridInd::RHSPPrime : GridInd::p);
    auto f = lagrangeVec(prime? LagrangeInd::RHSFPrime : LagrangeInd::f);
    auto boundaryRhsAdj = gridVec(GridInd::boundRHSAdj);

    multSparse(B, f, p, GPUScalar<Real>::get(2), GPUScalar<Real>::get(1), true);
    //p <- BT*f+p

    p.add(boundaryRhsAdj, &GPUScalar<Real>::get(1), hand5);

    auto RHS = gridVec(GridInd::RHS);
    eds->solve(RHS, p, hand5[0]);
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
 * @param dst         Output scalar grid (cell centers) where divergence is stored
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
    XYZ<DeviceData3d<Real>> u,
    DeviceData3d<Real> dst,
    const XYZ<Delta1d<Real>> delta,
    const Real* scalar
) {
    if (GridInd3d ind; ind < dst)
        dst[ind] = -(*scalar) * (
            (u.x(ind, 0, 1, 0) - u.x[ind])/delta.x[ind.col + 1] +
            (u.y(ind, 1, 0, 0) - u.y[ind])/delta.y[ind.row + 1] +
            (u.z(ind, 0, 0, 1) - u.z[ind])/delta.z[ind.layer + 1]
        );
}

/**
 * @brief Computes the discrete divergence (∇·u*) on a 2D staggered MAC grid.
 *
 * @tparam T Floating point type (float or double).
 * @param u The x-velocity component grid.
 * @param v The y-velocity component grid.
 * @param dst Output scalar grid (cell centers) for divergence results.
 *
 * @note **Requirement:** @p u and @p v must have 1 more element in their
 * respective staggered dimension than @p dst.
 */
template <typename Real>
__global__ void divergenceKernel2d(DeviceData2d<Real> u, DeviceData2d<Real> v, DeviceData2d<Real> dst, const XYZ<Delta1d<Real>> delta, const Real* scalar) {

    if(GridInd2d ind;ind < dst)
        dst[ind] = *scalar * (
            (u(ind, 0, 1) - u[ind])/delta.x[ind.col + 1] +
            (v(ind, 1, 0) - v[ind])/delta.y[ind.row + 1]
        );
}


template<typename Real, typename Int>
void ImmersedEq<Real, Int>::setRHSPPrime(Handle &hand) {

    auto u = velocities.subArray(0, dim.rows *(dim.cols + 1) * dim.layers);
    auto v = velocities.subArray(u.size(),(dim.rows + 1) * dim.cols * dim.layers);
    auto w = velocities.subArray(u.size() + v.size(), velocities.size() - u.size() - v.size());

    auto RHSPPrime = gridVec(GridInd::RHSPPrime);
    const KernelPrep kp = RHSPPrime.kernelPrep();

    if (dim.layers > 1) divergenceKernel3d<Real><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
            {
                u.tensor(dim.rows, dim.layers).toKernel3d(),
                v.tensor(dim.rows, dim.layers).toKernel3d(),
                w.tensor(dim.rows, dim.layers).toKernel3d()
            },
            RHSPPrime.tensor(dim.rows, dim.layers).toKernel3d(),
            delta,
            dT.data()
        );
    else divergenceKernel2d<Real><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
            u.matrix(dim.rows).toKernel2d(),
            v.matrix(dim.rows).toKernel2d(),
            RHSPPrime.matrix(dim.rows).toKernel2d(),
            delta,
            dT.data()
        );

}

template<typename Real, typename Int>
void ImmersedEq<Real, Int>::setRHSFPrime(Handle &hand) {

    auto RHSF = lagrangeVec(LagrangeInd::RHSFPrime);

    multSparse(R, velocities, RHSF, dT, GPUScalar<Real>::get(0), true);

    RHSF.subtract(lagrangeVec(LagrangeInd::UGamma), &dT, sparseMultBuffer->get(0), &hand);
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
    Real* uStar) {

    setSparse(R, nnzR, colOffsetsR, rowIndsR, valuesR, hand5[0]);
    velocities.set(uStar, hand5[0]);
    setRHSPPrime(hand5[0]);

    events12[0].record(hand5[0]);
    events12[0].hold(hand5[1]);
    lagrangeVec(LagrangeInd::UGamma).set(UGamma, hand5[1]);
    setRHSFPrime(hand5[1]);
    events12[1].record(hand5[1]);
    events12[1].hold(hand5[0]);

    auto resultDevice = solve(nnzB, rowOffsetsB, colIndsB, valuesB, true);
    resultDevice.get(resultP, hand5[0]);

    auto fResultDevice = lagrangeVec(LagrangeInd::fPrime);
    multSparse(B, gridVec(GridInd::pPrime), fResultDevice, GPUScalar<Real>::get(2), GPUScalar<Real>::get(0), false);

    fResultDevice.add(lagrangeVec(LagrangeInd::RHSFPrime), &GPUScalar<Real>::get(-2), &hand5[0]);
    fResultDevice.get(resultF, hand5[0]);
}

template<typename Real, typename Int>
SimpleArray<Real> ImmersedEq<Real, Int>::solve(
    size_t nnzB,
    Int *offsetsB,
    Int *indsB,
    Real *valuesB,
    bool prime
) {

    setSparse(B, nnzB, offsetsB, indsB, valuesB, hand5[0]);

    setRHS(prime);

    return solve();
}


template<typename Real, typename Int>
SimpleArray<Real> ImmersedEq<Real, Int>::solve() {
    //TODO: should the initial guess be random, or the RHS of the equation?

    auto result = gridVec(GridInd::Result);

    result.set(gridVec(GridInd::RHS), hand5[0]);
    // baseData.result.fillRandom(&hand5[0]);


    solver.solveUnpreconditioned(result);

    return result;
}


template<typename Real, typename Int>
void ImmersedEq<Real, Int>::solve(
    Real *result,
    const size_t nnzB,
    Int *offsetsB,
    Int *indsB,
    Real *valuesB
) {

    auto resultDevice = solve(nnzB, offsetsB, indsB, valuesB, false);
    resultDevice.get(result, hand5[0]);
}

template<typename Real, typename Int>
void ImmersedEqSolver<Real, Int>::mult(Vec<Real> &vec, Vec<Real> &product, Singleton<Real> multProduct,
                                       Singleton<Real> preMultResult) const {
    SimpleArray<Real> vecSA(vec), productSA(product);

    imEq.LHSTimes(vecSA, productSA, multProduct, preMultResult);
}

template<typename Real, typename Int>
ImmersedEqSolver<Real, Int>::ImmersedEqSolver(
    ImmersedEq<Real, Int>& imEq,
    Mat<Real> &gridVecs,
    Vec<Real> allocated9,
    Event* events11,
    Real tolerance,
    size_t maxIterations
):
    BiCGSTAB<Real>(
        imEq.gridVec(GridInd::RHS),
        imEq.hand5,
        events11,
        gridVecs.subMat(0, static_cast<size_t>(GridInd::Count), gridVecs._rows, 7),
        allocated9,
        tolerance,
        maxIterations
    ),
    imEq(imEq) {
}


#define INSTANTIATE_IMMERSED_EQ_BOUNDARY(Real, Int, SegX, SegY, SegZ) \
template ImmersedEq<Real, Int>::ImmersedEq( \
const BoundaryConfig<Real, SegX, SegY, SegZ>&, \
size_t, size_t, Real*, Real*, double, Real, size_t);

#define INSTANTIATE_FOR_SEG_COMBO(Real, Int) \
INSTANTIATE_IMMERSED_EQ_BOUNDARY(Real, Int, UniformSegment<Real>,  UniformSegment<Real>,  UniformSegment<Real>)  \
INSTANTIATE_IMMERSED_EQ_BOUNDARY(Real, Int, UniformSegment<Real>,  UniformSegment<Real>,  VariableSegment<Real>) \
INSTANTIATE_IMMERSED_EQ_BOUNDARY(Real, Int, UniformSegment<Real>,  VariableSegment<Real>, UniformSegment<Real>)  \
INSTANTIATE_IMMERSED_EQ_BOUNDARY(Real, Int, UniformSegment<Real>,  VariableSegment<Real>, VariableSegment<Real>) \
INSTANTIATE_IMMERSED_EQ_BOUNDARY(Real, Int, VariableSegment<Real>, UniformSegment<Real>,  UniformSegment<Real>)  \
INSTANTIATE_IMMERSED_EQ_BOUNDARY(Real, Int, VariableSegment<Real>, UniformSegment<Real>,  VariableSegment<Real>) \
INSTANTIATE_IMMERSED_EQ_BOUNDARY(Real, Int, VariableSegment<Real>, VariableSegment<Real>, UniformSegment<Real>)  \
INSTANTIATE_IMMERSED_EQ_BOUNDARY(Real, Int, VariableSegment<Real>, VariableSegment<Real>, VariableSegment<Real>)

INSTANTIATE_FOR_SEG_COMBO(float,  int32_t)
INSTANTIATE_FOR_SEG_COMBO(double, int32_t)
INSTANTIATE_FOR_SEG_COMBO(float,  int64_t)
INSTANTIATE_FOR_SEG_COMBO(double, int64_t)

template class ImmersedEqSolver<float, int32_t>;
template class ImmersedEqSolver<double, int32_t>;
template class ImmersedEqSolver<float, int64_t>;
template class ImmersedEqSolver<double, int64_t>;

template class ImmersedEq<float, int32_t>;
template class ImmersedEq<double, int32_t>;
template class ImmersedEq<float, int64_t>;
template class ImmersedEq<double, int64_t>;
