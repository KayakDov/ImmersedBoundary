#include "ImerssedEquation.h"

#include "Streamable.h"
#include "solvers/EigenDecompSolver.h"
#include "solvers/Event.h"
#include <span>

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
BaseData<Real, Int>::BaseData(const FileMeta &meta, const GridDim &dim, Handle& hand) :

    pSizeX4(Mat<Real>::create(meta.pSize, 4)),
    fSizeX2(Mat<Real>::create(meta.fSize, 2)),
    maxB(SparseCSC<Real, Int>::create(meta.nnz, meta.bRows, meta.bCols, hand)),
    delta(1.0 / dim.cols, 1.0 / dim.rows, 1.0 / dim.layers),
    dim(dim)
{
    auto p = pSizeX4.col(0);
    meta.xFile >> GpuIn<Real>(p, hand, false, true);
    auto f = fSizeX2.col(0);
    meta.xFile >> GpuIn<Real>(f, hand, false, true);

    meta.bFile >> GpuIn<Real>(maxB.values, hand, false, true);
}

template<typename Real, typename Int>
BaseData<Real, Int>::BaseData(
    SparseCSC<Real, Int> maxB,
    Mat<Real> fSizeX2,
    Mat<Real> pSizeX4,
    const GridDim &dim,
    const Real3d &delta
) : pSizeX4(pSizeX4),
    fSizeX2(fSizeX2),
    maxB(maxB),
    dim(dim),
    delta(delta) {
}

template<typename Real, typename Int>
BaseData<Real, Int>::BaseData(const GridDim &dim, size_t fSize, size_t nnzMaxB, const Real3d &delta, Real *f, Real *p, Handle& hand) :
    BaseData(
        SparseCSC<Real, Int>::create(nnzMaxB, fSize, dim.size(), hand),
        Mat<Real>::create(fSize, 2),
        Mat<Real>::create(dim.size(), 4),
        dim,
        delta
    ) {
    this->fSizeX2.col(0).set(f, hand);
    this->pSizeX4.col(0).set(p, hand);
}

template<typename Real, typename Int>
void BaseData<Real, Int>::setB(size_t nnzB, Int *colsB, Int *rowsB, Real *valsB, Handle& hand) {
    if (nnzB > maxB.nnz()) {
        throw std::invalid_argument(
            "BaseData::setB - NNZ Overflow: Requested nnzB (" + std::to_string(nnzB) +
            ") exceeds maxB capacity (" + std::to_string(maxB.nnz()) + ")."
        );
    }
    B = std::make_shared<SparseCSC<Real, Int> >(
        SparseCSC<Real, Int>::create(
            maxB.rows,
            maxB.values.subAray(0, nnzB),
            maxB.rowPointers.subAray(0, nnzB),
            maxB.columnOffsets,
            hand
        )
    );
    B->columnOffsets.set(colsB, hand);
    B->rowPointers.set(rowsB, hand);
    B->values.set(valsB, hand);
}

template<typename Real, typename Int>
SimpleArray<Real> BaseData<Real, Int>::allocatedFSize() const {
    return fSizeX2.col(1);
}

template<typename Real, typename Int>
SimpleArray<Real> BaseData<Real, Int>::allocatedPSize(bool ind) const {
    return pSizeX4.col(ind + 1);
}

template<typename Real, typename Int>
void BaseData<Real, Int>::printDenseB(Handle& hand) const {
    auto denseB = Mat<Real>::create(B->rows, B->cols);
    B->getDense(denseB, hand);
    std::cout << GpuOut<Real>(denseB, hand) << std::endl;
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

template<typename Real, typename Int>
void ImmersedEq<Real, Int>::multB(const SimpleArray<Real> &vec, SimpleArray<Real> &result, const Singleton<Real> &multProduct, const Singleton<Real> &preMultResult, bool transposeThis) const {

    size_t multBufferSizeNeeded = baseData.B->multWorkspaceSize(vec, result, multProduct, preMultResult, transposeThis, hand5[0]);
    if (!sparseMultBuffer || multBufferSizeNeeded > sparseMultBuffer->size()) sparseMultBuffer = std::make_shared<SimpleArray<Real> >(SimpleArray<Real>::create(1.5 * multBufferSizeNeeded, hand5[0]));

    baseData.B->mult(vec, result, multProduct, preMultResult, transposeThis, *sparseMultBuffer, hand5[0]);
}

template<typename Real, typename Int>
ImmersedEq<Real, Int>::ImmersedEq(BaseData<Real, Int> baseData, double tolerance, const size_t maxBCGIterations) :
    baseData(baseData),
    tolerance(tolerance),
    maxIterations(maxBCGIterations){
}

template<typename Real, typename Int>
ImmersedEq<Real, Int>::ImmersedEq(const GridDim &dim, size_t fSize, size_t nnzMaxB, Real *p, Real *f, const Real3d &delta, double tolerance, size_t maxBCGIterations) :
    baseData(dim, fSize, nnzMaxB, delta, f, p, hand5[0]),
    tolerance(tolerance),
    maxIterations(maxBCGIterations)
    {
}

template<typename Real, typename Int> //(I+2L^-1BT*B) * x = b, or equivilently, x = (I+2L^-1BT*B)^-1 b
void ImmersedEq<Real, Int>::LHSTimes(const SimpleArray<Real> &x, SimpleArray<Real> &result,
                                     const Singleton<Real> &multLinearOperationOutput, const Singleton<Real> &preMultResult) const {

    // std::cout << "x1 = " << GpuOut<Real>(x, hand) << std::endl;
    // std::cout << "Debugging LHS mult" << std::endl;

    auto fSize = baseData.allocatedFSize();
    auto pSize0 = baseData.allocatedPSize(0);
    auto invLBTBx = baseData.allocatedPSize(1);


    multB(x, fSize, Singleton<Real>::ONE, Singleton<Real>::ZERO, false);// f <- B * x

    multB(fSize, pSize0, Singleton<Real>::TWO, Singleton<Real>::ZERO, true);// p <- B^T * (B * x)

    eds->solve(invLBTBx, pSize0, hand5[0]); // workspace2 <- L^-1 * B^T * (B * x)

    invLBTBx.add(x, &Singleton<Real>::ONE, &hand5[0]);

    auto& invLxBTBxPlusX = invLBTBx;

    result.mult(preMultResult, &hand5[4]); //x <- x * preMultX
    Event preMult;
    preMult.record(hand5[4]);
    preMult.wait(hand5[0]);

    result.add(invLxBTBxPlusX, &multLinearOperationOutput, &hand5[0]); //result <- result + preMultResult * x * preMultX
}

template<typename Real, typename Int>
SquareMat<Real> ImmersedEq<Real, Int>::LHSMat() {
    auto id = SquareMat<Real>::create(baseData.p.size());
    id.setToIdentity(hand5[0]);

    auto result = SquareMat<Real>::create(baseData.p.size());
    for (size_t i = 0; i < baseData.p.size(); ++i) {
        auto col = result.col(i);
        LHSTimes(id.col(i), static_cast<SimpleArray<Real> &>(col), Singleton<Real>::ONE, Singleton<Real>::ZERO);
    }
    return result;
}

template<typename Real, typename Int>
SimpleArray<Real> &ImmersedEq<Real, Int>::RHS(const bool reset) {

    if (reset) {
        auto f = baseData.f;

        auto pSize = baseData.allocatedPSize(0);

        pSize.set(baseData.p, hand5[0]);

        multB(f, pSize, Singleton<Real>::TWO, Singleton<Real>::ONE, true);
        //p <- BT*f+p

        eds->solve(RHSSpace, pSize, hand5[0]);
    }
    return RHSSpace;
}


template<typename Real, typename Int>
void ImmersedEq<Real, Int>::solve(
    Real *result,
    const size_t nnzB,
    Int *rowPointersB,
    Int *colPointersB,
    Real *valuesB,
    const bool multiStream
) {

    auto resultDevice = solve(nnzB, rowPointersB, colPointersB, valuesB, multiStream);
    resultDevice.get(result, hand5[0]);
}

template<typename Real, typename Int>
SimpleArray<Real> ImmersedEq<Real, Int>::solve(
    size_t nnzB,
    Int *rowPointersB,
    Int *colPointersB,
    Real *valuesB,
    const bool multithreadBCG
) {
    baseData.setB(nnzB, colPointersB, rowPointersB, valuesB, hand5[0]);
    RHS(true);

    //TODO: should the initial guess be random, or the RHS of the equation?

    baseData.result.set(RHSSpace, hand5[0]);
    // baseData.result.fillRandom(hand4);

    ImmersedEqSolver<Real, Int> solver(*this, allocatedRHSHeightX7, allocated9, tolerance, maxIterations);
    if (multithreadBCG) solver.solveUnconditionedMultiStream(baseData.result);
    else solver.solveUnpreconditioned(baseData.result);

    return baseData.result;
}


template<typename Real, typename Int>
ImmersedEqSolver<Real, Int>::ImmersedEqSolver(
    ImmersedEq<Real, Int>& imEq,

    Mat<Real> &allocatedRHSHeightX7,
    Vec<Real> &allocated9,
    Real tolerance,
    size_t maxIterations
)
    : BiCGSTAB<Real>(imEq.RHS(false), imEq.hand5.get(), &allocatedRHSHeightX7, &allocated9, tolerance, maxIterations),
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