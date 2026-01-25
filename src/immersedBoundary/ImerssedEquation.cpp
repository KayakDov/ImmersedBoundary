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
BaseData<Real, Int>::BaseData(const FileMeta &meta, const GridDim &dim) :
    hand4(new Handle[4]),
    pSizeX4(Mat<Real>::create(meta.pSize, 4)),
    fSizeX2(Mat<Real>::create(meta.fSize, 2)),
    p(pSizeX4.col(0, true)),
    f(fSizeX2.col(0, true)),
    result(pSizeX4.col(3, true)),
    maxB(SparseCSC<Real, Int>::create(
    meta.nnz, meta.bRows, meta.bCols,
    hand4[0])),
    delta(1.0 / dim.cols, 1.0 / dim.rows, 1.0 / dim.layers),
    dim(dim)
{
    auto p = pSizeX4.col(0);
    meta.xFile >> GpuIn<Real>(p, hand4[0], false, true);
    auto f = fSizeX2.col(0);
    meta.xFile >> GpuIn<Real>(f, hand4[0], false, true);

    meta.bFile >> GpuIn<Real>(maxB.values, hand4[0], false, true);
}

template<typename Real, typename Int>
BaseData<Real, Int>::BaseData(
    SparseCSC<Real, Int> maxB,
    Mat<Real> fSizeX2,
    Mat<Real> pSizeX4,
    const GridDim &dim,
    const Real3d &delta
) : hand4(new Handle[4]),
    pSizeX4(pSizeX4),
    fSizeX2(fSizeX2),
    f(fSizeX2.col(0, true)),
    p(pSizeX4.col(0, true)),
    result(pSizeX4.col(3, true)),
    maxB(maxB),
    dim(dim),
    delta(delta) {
}

template<typename Real, typename Int>
BaseData<Real, Int>::BaseData(const GridDim &dim, size_t fSize, size_t nnzMaxB, const Real3d &delta, Real *f, Real *p) :
    BaseData(
        SparseCSC<Real, Int>::create(nnzMaxB, fSize, dim.volume(), hand4[0]),
        Mat<Real>::create(fSize, 2),
        Mat<Real>::create(dim.volume(), 4),
        dim,
        delta
    ) {
    this->fSizeX2.col(0).set(f, hand4[0]);
    this->pSizeX4.col(0).set(p, hand4[0]);
}

template<typename Real, typename Int>
void BaseData<Real, Int>::setB(size_t nnzB, Int *colsB, Int *rowsB, Real *valsB) {
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
            hand4[0]
        )
    );
    B->columnOffsets.set(colsB, hand4[0]);
    B->rowPointers.set(rowsB, hand4[0]);
    B->values.set(valsB, hand4[0]);
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
void BaseData<Real, Int>::printDenseB() const {
    auto denseB = Mat<Real>::create(B->rows, B->cols);
    B->getDense(denseB, hand4[0]);
    std::cout << GpuOut<Real>(denseB, hand4[0]) << std::endl;
}

template<typename Real>
std::shared_ptr<EigenDecompSolver<Real>> createEDS(
    const GridDim &dim,
    SimpleArray<Real> sizeOfP,
    Handle *hand,
    Real3d delta = Real3d(1, 1, 1)) {
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

    size_t multBufferSizeNeeded = baseData.B->multWorkspaceSize(vec, result, multProduct, preMultResult, transposeThis, baseData.hand4[0]);
    if (!sparseMultBuffer || multBufferSizeNeeded > sparseMultBuffer->size()) sparseMultBuffer = std::make_shared<SimpleArray<Real> >(SimpleArray<Real>::create(1.5 * multBufferSizeNeeded, baseData.hand4[0]));

    baseData.B->mult(vec, result, multProduct, preMultResult, transposeThis, *sparseMultBuffer, baseData.hand4[0]);
}

template<typename Real, typename Int>
ImmersedEq<Real, Int>::ImmersedEq(BaseData<Real, Int> baseData, double tolerance, const size_t maxBCGIterations) :
    baseData(baseData),
    RHSSpace(SimpleArray<Real>::create(baseData.p.size(), baseData.hand4[0])),
    sparseMultBuffer(nullptr),
    allocatedRHSHeightX7(Mat<Real>::create(baseData.p.size(), 7)),
    allocated9(Vec<Real>::create(9, baseData.hand4[0])),
    tolerance(tolerance),
    maxIterations(maxBCGIterations),
    eds(createEDS(baseData.dim, baseData.allocatedPSize(0), &baseData.hand4[0], baseData.delta)) {
}

template<typename Real, typename Int>
ImmersedEq<Real, Int>::ImmersedEq(const GridDim &dim, size_t fSize, size_t nnzMaxB, Real *p, Real *f, const Real3d &delta, double tolerance, size_t maxBCGIterations) :
    ImmersedEq(
        BaseData<Real, Int>(dim, fSize, nnzMaxB, delta, f, p),
        tolerance,
        maxBCGIterations
    ) {
}

template<typename Real, typename Int> //(I+2L^-1BT*B) * x = b, or equivilently, x = (I+2L^-1BT*B)^-1 b
void ImmersedEq<Real, Int>::LHSTimes(const SimpleArray<Real> &x, SimpleArray<Real> &result,
                                     const Singleton<Real> &multLinearOperationOutput, const Singleton<Real> &preMultResult) const {
    //TODO:maybe use a 2nd handle for scaling x?
    // std::cout << "x1 = " << GpuOut<Real>(x, hand) << std::endl;
    // std::cout << "Debugging LHS mult" << std::endl;

    auto fSize = baseData.allocatedFSize();
    auto pSize0 = baseData.allocatedPSize(0);
    auto invLXBTBx = baseData.allocatedPSize(1);


    multB(x, fSize, Singleton<Real>::ONE, Singleton<Real>::ZERO, false);// f <- B * x

    multB(fSize, pSize0, Singleton<Real>::TWO, Singleton<Real>::ZERO, true);// p <- B^T * (B * x)

    eds->solve(invLXBTBx, pSize0, baseData.hand4[0]); // workspace2 <- L^-1 * B^T * (B * x)

    invLXBTBx.add(x, &Singleton<Real>::ONE, &baseData.hand4[0]);

    auto& invLxBTBxPlusX = invLXBTBx;

    result.mult(preMultResult, &baseData.hand4[0]); //x <- x * preMultX

    result.add(invLxBTBxPlusX, &multLinearOperationOutput, &baseData.hand4[0]); //result <- result + preMultResult * x * preMultX
}

template<typename Real, typename Int>
SquareMat<Real> ImmersedEq<Real, Int>::LHSMat() {
    auto id = SquareMat<Real>::create(baseData.p.size());
    id.setToIdentity(baseData.hand4[0]);

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

        pSize.set(baseData.p, baseData.hand4[0]);

        multB(f, pSize, Singleton<Real>::TWO, Singleton<Real>::ONE, true);
        //p <- BT*f+p

        eds->solve(RHSSpace, pSize, baseData.hand4[0]);
    }
    return RHSSpace;
}


template<typename Real, typename Int>
void ImmersedEq<Real, Int>::solve(
    Real *result,
    size_t nnzB,
    Int *rowPointersB,
    Int *colPointersB,
    Real *valuesB,
    const bool multiStream
) {

    auto resultDevice = solve(nnzB, rowPointersB, colPointersB, valuesB, multiStream);
    resultDevice.get(result, baseData.hand4[0]);
}

template<typename Real, typename Int>
SimpleArray<Real> ImmersedEq<Real, Int>::solve(
    size_t nnzB,
    Int *rowPointersB,
    Int *colPointersB,
    Real *valuesB,
    const bool multithreadBCG
) {
    baseData.setB(nnzB, colPointersB, rowPointersB, valuesB);
    RHS(true);

    //TODO: should the initial guess be random, or the RHS of the equation?

    baseData.result.set(RHSSpace, baseData.hand4[0]);
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
    : BiCGSTAB<Real>(imEq.RHS(false), imEq.baseData.hand4.get(), &allocatedRHSHeightX7, &allocated9, tolerance, maxIterations),
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