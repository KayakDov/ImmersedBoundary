#include "BaseData.h"

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
BaseData<Real,
    Int>::BaseData(const FileMeta &meta, const GridDim &dim, Handle *hand) : p(SimpleArray<Real>::create(
                                                                                 meta.pSize, hand[0])),
                                                                             f(SimpleArray<Real>::create(
                                                                                 meta.fSize, hand[0])),
                                                                             maxB(SparseCSC<Real, Int>::create(
                                                                                 meta.nnz, meta.bRows, meta.bCols,
                                                                                 hand[0])),
                                                                             delta(
                                                                                 1.0 / dim.cols,
                                                                                 1.0 / dim.rows,
                                                                                 1.0 / dim.layers
                                                                             ),
                                                                             dim(dim) {
    meta.xFile >> GpuIn<Real>(p, hand[0], false, true);
    meta.bFile >> GpuIn<Real>(maxB.values, hand[0], false, true);
}

template<typename Real, typename Int>
BaseData<Real, Int>::BaseData(
    SparseCSC<Real, Int> maxB,
    Mat<Real> fSizeX2,
    Mat<Real> pSizeX3,
    const GridDim &dim,
    const Real3d &delta
) : pSizeX3(pSizeX3),
    fSizeX2(fSizeX2),
    maxB(maxB),
    dim(dim),
    delta(delta) {
}

template<typename Real, typename Int>
BaseData<Real, Int>::BaseData(const GridDim &dim, size_t fSize, size_t nnzMaxB, const Real3d &delta, Real *f, Real *p,
                              Handle &hand) : BaseData(
    SparseCSC<Real, Int>::create(nnzMaxB, fSize, dim.volume(), hand),
    Mat<Real>::create(fSize, 2),
    Mat<Real>::create(dim.volume(), 3),
    dim,
    delta
) {
    this->fSizeX2.col(0).set(f, hand);
    this->pSizeX3.col(0).set(p, hand);
}

template<typename Real, typename Int>
void BaseData<Real, Int>::setB(size_t nnzB, Int *colsB, Int *rowsB, Real *valsB, Handle &hand) {
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
SimpleArray<Real> &BaseData<Real, Int>::fSize() {
    return fSizeX2.col(1);
}

template<typename Real, typename Int>
SimpleArray<Real> &BaseData<Real, Int>::pSize(bool ind) {
    return pSizeX3.col(ind + 1);
}

template<typename Real, typename Int>
const SimpleArray<Real> &BaseData<Real, Int>::f() const {
    return fSizeX2.col(0);
}

template<typename Real, typename Int>
const SimpleArray<Real> &BaseData<Real, Int>::p() const {
    return pSizeX3.col(0);
}

template<typename Real>
EigenDecompSolver<Real> *createEDS(
    const GridDim &dim,
    SimpleArray<Real> &sizeOfP,
    Handle *hand,
    Real3d delta = Real3d(1, 1, 1)) {
    auto maxDimX2Or3 = Mat<Real>::create(dim.maxDim(), dim.numDims());
    auto rowsXrows = SquareMat<Real>::create(dim.rows);
    auto colsXcols = dim.cols != dim.rows ? SquareMat<Real>::create(dim.cols) : rowsXrows;

    if (dim.numDims() == 3) {
        //(SquareMat<T> &rowsXRows, SquareMat<T> &colsXCols, SquareMat<T> &depthsXDepths, Mat<T> &maxDimX3, SimpleArray<T>& sizeOfB, Handle* hand3, Real3d delta = Real3d(1, 1, 1));
        auto layersXLayers = dim.layers != dim.rows ? SquareMat<Real>::create(dim.layers) : rowsXrows;
        return new EigenDecompSolver3d<Real>(rowsXrows, colsXcols, layersXLayers, maxDimX2Or3, sizeOfP, hand, delta);
    } else {
        return new EigenDecompSolver2d<Real>(rowsXrows, colsXcols, maxDimX2Or3, sizeOfP, hand,
                                             Real2d(delta.x, delta.y));
    }
}

template<typename Real, typename Int>
size_t ImmersedEq<Real, Int>::sparseMultWorkspaceSize(bool max) {
    if (max)
        return std::min(
            baseData.maxB->multWorkspaceSize(baseData.p, baseData.f, Singleton<Real>::ONE, Singleton<Real>::ZERO, false,
                                             hand4[0]),
            baseData.maxB->multWorkspaceSize(baseData.f, RHSSpace, Singleton<Real>::ONE, Singleton<Real>::ZERO, true,
                                             hand4[0])
        );
    return std::min(
        baseData.B->multWorkspaceSize(baseData.p, baseData.f, Singleton<Real>::ONE, Singleton<Real>::ZERO, false,
                                      hand4[0]),
        baseData.B->multWorkspaceSize(baseData.f, RHSSpace, Singleton<Real>::ONE, Singleton<Real>::ZERO, true, hand4[0])
    );
}

template<typename Real, typename Int>
ImmersedEq<Real, Int>::ImmersedEq(BaseData<Real, Int> baseData, Handle *hand4, double tolerance,
                                  size_t maxBCGIterations) : baseData(baseData),
                                                             RHSSpace(SimpleArray<Real>::create(
                                                                 baseData.p.size(), hand4[0])),
                                                             sparseMultBuffer(
                                                                 std::make_shared<SparseCSC<Real, Int> >(
                                                                     SimpleArray<Real>::create(
                                                                         sparseMultWorkspaceSize(true), hand4[0]))),
                                                             sizeOfP(SimpleArray<Real>::create(
                                                                 baseData.p().size(), hand4[0])),
                                                             eds(createEDS(
                                                                 baseData.dim, sizeOfP, hand4, baseData.delta)),
                                                             hand4(hand4),
                                                             allocatedRHSHeightX7(
                                                                 Mat<Real>::create(baseData.p.size(), 7)),
                                                             allocated9(Vec<Real>::create(9, hand4[0])),
                                                             tolerance(tolerance),
                                                             maxIterations(maxBCGIterations) {
}

//BaseData(const GridDim &dim, size_t fSize, size_t nnzMaxB, const Real3d &delta, Real * f, Real *p, Handle &hand);
template<typename Real, typename Int>
ImmersedEq<Real, Int>::ImmersedEq(const GridDim &dim, Handle *hand4, size_t fSize, size_t nnzMaxB, Real *p, Real *f,
                                  const Real3d &delta, double tolerance, size_t maxBCGIterations) : ImmersedEq(
    BaseData<Real, Int>(dim, fSize, nnzMaxB, delta, f, p, hand4[0]),
    hand4,
    tolerance,
    maxBCGIterations
) {
}

template<typename Real, typename Int> //(I+2L^-1BT*B) * x = b, or equivilently, x = (I+2L^-1BT*B)^-1 b
void ImmersedEq<Real, Int>::LHSTimes(const SimpleArray<Real> &x, SimpleArray<Real> &result, Handle &hand,
                                     const Singleton<Real> &multInverseOp, const Singleton<Real> &preMultX) const {
    //TODO:maybe use a 2nd handle for scaling x?

    baseData.B->mult(x, baseData.fSize(), Singleton<Real>::ONE, Singleton<Real>::ZERO, false, sparseMultBuffer, hand);
    // f <- B * x
    baseData.B->mult(baseData.fSize(), baseData.pSize(0), Singleton<Real>::TWO, Singleton<Real>::ZERO, true,
                     sparseMultBuffer, hand);
    // p <- B^T * (B * x)
    eds->solve(baseData.pSize(0), baseData.pSize(1), hand); // workspace2 <- L^-1 * B^T * (B * x)

    result.mult(preMultX, &hand); //x <- x * preMultX
    result.add(x, &multInverseOp, &hand); //result <- result + multInverseOp * x * preMultX
    result.add(baseData.pSize(1), &multInverseOp, &hand); // result <- result + multInverseOp * (L^-1 * B^T * (B * x))}
}

template<typename Real, typename Int>
SquareMat<Real> ImmersedEq<Real, Int>::LHSMat(Handle &hand) {
    auto id = SquareMat<Real>::create(baseData.p.size());
    id.setToIdentity(hand);

    auto result = SquareMat<Real>::create(baseData.p.size());
    for (size_t i = 0; i < baseData.p.size(); ++i) {
        auto col = result.col(i);
        LHSTimes(id.col(i), static_cast<SimpleArray<Real> &>(col), hand, Singleton<Real>::ONE, Singleton<Real>::ZERO);
    }
    return result;
}

template<typename Real, typename Int>
SimpleArray<Real> &ImmersedEq<Real, Int>::RHS(Handle &hand, bool reset) {
    //TODO This method ovrewrites p and should not.
    if (reset) {
        baseData.B->mult(baseData.f, baseData.pSize(), Singleton<Real>::TWO, Singleton<Real>::ONE, true, hand);
        //p <- BT*f+p
        eds->solve(RHSSpace, baseData.pSize(), hand);
    }
    return RHSSpace;
}

template<typename Real, typename Int>
void ImmersedEq<Real, Int>::solve(
    SimpleArray<Real> result,
    size_t nnzB,
    Int *rowPointersB,
    Int *colPointersB,
    Real *valuesB
) {
    baseData.setB(nnzB, colPointersB, rowPointersB, valuesB, hand4[0]);

    size_t newWorkSize = sparseMultWorkspaceSize();//TODO:this check may be expensive.  Experimentation or more research may show it's not neccessary.
    if (newWorkSize > sparseMultBuffer->size()) sparseMultBuffer = std::make_shared<SimpleArray<Real>>(newWorkSize, hand4[0]);

    ImmersedEqSolver(hand4, *this, allocatedRHSHeightX7, allocated9, tolerance, maxIterations)
            .solveUnpreconditionedBiCGSTAB(result);
}


template<typename Real, typename Int>
ImmersedEqSolver<Real, Int>::ImmersedEqSolver(
    Handle *hand4,
    ImmersedEq<Real, Int> &imEq,
    Mat<Real> &allocatedRHSHeightX7,
    Vec<Real> &allocated9,
    Real tolerance,
    size_t maxIterations
)
    : BiCGSTAB<Real>(imEq.RHS(hand4[0]), hand4, &allocatedRHSHeightX7, &allocated9, tolerance, maxIterations),
      imEq(imEq) {
}

template<typename Real, typename Int>
void ImmersedEqSolver<Real, Int>::mult(Vec<Real> &vec, Vec<Real> &product, Handle &hand, Singleton<Real> multProduct,
                                       Singleton<Real> preMultResult) const {
    SimpleArray<Real> vecSA(vec), productSA(product);
    return imEq.LHSTimes(vecSA, productSA, hand, multProduct, preMultResult);
}


// If your SparseCSC usually uses uint32_t and uint64_t:
template class BaseData<float, uint32_t>;
template class BaseData<double, uint32_t>;
template class BaseData<float, size_t>;
template class BaseData<double, size_t>;

template class ImmersedEq<float, uint32_t>;
template class ImmersedEq<double, uint32_t>;
template class ImmersedEq<float, size_t>;
template class ImmersedEq<double, size_t>;

template class ImmersedEqSolver<float, uint32_t>;
template class ImmersedEqSolver<double, uint32_t>;
template class ImmersedEqSolver<float, size_t>;
template class ImmersedEqSolver<double, size_t>;
