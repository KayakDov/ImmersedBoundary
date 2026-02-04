/**
 * @brief Handles file I/O and setup for Immersed Boundary equations using CUDA-accelerated sparse structures.
 * @date 1/7/26
 */

#ifndef CUDABANDED_YURIFILEREADER_H
#define CUDABANDED_YURIFILEREADER_H
#include <fstream>

#include "SparseCSR.h"
#include "SparseCSC.cuh"
#include "solvers/BiCGSTAB.cuh"
#include "solvers/EigenDecompSolver.h"

/**
 * @class FileMeta
 * @brief Extracts and stores metadata from binary input files.
 */
class FileMeta {
    public:
    size_t pSize;   /**< Size related to xp vector */
    size_t fSize;   /**< Size related to xf vector */
    size_t bRows;   /**< Number of rows in matrix B */
    size_t bCols;   /**< Number of columns in matrix B */
    size_t nnz;     /**< Number of non-zero elements in matrix B */
    std::ifstream& xFile; /**< Reference to the vector data stream */
    std::ifstream& bFile; /**< Reference to the matrix data stream */

    /**
     * @brief Construct metadata by reading headers from provided file streams.
     */
    FileMeta(std::ifstream& xFile, std::ifstream& bFile);
};


enum class GridInd : size_t {//eularian
    p         = 0,
    RHSPPrime = 1,
    Result    = 2,
    pPrime    = 2,
    EDS       = 3,
    LHS2      = 4,
    RHS       = 4,
    LHS1      = 5,
    Count     = 6
};

// Grouping F-size vector indices
enum class LagrangeInd : size_t {
    f         = 0,
    RHSFPrime = 1,
    LHS       = 2,
    fPrime    = 2,
    UGamma    = 2,
    Count     = 3
};

/**
 * @class BaseData
 * @brief Container for the sparse matrix B and associated physical vectors on the GPU.
 * @tparam Real Precision type (float or double).
 * @tparam Int Integer type for row indices (default uint32_t).
 * @tparam IntCols Integer type for column offsets (default uint32_t).
 */
template <typename Real, typename Int = uint32_t>
class BaseData {
    void checkNNZB(size_t nnzB) const;

public:

    const GridDim dim;

    SimpleArray<Int> maxSparseInds;
    SimpleArray<Int> maxSparseOffsets;
    SimpleArray<Real> maxSparseVals = SimpleArray<Real>::create(maxSparseInds.size(), nullptr);//TODO:move handles here and remove multiple nullptr streams

    mutable Mat<Real> gridVecs = Mat<Real>::create(dim.size(), static_cast<size_t>(GridInd::Count));
    mutable Mat<Real> lagrangeVecs = Mat<Real>::create(maxSparseOffsets.size() - 1, static_cast<size_t>(LagrangeInd::Count)) ;

    SimpleArray<Real> velocities = SimpleArray<Real>::create(dim.numDims() * dim.size()
        + dim.cols * dim.layers
        + dim.rows * dim.layers
        + dim.cols * dim.rows * (dim.layers > 1), nullptr); //TODO:move handles here and remove multiple nullptr streams

    std::shared_ptr<SimpleArray<Real>> f = std::make_shared<SimpleArray<Real>>(lagrangeVecs.col(static_cast<size_t>(LagrangeInd::f)));//TODO:these should probably be deleted and the primes should be resolved with passing indecies to setRHS.
    std::shared_ptr<SimpleArray<Real>> p = std::make_shared<SimpleArray<Real>>(gridVecs.col(static_cast<size_t>(GridInd::p)));
    mutable SimpleArray<Real> result = gridVecs.col(static_cast<size_t>(GridInd::Result));

    //CSR, maps from Eularian where p lives space to Lagrangian space where f lives (f rows, p cols)
    std::shared_ptr<SparseMat<Real, Int>> B = std::make_shared<SparseCSR<Real, Int>>(SparseCSR<Real, Int>::create(dim.size(), maxSparseVals.subArray(0,0), maxSparseOffsets, maxSparseInds.subArray(0,0)));//TODO:these would probably be safer as unique pointers, since reasignment doesn't change every copy.
    //CSC, maps from Lagrangian space to the discretized vector field space R^3 -> R^3 (3p + rows, f cols)
    std::shared_ptr<SparseMat<Real, Int>> R = std::make_shared<SparseCSC<Real, Int>>(SparseCSC<Real, Int>::create(velocities.size(), maxSparseVals.subArray(0,0), maxSparseOffsets, maxSparseInds.subArray(0,0)));//TODO:these would probably be safer as unique pointers, since reasignment doesn't change every copy.

    const Real3d delta;

    const Singleton<Real> dT;
    /**
     * @brief Initializes GPU memory and loads data from FileMeta.
     */
    BaseData(const FileMeta &meta, const GridDim &dim, Handle &hand);

    /**
     *
     * @param maxSparse
     * @param fSizeX3
     * @param pSizeX5
     * @param dim
     * @param delta
     */
    BaseData(const FileMeta &meta, const GridDim &dim, double dT, Handle &hand);

    BaseData(SimpleArray<Int> maxSparseInds, SimpleArray<Int> maxSparseOffsets, const GridDim &dim, const Real3d &delta,
             Singleton<Real> dT);

    BaseData(const GridDim &dim, size_t fSize, size_t nnzMax, const Real3d &delta, Real *f, Real *p, double dT,
             Handle &hand);

    /**
     * @brief Resets all the base values. TODO: ask if modifications to B will be small instead of a total rewrite.
     */
    void setSparse(std::shared_ptr<SparseMat<Real, Int>> &sparse, size_t nnz, Int *offsets, Int *inds, Real *vals, Handle &hand);

    void printDenseB(Handle &hand) const;

    SimpleArray<Real> lagrangeVec(LagrangeInd ind) const;
    SimpleArray<Real> gridVec(GridInd ind)const ;

    void setRHSPPrime(Handle &hand);

    void divergence(SimpleArray<Real> result, SimpleArray<Real> u, SimpleArray<Real> v, SimpleArray<Real> w,
                    Singleton<Real> scalar, Handle &hand);
};


template <typename Real, typename Int> class ImmersedEqSolver;
/**
 * @class ImmersedEq
 * @brief Represents the Immersed Boundary linear system operators.
 */
template <typename Real, typename Int>
class ImmersedEq {

public:
    std::shared_ptr<Handle[]> hand5{new Handle[5]};
    BaseData<Real, Int> baseData;
    mutable std::shared_ptr<SimpleArray<Real>> sparseMultBuffer = nullptr;  //TODO:these would probably be safer as unique pointers, since reasignment doesn't change every copy.
    Event events11[11]{};
private:

    friend ImmersedEqSolver<Real, Int>;
    Mat<Real> allocatedRHSHeightX7 = Mat<Real>::create(baseData.p->size(), 7);
    Vec<Real> allocated9 = Vec<Real>::create(9, hand5[0]);
    Real tolerance;
    size_t maxIterations;
    Event lhsTimes;

    SimpleArray<Real> RHSSpace = SimpleArray<Real>::create(baseData.p->size(), hand5[0]);

    std::shared_ptr<EigenDecompSolver<Real>> eds = createEDS(baseData.dim, baseData.gridVec(GridInd::EDS), &hand5[0], baseData.delta);


    /**
     * @brief Computes the Left-Hand Side (LHS) operation: $x = A \cdot b$.
     * @param x Input vector.
     * @param result Output product vector.
     * @param multLinearOperationOutput Scaling for the addition.
     * @param preMultResult Initial scaling factor for x.  Use Singleton<T>::ZERO to turn NaNs into 0's.  Use another 0 value to preserve NaNs.
     */
    void LHSTimes(const SimpleArray<Real> &x, SimpleArray<Real> &result, const Singleton<Real> &multLinearOperationOutput, const Singleton<Real> &preMultResult) const;

    void setRHSFPrime(Handle &hand);

    /**
     *
     * @param nnzB  The number of non zero elements in B.
     * @param offsetsB if csc == true, then this is the column offsets, otherwise the row offsets.
     * @param indsB  if csc == true then this should be row indices, otherwise the column indices.
     * @param valuesB The non zero values in B.
     * @param multithreadBCG true to multistream BiCGSTAB, false to run in a single stream.
     * @return The solution, best to write this data to somewhere else as it will be overwritten when the method runs again.
     */
    SimpleArray<Real> solve(size_t nnzB, Int *offsetsB, Int *indsB, Real *valuesB, bool multithreadBCG);

public:
    void multB(const SimpleArray<Real> &vec, SimpleArray<Real> &result, const Singleton<Real> &multProduct, const Singleton<Real> &preMultResult, bool transposeB) const;

    ImmersedEq(BaseData<Real, Int> baseData, double tolerance, size_t maxBCGIterations);



    /**
     * @brief Sets up the immersed equation system using the provided base data.
     */
    ImmersedEq(const GridDim &dim, size_t fSize, size_t nnzMaxSparse, Real *p, Real *f, const Real3d &delta, double dT,  double tolerance, size_t maxBCGIterations);

    /**
     * This method creates the LHS matrix.  For large matrices this may be inefficient.
     * It's really just meant for debugging purposes.
     * @param hand
     * @return
     */
    SquareMat<Real> LHSMat();



    /**
     * Generates the RHS value from the base data.
     * @param reset Check to false if this method was already called with the current base data.
     * @return The right hand side of the equation.
     */
    SimpleArray<Real> &RHS(bool reset = true);

    SimpleArray<Real> &solve(bool multiStream);

    /**
     *
     * @param result The result will be written here.
     * @param nnzB  The number of nonzero elements in B.
     * @param offsetsB if csc == true, then this is the column offsets, otherwise the row offsets.
     * @param indsB  if csc == true then this should be row indices, otherwise the column indices.
     * @param valuesB The non zero values in B.
     * @param multiStream true to multistream BiCGSTAB, false to run in a single stream.
     * @return
     */
    void solve(Real *result, size_t nnzB, Int *offsetsB, Int *indsB, Real *valuesB, bool multiStream);

    void solve(Real *resultP, Real *resultF, size_t nnzB, Int *rowOffsetsB, Int *colIndsB, Real *valuesB, size_t nnzR,
               Int *colOffsetsR, Int *rowIndsR, Real *valuesR, Real *UGamma, Real *uStar, bool multiStream);


};

/**
 * @class ImmersedEqSolver
 * @brief BiCGSTAB implementation specifically tailored for ImmersedEq systems.
 */
template <typename Real, typename Int>
class ImmersedEqSolver:  public BiCGSTAB<Real> {

    ImmersedEq<Real, Int>& imEq;

    /**
     * @brief Implementation of the matrix-vector multiplication for the BiCGSTAB loop.
     */
    void mult(Vec<Real>& vec, Vec<Real>& product, Singleton<Real> multProduct, Singleton<Real> preMultResult) const override;

public:
    /**
     * @brief Constructor for the iterative solver.
     * @param p
     */
    ImmersedEqSolver(ImmersedEq<Real, Int> &imEq, Mat<Real> &allocatedRHSHeightX7, Vec<Real> &allocated9, Event* events11, Real tolerance, size_t maxIterations);
};




#endif //CUDABANDED_YURIFILEREADER_H