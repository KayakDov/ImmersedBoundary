/**
 * @file YuriFileReader.h
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
    LHS1       = 5,
    Count     = 6
};

// Grouping F-size vector indices
enum class LagrangeInd : size_t {
    f         = 0,
    RHSFPrime = 1,
    fPrime    = 1,
    LHS       = 2,
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

    mutable Mat<Real> gridVecs,  lagrangeVecs;
    std::shared_ptr<SimpleArray<Real>> f = std::make_shared<SimpleArray<Real>>(lagrangeVecs.col(static_cast<size_t>(LagrangeInd::f)));
    std::shared_ptr<SimpleArray<Real>> p = std::make_shared<SimpleArray<Real>>(gridVecs.col(static_cast<size_t>(GridInd::p)));
    mutable SimpleArray<Real> result = gridVecs.col(static_cast<size_t>(GridInd::Result));

    SparseCSR<Real, Int> maxB;
    std::shared_ptr<SparseMat<Real, Int>> B;
    const GridDim dim;
    const Real3d delta;

    /**
     * @brief Initializes GPU memory and loads data from FileMeta.
     */
    BaseData(const FileMeta &meta, const GridDim &dim, Handle &hand);
    /**
     *
     * @param maxB
     * @param fSizeX3
     * @param pSizeX5
     * @param dim
     * @param delta
     */
    BaseData(SparseCSR<Real, Int> maxB, Mat<Real> fSizeX3, Mat<Real> pSizeX5, const GridDim &dim, const Real3d &delta);

    BaseData(const GridDim &dim, size_t fSize, size_t nnzMaxB, const Real3d &delta, Real *f, Real *p, Handle &hand);

    /**
     * @brief Resets all the base values. TODO: ask if modifications to B will be small instead of a total rewrite.
     */
    void setB(size_t nnzB, Int *rowOffsetsB, Int *colIndsB, Real *valsB, Handle &hand);

    void printDenseB(Handle &hand) const;

    SimpleArray<Real> lagrangeVec(LagrangeInd ind) const;
    SimpleArray<Real> gridVec(GridInd ind)const ;
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
    mutable std::shared_ptr<SimpleArray<Real>> sparseMultBuffer;
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

    ImmersedEq(BaseData<Real, Int> baseData, double tolerance, size_t maxBCGIterations, std::shared_ptr<SimpleArray<Real>> sparseMultBuffer = nullptr);

    /**
     * @brief Sets up the immersed equation system using the provided base data.
     */
    ImmersedEq(const GridDim &dim, size_t fSize, size_t nnzMaxB, Real *p, Real *f, const Real3d &delta, double tolerance, size_t maxBCGIterations, std::shared_ptr<SimpleArray<Real>> sparseMultBuffer = nullptr);

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
    SimpleArray<Real>& RHS(bool reset = true);

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