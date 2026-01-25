/**
 * @file YuriFileReader.h
 * @brief Handles file I/O and setup for Immersed Boundary equations using CUDA-accelerated sparse structures.
 * @date 1/7/26
 */

#ifndef CUDABANDED_YURIFILEREADER_H
#define CUDABANDED_YURIFILEREADER_H
#include <fstream>

#include "SparseCSC.cuh"
#include "Streamable.h"
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

/**
 * @class BaseData
 * @brief Container for the sparse matrix B and associated physical vectors on the GPU.
 * @tparam Real Precision type (float or double).
 * @tparam Int Integer type for row indices (default uint32_t).
 * @tparam IntCols Integer type for column offsets (default uint32_t).
 */
template <typename Real, typename Int = uint32_t> //TODO: f and p should never change.  B should be able to change everything.
class BaseData {

public:
    std::shared_ptr<Handle[]> hand4;//TODO: make things more private.

    mutable Mat<Real> pSizeX4,  fSizeX2;
    const SimpleArray<Real> f;
    const SimpleArray<Real> p;
    mutable SimpleArray<Real> result;

    SparseCSC<Real, Int> maxB; /**< Sparse matrix in CSC format */
    std::shared_ptr<SparseCSC<Real, Int>> B;
    const GridDim dim;
    const Real3d delta;

    /**
     * @brief Initializes GPU memory and loads data from FileMeta.
     */
    BaseData(const FileMeta &meta, const GridDim& dim);

    /**
     *
     * @param maxB
     * @param fSizeX2
     * @param pSizeX4
     * @param dim
     * @param delta
     */
    BaseData(SparseCSC<Real, Int> maxB, Mat<Real> fSizeX2, Mat<Real> pSizeX4, const GridDim &dim, const Real3d &delta);

    BaseData(const GridDim &dim, size_t fSize, size_t nnzMaxB, const Real3d &delta, Real * f, Real *p);

    /**
     * @brief Resets all the base values. TODO: ask if modifications to B will be small instead of a total rewrite.
     */
    void setB(size_t nnzB, Int * colsB, Int * rowsB, Real * valsB);

    SimpleArray<Real> allocatedFSize() const;

    SimpleArray<Real> allocatedPSize(bool ind) const;

    void printDenseB() const;
};


template <typename Real, typename Int> class ImmersedEqSolver;
/**
 * @class ImmersedEq
 * @brief Represents the Immersed Boundary linear system operators.
 */
template <typename Real, typename Int = uint32_t>
class ImmersedEq {

    friend ImmersedEqSolver<Real, Int>;
    BaseData<Real, Int> baseData;

    mutable std::shared_ptr<SimpleArray<Real>> sparseMultBuffer;

    Mat<Real> allocatedRHSHeightX7;
    Vec<Real> allocated9;
    Real tolerance;
    size_t maxIterations;

    void multB(const SimpleArray<Real> &vec, SimpleArray<Real> &result, const Singleton<Real> &multProduct, const Singleton<Real> &preMultResult, bool transposeThis) const;

    SimpleArray<Real> RHSSpace;

    std::shared_ptr<EigenDecompSolver<Real>> eds;

    ImmersedEq(BaseData<Real, Int> baseData, double tolerance, size_t maxBCGIterations);


    /**
     * @brief Computes the Left-Hand Side (LHS) operation: $x = A \cdot b$.
     * @param x Input vector.
     * @param result Output product vector.
     * @param multLinearOperationOutput Scaling for the addition.
     * @param preMultResult Initial scaling factor for x.
     */
    void LHSTimes(const SimpleArray<Real> &x, SimpleArray<Real> &result, const Singleton<Real> &multLinearOperationOutput, const Singleton<Real> &preMultResult) const;

    SimpleArray<Real> solve(size_t nnzB, Int *rowPointersB, Int *colPointersB, Real *valuesB, bool multithreadBCG = true);

public:

    /**
     * @brief Sets up the immersed equation system using the provided base data.
     */
    ImmersedEq(const GridDim &dim, size_t fSize, size_t nnzMaxB, Real *p, Real *f, const Real3d &delta, double tolerance, size_t maxBCGIterations);

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


    /**
     * Solves the equation.
     * @param result The solution will be placed here.
     * @param nnzB
     * @param rowPointersB
     * @param colPointersB
     * @param valuesB
     * @param hand4 A pointer to four handles.
     * @param allocatedRHSHeightX7 preallocated memory
     * @param allocated9 preallocated memory
     * @param tolerance The tolerance
     * @param maxIterations The maximum number of iterations.
     */
    void solve(Real *result, size_t nnzB, Int *rowPointersB, Int *colPointersB, Real *valuesB, bool multiStream = true);
};

/**
 * @class ImmersedEqSolver
 * @brief BiCGSTAB implementation specifically tailored for ImmersedEq systems.
 */
template <typename Real, typename Int = uint32_t>
class ImmersedEqSolver:  public BiCGSTAB<Real> {

    ImmersedEq<Real, Int> imEq;

    /**
     * @brief Implementation of the matrix-vector multiplication for the BiCGSTAB loop.
     */
    void mult(Vec<Real>& vec, Vec<Real>& product, Singleton<Real> multProduct, Singleton<Real> preMultResult) const override;

public:
    /**
     * @brief Constructor for the iterative solver.
     * @param p
     */
    ImmersedEqSolver(ImmersedEq<Real, Int> &imEq, Mat<Real> &allocatedRHSHeightX7, Vec<Real> &allocated9, Real tolerance, size_t maxIterations);
};




#endif //CUDABANDED_YURIFILEREADER_H