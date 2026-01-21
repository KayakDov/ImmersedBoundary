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
    mutable Mat<Real> pSizeX3,  fSizeX2;
    const SimpleArray<Real> f;
    const SimpleArray<Real> p;

    SparseCSC<Real, Int> maxB; /**< Sparse matrix in CSC format */
    std::shared_ptr<SparseCSC<Real, Int>> B;
    const GridDim dim;
    const Real3d delta;

    /**
     * @brief Initializes GPU memory and loads data from FileMeta.
     */
    BaseData(const FileMeta &meta, const GridDim& dim, Handle* hand);

    /**
     *
     * @param maxB
     * @param fSizeX2
     * @param pSizeX3
     * @param dim
     * @param delta
     */
    BaseData(SparseCSC<Real, Int> maxB, Mat<Real> fSizeX2, Mat<Real> pSizeX3, const GridDim &dim, const Real3d &delta);

    BaseData(const GridDim &dim, size_t fSize, size_t nnzMaxB, const Real3d &delta, Real * f, Real *p, Handle &hand);

    /**
     * @brief Resets all the base values. TODO: ask if modifications to B will be small instead of a total rewrite.
     */
    void setB(size_t nnzB, Int * colsB, Int * rowsB, Real * valsB, Handle &hand);

    SimpleArray<Real> allocatedFSize();

    SimpleArray<Real> allocatedPSize(bool ind) const;

    void printDenseB(Handle &hand) const;
};

/**
 * @class ImmersedEq
 * @brief Represents the Immersed Boundary linear system operators.
 */
template <typename Real, typename Int = uint32_t>
class ImmersedEq {

    mutable BaseData<Real, Int> baseData;
    SimpleArray<Real> RHSSpace;
    mutable std::shared_ptr<SimpleArray<Real>> sparseMultBuffer;


    Handle *hand4;
    Mat<Real> allocatedRHSHeightX7;
    Vec<Real> allocated9;
    Real tolerance;
    size_t maxIterations;

public:
    std::shared_ptr<EigenDecompSolver<Real>> eds;


    size_t sparseMultWorkspaceSize(bool max = false);



    /**
     * @brief Sets up the immersed equation system using the provided base data.
     */
    ImmersedEq(const BaseData<Real, Int> &baseData, Handle *hand4, double tolerance, size_t maxBCGIterations);

    ImmersedEq(const GridDim &dim, Handle *hand4, size_t fSize, size_t nnzMaxB, Real *p, Real *f, const Real3d &delta,
               double tolerance, size_t maxBCGIterations);

    //  size_t fSize, size_t nnzB, const double deltaX, const double deltaY, const double deltaZ,/


    /**
     * @brief Computes the Left-Hand Side (LHS) operation: $x = A \cdot b$.
     * @param x Input vector.
     * @param result Output product vector.
     * @param hand CUDA Handle for stream management.
     * @param multInverseOp Scaling for the addition.
     * @param preMultX Initial scaling factor for x.
     */
    void LHSTimes(const SimpleArray<Real> &x, SimpleArray<Real> &result, Handle &hand, const Singleton<Real> &multInverseOp,
             const Singleton<Real> &preMultX) const;

    /**
     * This method creates the LHS matrix.  For large matrices this may be inefficient.
     * It's really just meant for debugging purposes.
     * @param hand
     * @return
     */
    SquareMat<Real> LHSMat(Handle &hand);

    /**
     * Generates the RHS value from the base data.
     * @param hand The context.
     * @param reset Check to false if this method was already called with the current base data.
     * @return The right hand side of the equation.
     */
    SimpleArray<Real>& RHS(Handle &hand, bool reset = true);

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
    void solve(SimpleArray<Real> result, size_t nnzB, Int *rowPointersB, Int *colPointersB, Real *valuesB);

    void solve(Real *result, size_t nnzB, Int *rowPointersB, Int *colPointersB, Real *valuesB);

    SimpleArray<Real> solve(size_t nnzB, Int *rowPointersB, Int *colPointersB, Real *valuesB);
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
    void mult(Vec<Real>& vec, Vec<Real>& product,  Handle& hand, Singleton<Real> multProduct,
              Singleton<Real> preMultResult) const override;

public:
    /**
     * @brief Constructor for the iterative solver.
     * @param p
     */
    ImmersedEqSolver(Handle *hand4, ImmersedEq<Real, Int> imEq, Mat<Real> &allocatedRHSHeightX7, Vec<Real> &allocated9, Real tolerance, size_t maxIterations);
};


#endif //CUDABANDED_YURIFILEREADER_H