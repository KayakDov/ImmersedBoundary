/**
 * @file YuriFileReader.h
 * @brief Handles file I/O and setup for Immersed Boundary equations using CUDA-accelerated sparse structures.
 * @date 1/7/26
 */

#ifndef CUDABANDED_YURIFILEREADER_H
#define CUDABANDED_YURIFILEREADER_H
#include <fstream>

#include "SparseCSR.h"
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
    constexpr static size_t numPSizeVecs = 6;//TODO: see if this works at 5
public:

    mutable Mat<Real> pSizeX5,  fSizeX2;
    const SimpleArray<Real> f = fSizeX2.col(0, true);
    const SimpleArray<Real> p = pSizeX5.col(0, true);
    mutable SimpleArray<Real> result = pSizeX5.col(numPSizeVecs - 1, true);

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
     * @param fSizeX2
     * @param pSizeX5
     * @param dim
     * @param delta
     */
    BaseData(SparseCSR<Real, Int> maxB, Mat<Real> fSizeX2, Mat<Real> pSizeX5, const GridDim &dim, const Real3d &delta);

    BaseData(const GridDim &dim, size_t fSize, size_t nnzMaxB, const Real3d &delta, Real *f, Real *p, Handle &hand);

    /**
     * @brief Resets all the base values. TODO: ask if modifications to B will be small instead of a total rewrite.
     */
    void setB(size_t nnzB, Int *offsetsB, Int *indsB, Real *valsB, Handle &hand);

    SimpleArray<Real> allocatedFSize() const;

    SimpleArray<Real> allocatedPSize(uint8_t ind) const;

    void printDenseB(Handle &hand) const;
};


template <typename Real, typename Int> class ImmersedEqSolver;
/**
 * @class ImmersedEq
 * @brief Represents the Immersed Boundary linear system operators.
 */
template <typename Real, typename Int = uint32_t>
class ImmersedEq {

    std::shared_ptr<Handle[]> hand5{new Handle[5]};

    friend ImmersedEqSolver<Real, Int>;

public: //TODO: this is public for debugging purposes only.
    BaseData<Real, Int> baseData;
private:

    mutable std::shared_ptr<SimpleArray<Real>> sparseMultBuffer{nullptr};

    Mat<Real> allocatedRHSHeightX7 = Mat<Real>::create(baseData.p.size(), 7);
    Vec<Real> allocated9 = Vec<Real>::create(9, hand5[0]);
    Real tolerance;
    size_t maxIterations;
    Event events11[11]{};
    Event lhsTimes;

    void multB(const SimpleArray<Real> &vec, SimpleArray<Real> &result, const Singleton<Real> &multProduct, const Singleton<Real> &preMultResult, bool transposeB) const;

    SimpleArray<Real> RHSSpace = SimpleArray<Real>::create(baseData.p.size(), hand5[0]);

public://TODO: this is public for debugging purposes only.
    std::shared_ptr<EigenDecompSolver<Real>> eds = createEDS(baseData.dim, baseData.allocatedPSize(0), &hand5[0], baseData.delta);
private:

    ImmersedEq(BaseData<Real, Int> baseData, double tolerance, size_t maxBCGIterations);


    /**
     * @brief Computes the Left-Hand Side (LHS) operation: $x = A \cdot b$.
     * @param x Input vector.
     * @param result Output product vector.
     * @param multLinearOperationOutput Scaling for the addition.
     * @param preMultResult Initial scaling factor for x.  Use Singleton<T>::ZERO to turn NaNs into 0's.  Use another 0 value to preserve NaNs.
     */
    void LHSTimes(const SimpleArray<Real> &x, SimpleArray<Real> &result, const Singleton<Real> &multLinearOperationOutput, const Singleton<Real> &preMultResult) const;

    SimpleArray<Real> solve(size_t nnzB, Int *rowOffsetsB, Int *colIndsB, Real *valuesB, bool multithreadBCG);

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