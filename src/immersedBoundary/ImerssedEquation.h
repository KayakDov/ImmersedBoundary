/**
 * @brief Handles file I/O and setup for Immersed Boundary equations using CUDA-accelerated sparse structures.
 * @date 1/7/26
 */

#ifndef CUDABANDED_YURIFILEREADER_H
#define CUDABANDED_YURIFILEREADER_H
#include <fstream>

#include "sparse/SparseCSR.h"
#include "../deviceArrays/headers/sparse/SparseCSC.cuh"
#include "solvers/BiCGSTAB.cuh"
#include "solvers/EigenDecompSolver.h"

template <typename Real, typename Int> class ImmersedEq;

enum class GridInd : size_t {//eularian
    p         = 0,
    RHSPPrime = 1,
    Result    = 2,
    pPrime    = 2,
    EDS       = 3,
    LHS_invLBTBx      = 4,
    RHS       = 4,
    LHS_BTBx      = 5,
    Count     = 6
};

// Grouping F-size vector indices
enum class LagrangeInd : size_t {
    f         = 0,
    RHSFPrime = 1,
    LHS_Bx       = 2,
    fPrime    = 2,
    UGamma    = 2,
    Count     = 3
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

template <typename Real, typename Int>
class SolverLauncher {
private:
    Real tolerance;
    size_t maxIterations;
    Vec<Real> allocated9;
    Mat<Real> allocatedRHSHeightX7;

public:

    SolverLauncher(const Real &tolerance, size_t max_iterations, const Mat<Real> &gridVecs, Handle &hand);
    void launch(ImmersedEq<Real, Int> &imEq, Event *events11, SimpleArray<Real> &result);

};

template <typename Real, typename Int> class ImmersedEqSolver;
/**
 * @class ImmersedEq
 * @brief Represents the Immersed Boundary linear system operators.
 */
template <typename Real, typename Int>
class ImmersedEq {

    std::shared_ptr<Handle[]> hand5{new Handle[5]};
    mutable std::unique_ptr<SimpleArray<Real>> sparseMultBuffer = nullptr;
    Event events11[11]{};

    const GridDim dim;

    SimpleArray<Int> maxSparseInds;
    SimpleArray<Int> maxSparseOffsets;
    SimpleArray<Real> maxSparseVals = SimpleArray<Real>::create(maxSparseInds.size(), hand5[0]);

    mutable Mat<Real> gridVecs = Mat<Real>::create(dim.size(), static_cast<size_t>(GridInd::Count) + 7);
    mutable Mat<Real> lagrangeVecs = Mat<Real>::create(maxSparseOffsets.size() - 1, static_cast<size_t>(LagrangeInd::Count)) ;


    SimpleArray<Real> velocities = SimpleArray<Real>::create(dim.numDims() * dim.size()
        + dim.cols * dim.layers
        + dim.rows * dim.layers
        + dim.cols * dim.rows * (dim.layers > 1), hand5[0]);

    //CSR, maps from Eularian where p lives space to Lagrangian space where f lives (f rows, p cols)
    std::unique_ptr<SparseMat<Real, Int>> B = std::make_unique<SparseCSR<Real, Int>>(SparseCSR<Real, Int>::create(dim.size(), maxSparseVals.subArray(0,0), maxSparseOffsets, maxSparseInds.subArray(0,0)));
    //CSC, maps from Lagrangian space to the discretized vector field space R^3 -> R^3 (3p + rows, f cols)
    std::unique_ptr<SparseMat<Real, Int>> R = std::make_unique<SparseCSC<Real, Int>>(SparseCSC<Real, Int>::create(velocities.size(), maxSparseVals.subArray(0,0), maxSparseOffsets, maxSparseInds.subArray(0,0)));

    const Real3d delta;

    const Singleton<Real> dT;

    Event lhsTimes;

    /**
     * @brief Resets all the base values. TODO: ask if modifications to B will be small instead of a total rewrite.
     */
    void setSparse(std::unique_ptr<SparseMat<Real, Int>> &sparse, size_t nnz, Int *offsets, Int *inds, Real *vals, Handle &hand);

    SimpleArray<Real> lagrangeVec(LagrangeInd ind) const;
    SimpleArray<Real> gridVec(GridInd ind)const ;

    void setRHSPPrime(Handle &hand);

    void divergence(SimpleArray<Real> result, SimpleArray<Real> u, SimpleArray<Real> v, SimpleArray<Real> w,
                    Singleton<Real> scalar, Handle &hand);

    void checkNNZB(size_t nnzB) const;
    friend ImmersedEqSolver<Real, Int>;


    SolverLauncher<Real, Int> solverLauncher;

    std::shared_ptr<EigenDecompSolver<Real>> eds = createEDS(dim, gridVec(GridInd::EDS), &hand5[0], delta, events11);


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
     * @param prime
     * @return The solution, best to write this data to somewhere else as it will be overwritten when the method runs again.
     */
    SimpleArray<Real> solve(size_t nnzB, Int *offsetsB, Int *indsB, Real *valuesB, bool prime);

    void multSparse(const std::unique_ptr<SparseMat<Real, Int>> &mat, const SimpleArray<Real> &vec, SimpleArray<Real> &result, const
                    Singleton<Real> &multProduct, const Singleton<Real> &preMultResult, bool transposeB) const;

    ImmersedEq(SimpleArray<Int> maxSparseInds, SimpleArray<Int> maxSparseOffsets, const GridDim &dim, const Real3d &delta, Singleton<Real> dT, double tolerance, size_t maxBCGIterations);


    /**
     * This method creates the LHS matrix.  For large matrices this may be inefficient.
     * It's really just meant for debugging purposes.
     * @param hand
     * @return
     */
    SquareMat<Real> LHSMat();

    /**
     * Generates the RHS value from the base data.
     * @return The right hand side of the equation.
     */
    void setRHS(bool prime);


    SimpleArray<Real> solve();
public:

    ImmersedEq(
        const GridDim &dim,
        size_t fSize,
        size_t nnzMax,
        Real *p,
        Real *f,
        const Real3d &delta,
        double dT,
        double tolerance,
        size_t maxBCGIterations
    );

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
    void solve(Real *result, size_t nnzB, Int *offsetsB, Int *indsB, Real *valuesB);

    void solve(Real *resultP, Real *resultF, size_t nnzB, Int *rowOffsetsB, Int *colIndsB, Real *valuesB, size_t nnzR,
               Int *colOffsetsR, Int *rowIndsR, Real *valuesR, Real *UGamma, Real *uStar);


};

#endif //CUDABANDED_YURIFILEREADER_H