/**
* @file ImmersedEq.h
 * @brief High-performance CUDA implementation of the Immersed Boundary Method linear system.
 * * This file contains the primary operators for solving the pressure-force coupled system:
 * @f[ (L + 2 B^T B)x = 2 B^T f + p@f]
 * * @author E. Dov Neimand
 * @ingroup immersed_boundary
 */

#ifndef CUDABANDED_YURIFILEREADER_H
#define CUDABANDED_YURIFILEREADER_H
#include <fstream>

#include "deviceArrays/headers/sparse/SparseCSR.h"
#include "../deviceArrays/headers/sparse/SparseCSC.cuh"
#include "poisson/BoundaryConfig.cuh"
#include "solvers/BiCGSTAB.cuh"
#include "poisson/AxisSegment.cuh"


template <typename Real, typename Int> class ImmersedEqSolver;
template <typename Real> class EigenDecompSolver;



/** * @enum GridInd
 * @brief Offsets for multi-column Eulerian (Grid) memory buffers.
 */
enum class GridInd : size_t {
    p           = 0,  ///< Pressure field
    RHSPPrime   = 1,  ///< Right-Hand Side for the pressure correction (p')
    Result      = 2,  ///< General output buffer
    pPrime      = 2,  ///< Pressure correction field (aliases Result)
    LHS_BTBx    = 3, ///< LHS intermediate: $B^T B x$ (aliases RHS_BTF)
    LHS_invLBTBx= 4,  ///< LHS intermediate: $L^{-1} B^T B x$
    RHS         = 4,  ///< Global Right-Hand Side vector (aliases LHS_invLBTBx)
    boundRHSAdj = 5, ///< adjustmant to the right hand side of the equation on account of the boundary.
    Count       = 6   ///< Number of standard grid indices
};

/** * @enum LagrangeInd
 * @brief Offsets for Lagrangian memory buffers.
 */
enum class LagrangeInd : size_t {
    f         = 0, ///< Boundary force vector
    RHSFPrime = 1, ///< Right-Hand Side for boundary force correction
    LHS_Bx    = 2, ///< LHS intermediate: $B x$
    fPrime    = 2, ///< Force correction (aliases LHS_Bx)
    UGamma    = 2, ///< Prescribed boundary velocity (aliases LHS_Bx)
    Count     = 3  ///< Number of standard Lagrangian indices
};

/**
 * @class ImmersedEq
 * @brief Represents the Immersed Boundary linear system operators.
 */
template <typename Real, typename Int>
class ImmersedEq {

    const XYZ<Delta1d<Real>> delta;

    const GridDim dim;

    mutable Handle hand5[5]{}; ///< Array of 5 CUDA Handles for multi-streaming.
    mutable std::unique_ptr<SimpleArray<Real>> sparseMultBuffer = nullptr; ///< A buffer space for sparse vector multiplication.  The space grows as needed.
    Event events12[12]{}; ///< CUDA events for fine-grained synchronization.


    SimpleArray<Int> maxSparseInds; ///< Storage for maximum allowed sparse indices.
    SimpleArray<Int> maxSparseOffsets; ///< Storage for maximum allowed sparse offsets.
    SimpleArray<Real> maxSparseVals = SimpleArray<Real>::create(maxSparseInds.size(), hand5[0]); ///< Storage for maximum allowed sparse values.

    mutable Mat<Real> gridVecs = Mat<Real>::create(dim.size(), static_cast<size_t>(GridInd::Count) + 7);  ///< Storage for all vectors of length grid size.
    mutable Mat<Real> lagrangeVecs = Mat<Real>::create(maxSparseOffsets.size() - 1, static_cast<size_t>(LagrangeInd::Count)) ; ///< Storage for all the lagrangian vectors.

    /**
     * @brief Vector field storage on the staggered grid.
     * * Allocation accounts for face-centered velocities in 2d and 3D.
     */
    SimpleArray<Real> velocities = SimpleArray<Real>::create(dim.velocitiesStaggeredSize(), hand5[0]);

    /// CSR matrix mapping Eulerian space to Lagrangian space ($B$).
    std::unique_ptr<SparseMat<Real, Int>> B = std::make_unique<SparseCSR<Real, Int>>(SparseCSR<Real, Int>::create(dim.size(), maxSparseVals.subArray(0,0), maxSparseOffsets, maxSparseInds.subArray(0,0)));
    /// CSC matrix mapping Lagrangian space to discretized staggered vector field space ($R$).
    std::unique_ptr<SparseMat<Real, Int>> R = std::make_unique<SparseCSC<Real, Int>>(SparseCSC<Real, Int>::create(velocities.size(), maxSparseVals.subArray(0,0), maxSparseOffsets, maxSparseInds.subArray(0,0)));

    const Singleton<Real> dT; ///< Time step size ($\Delta t$).

    Event lhsTimes; ///< used for LHS operator multiplication multi streaming.

    /**
     * @brief Resets all the values in the selected sparse matrix.  Pass nullptr to leave values unchanged.
     * @param sparse The sparse matrix to be reset.
     * @param nnz The number of non zero values in the sparse matrix.
     * @param offsets The array of offsets in the sparse matrix.
     * @param inds The array of indices in the sparse matrix.
     * @param vals The values in the sparse matrix.
     * @param hand The context.
     */
    void setSparse(std::unique_ptr<SparseMat<Real, Int>> &sparse, size_t nnz, Int *offsets, Int *inds, Real *vals, Handle &hand);

    /**
     * @brief Retrieves the Lagrange vector at the desired index.  This can be used for both read and write.
     * @param ind The index of the desired lagrange vector.
     * @return The Lagrange vector at the desired index.
     */
    SimpleArray<Real> lagrangeVec(LagrangeInd ind) const;
    /**
     * @brief Retrieves the Eulerian vector at the desired index.  This can be used for both read and write.
     * @param ind The index of the desired lagrange vector.
     * @return The Eulerian vector at the desired index.
     */
    SimpleArray<Real> gridVec(GridInd ind)const ;

    /**
     * Sets the value for the variable RHS_{p'}.
     * @param hand The context.
     */
    void setRHSPPrime(Handle &hand);

    /**
     * Sets RHS_{F'}
     * @param hand the context.
     */
    void setRHSFPrime(Handle &hand);

    /**
     * Checks if the value passed is permitted as the number of non zero values.
     * @param nnzThe ptential number of non zer values for some sparse matrix.
     */
    void checkNNZ(size_t nnz) const;
    friend ImmersedEqSolver<Real, Int>;

    /**
     * Used to launch the BCG solver.
     */
    ImmersedEqSolver<Real, Int> solver;

    /**
     * used for eigen decomposition.
     */
    std::shared_ptr<EigenDecompSolver<Real>> eds;// = createEDS(boundary, &hand5[0], events12);

    /**
     * @brief Computes the Left-Hand Side (LHS) operation: $x = A \cdot b$.
     * @param x Input vector.
     * @param result Output product vector.
     * @param multLinearOperationOutput Scaling for the addition.
     * @param preMultResult Initial scaling factor for x.  Use Singleton<T>::ZERO to turn NaNs into 0's.  Use another 0 value to preserve NaNs.
     */
    void LHSTimes(const SimpleArray<Real> &x, SimpleArray<Real> &result, const Singleton<Real> &multLinearOperationOutput, const Singleton<Real> &preMultResult);


    /**
     * @brief solves the equation and returns a simple arra with the solution.
     * @param nnzB  The number of non zero elements in B.
     * @param offsetsB if csc == true, then this is the column offsets, otherwise the row offsets.
     * @param indsB  if csc == true then this should be row indices, otherwise the column indices.
     * @param valuesB The non zero values in B.
     * @param prime True to solve for p', otherwise false.
     * @return The solution, best to write this data to somewhere else as it will be overwritten when the method runs again.
     */
    SimpleArray<Real> solve(size_t nnzB, Int *offsetsB, Int *indsB, Real *valuesB, bool prime);

    /**
     * @brief Multiplies a sparse matrix times a dense vector.  Uses the existing allocated sparse multiplication buffer.
     * result <- multProduct * mat + preMultResult * result
     * @param mat The sparse matrix.
     * @param vec The vector.
     * @param result The result is stored here.  This vector will be overwritten.
     * @param multProduct alpha
     * @param preMultResult beta
     * @param transposeB true to transpose the matrix, false otherwise.
     */
    void multSparse(const std::unique_ptr<SparseMat<Real, Int>> &mat, const SimpleArray<Real> &vec, SimpleArray<Real> &result, const
                    Singleton<Real> &multProduct, const Singleton<Real> &preMultResult, bool transposeB) const;

    /**
     * @brief Prepares the RHS vector based on current state.
     * @param prime Boolean flag for correction system.
     */
    void setRHS(bool prime);

    /**
     * @brief solves the system.
     * @return The solution.
     */
    SimpleArray<Real> solve();
public:


    /**
     * @brief Debug method to materialize the full LHS matrix.
     * @warning Extremely memory intensive; for small grids only.
     */
    SquareMat<Real> LHSMat();

    /**
     * Gets the RHS.
     * @return
     */
    SimpleArray<Real> getRHS(SimpleArray<Real>& p, SimpleArray<Real>& f, SimpleArray<Real>& rhsAdjustment, SparseCSR<Real, Int>& B);

    /**
     * @brief Constructor for the ImmersedEq system.
     * * Initializes the CUDA environment, pre-allocates workspace matrices for
     * iterative solving, and sets up the Eigen-Decomposition for the Laplacian operator.
     * * @param boundary              Dimensions of the Eulerian grid.
     * @param fSize            Size of the Lagrangian force vector (number of boundary points).
     * @param nnzMax           Maximum expected non-zero elements in the sparse matrices B and R.
     * @param p                Initial pressure field array (Size: dim.size()).
     * @param f                Initial boundary force array (Size: fSize).
     * @param delta            Physical grid spacing (dx, dy, dz).
     * @param dT               Time step size ($\Delta t$).
     * @param tolerance        Convergence threshold for the BiCGSTAB iterative solver.
     * @param maxBCGIterations Maximum iterations permitted for the BiCGSTAB solver.
     */
    template<typename BoundaryConfigT>
    ImmersedEq(const BoundaryConfigT &boundary, size_t fSize, size_t nnzMax, Real *p, Real *f, double dT, Real tolerance, size_t maxBCGIterations);

    /**
     * @brief Solves the pressure-correction system for the Eulerian grid.
     *
 * This method executes the iterative solver for the operation:
     * $(I + 2L^{-1} B^T B)x = L^{-1}(B^T f + p)$
     * * @param result   [out] Host or Device pointer where the resulting grid field is written.
     * @param nnzB     Number of non-zero elements in the current Sparse Matrix B.
     * @param offsetsB Pointer to row (CSR) or column (CSC) offsets for Matrix B.
     * @param indsB    Pointer to column (CSR) or row (CSC) indices for Matrix B.
     * @param valuesB  Pointer to the non-zero values for Matrix B.
     */
    void solve(Real *result, size_t nnzB, Int *offsetsB, Int *indsB, Real *valuesB);

    /**
     * @brief Solves the coupled system for both Pressure (Eulerian) and Force (Lagrangian) corrections.
     * *
     * * Performs a full system solve using the provided boundary constraints (R) and
     * velocity fields (uStar), mapping results back to both domain and interface.
     * * @param resultP      [out] Pointer for the Eulerian grid result (Pressure field).
     * @param resultF      [out] Pointer for the Lagrangian boundary result (Force field).
     * @param nnzB         Non-zero count for Sparse Matrix B.
     * @param rowOffsetsB  Row offsets for CSR Matrix B.
     * @param colIndsB     Column indices for CSR Matrix B.
     * @param valuesB      Non-zero values for Matrix B.
     * @param nnzR         Non-zero count for Sparse Matrix R.
     * @param colOffsetsR  Column offsets for CSC Matrix R.
     * @param rowIndsR     Row indices for CSC Matrix R.
     * @param valuesR      Non-zero values for Matrix R.
     * @param UGamma       Pointer to prescribed boundary velocities/states.
     * @param uStar        Pointer to the intermediate staggered velocity field.  The first contiguouse third should be
     * in the x component of each velocity vector, then next contiguouse third in the y component of each velocity
     * vector, and the final contiguouse third thousl be the z component of the velocity vector.
     */
    void solve(Real *resultP, Real *resultF, size_t nnzB, Int *rowOffsetsB, Int *colIndsB, Real *valuesB, size_t nnzR,
               Int *colOffsetsR, Int *rowIndsR, Real *valuesR, Real *UGamma, Real *uStar);

    template<class BoundaryConfigT>
    ImmersedEq(BoundaryConfigT boundary, SimpleArray<Int> maxSparseInds, SimpleArray<Int> maxSparseOffsets, Singleton<Real> dT, Real tolerance, size_t maxBCGIterations);
};


/**
 * @class ImmersedEqSolver
 * @brief Specialized BiCGSTAB implementation for the Immersed Boundary operator.
 * * @tparam Real Floating point precision (float/double).
 * @tparam Int Integer precision for indexing (int32_t/int64_t).
 */
template <typename Real, typename Int>
class ImmersedEqSolver:  public BiCGSTAB<Real> {

    ImmersedEq<Real, Int>& imEq;

protected:
    /**
     * @brief Implements the custom linear operator @f$ A \cdot x @f$ for BiCGSTAB.
     * Overrides the BiCGSTAB base class multiplication interface.
     */
    void mult(Vec<Real>& vec, Vec<Real>& product, Singleton<Real> multProduct, Singleton<Real> preMultResult) const override;


public:
    /**
     * @brief Constructor for the iterative solver.
     * @param imEq Reference to the system operator class.
     * @param allocatedRHSHeightX7 Pre-allocated workspace matrix (Grid height x 7 columns).
     * @param allocated9 Pre-allocated workspace vector of size 9.
     * @param events11 Pointer to 11 CUDA events for synchronization.
     * @param tolerance Convergence threshold.
     * @param maxIterations Maximum BiCGSTAB iterations.
     */
    ImmersedEqSolver(ImmersedEq<Real, Int> &imEq, Mat<Real> &allocatedRHSHeightX7, Vec<Real> allocated9, Event* events11, Real tolerance, size_t maxIterations);

};

#endif //CUDABANDED_YURIFILEREADER_H