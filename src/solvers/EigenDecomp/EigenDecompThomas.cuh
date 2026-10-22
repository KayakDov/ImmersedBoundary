/**
 * @file EigenDecompThomas.cuh
 * @brief Eigen-decomposition solver that applies the Thomas algorithm along one axis.
 * @ingroup eigen_decomp
 */

#ifndef CUDABANDED_EIGENDECOMPTHOMAS_CUH
#define CUDABANDED_EIGENDECOMPTHOMAS_CUH
#include "EigenDecomp3d.cuh"


/**
 * @class EigenDecompThomas
 * @brief A hybrid 3D Poisson solver using eigen-decomposition along Y and Z and the Thomas Algorithm along X.
 * This class implements a semi-direct solver for the discrete 3D Laplacian. It optimizes
 * the Fast Diagonalization Method by diagonalizing only two dimensions, Y (rows) and Z (layers).
 * The X dimension (columns) is solved in the transformed eigen-space as a set of independent
 * tridiagonal systems using the Thomas Algorithm (TDMA), one system for each (y, z) pair.
 * This approach is particularly useful when:
 * 1. The X-dimension has boundary conditions or grid spacing that make full diagonalization difficult.
 * 2. You want to reduce the number of high-latency matrix multiplications by one of the three.
 * @tparam T Floating-point type (float or double).
 */
template<typename T, typename SegmentT>
class EigenDecompThomas : public EigenDecomp3d<T> {
protected:
    /** @brief GPU workspace for the Thomas solver's modified super-diagonal coefficients. */
    Tensor<T> workSpaceSuperPrime;

    /** @brief GPU workspace for the Thomas solver's modified intermediate RHS values. */
    Tensor<T> workSpaceRHSPrime;

    /** The boundary conditions for the x dimension. */
    const AxisSegmentHost<SegmentT> boundaryX;

    /**
     * @brief Solves the tridiagonal systems in the eigen-space.
     * Applies the Thomas algorithm along the columns (the X direction) of the tensor. For each (j, k)
     * in the eigen-transformed YZ plane, it solves the system:
     * \f[ ( L_x + \lambda_y[j] + \lambda_z[k] - \sigma I ) \tilde{u} = \tilde{f} \f]
     * where \f$ L_x \f$ is the 1D tridiagonal Laplacian operator for the X direction and
     * \f$ \sigma \f$ is the Helmholtz shift.
     * @param src Input tensor in transformed eigen-space (f-tilde).
     * @param dst Output tensor in transformed eigen-space (u-tilde).
     * @param hand CUDA handle for kernel execution.
     */
    void multLEigenValInverse(const SimpleArray<T> &src, SimpleArray<T> &dst, Handle &hand) const override;

public:
    /**
     * @brief Constructs the hybrid solver using existing matrix workspaces.
     * @param eigen The eigen vectors and values.
     * @param sizeOfBX3 A 3-column matrix providing scratch space for [Solution, SuperPrime, RHSPrime].
     * @param isSingular true if the laplacian is singular.
     * @param helmholtzShift set to a non zero value to solve (L - sigma I)x =  b where the helmholtzShift is sigma.
     */
    EigenDecompThomas(const Eigen<T> &eigen, const AxisSegmentHost<SegmentT> &boundX, Mat<T> &sizeOfBX3, bool isSingular, T helmholtzShift = 0);

    /**
     * @brief Constructs the hybrid solver and manages its own internal memory.
     * @param boundary The boundary conditions.
     * @param hands At least 3 handles.
     * @param sizeOfBX3 allocated memory for the right hand side, and the thomas calculations.  It should have as many rows
     * as there are rows in L, and 3 columns.
     * @param helmholtzShift set to a non zero value to solve (L - sigma I)x =  b where the helmholtzShift is sigma.
     */
    template<typename SegY, typename SegZ>
    EigenDecompThomas(const BoundaryConfigHost<T, SegmentT, SegY, SegZ>& boundary, Handles &hands, Mat<T> sizeOfBX3, T helmholtzShift = 0);




    /**
     * @brief Constructs the hybrid solver and manages its own internal memory.
     * @param boundary The boundary conditions.
     * @param hands At least 3 handles.
     * @param helmholtzShift set to a non zero value to solve (L - sigma I)x =  b where the helmholtzShift is sigma.
     */
    template<class SegY, class SegZ>
    EigenDecompThomas(const BoundaryConfigHost<T, SegmentT, SegY, SegZ> &boundary, Handles &hands, T helmholtzShift = 0);

    void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const override;
};


// Deduction guide for the constructor without a Mat workspace
template<typename Real, typename SegX, typename SegY, typename SegZ>
EigenDecompThomas(const BoundaryConfigHost<Real, SegX, SegY, SegZ>&, Handles&, Real)
    -> EigenDecompThomas<Real, SegX>;

// Deduction guide for the constructor with a Mat workspace
template<typename Real, typename SegX, typename SegY, typename SegZ>
EigenDecompThomas(const BoundaryConfigHost<Real, SegX, SegY, SegZ>&, Handles&, Mat<Real>, Real)
    -> EigenDecompThomas<Real, SegX>;

#endif //CUDABANDED_EIGENDECOMPTHOMAS_CUH