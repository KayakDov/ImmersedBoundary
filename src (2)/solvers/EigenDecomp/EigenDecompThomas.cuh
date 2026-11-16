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
 * @brief A hybrid 3D Poisson solver using eigen-decomposition along Y and X and the Thomas Algorithm along Z.
 * This class implements a semi-direct solver for the discrete 3D Laplacian. It optimizes
 * the Fast Diagonalization Method by diagonalizing only two dimensions, X (columns) and Y (rows).
 * The X dimension (columns) is solved in the transformed eigen-space as a set of independent
 * tridiagonal systems using the Thomas Algorithm (TDMA), one system for each (x, y) pair.
 * This approach is particularly useful when:
 * 1. The Z-dimension has boundary conditions or grid spacing that make full diagonalization difficult.
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

    /** The boundary conditions for the z dimension. */
    const AxisSegmentHost<SegmentT> boundaryZ;

    /**
     * @brief Solves the tridiagonal systems in the eigen-space.
     * Applies the Thomas algorithm along the depths (the Z direction) of the tensor. For each (i, j)
     * in the eigen-transformed XY plane, it solves the system:
     * \f[ ( L_z + \lambda_x[i] + \lambda_y[j] - \sigma I ) \tilde{u} = \tilde{f} \f]
     * where \f$ L_z \f$ is the 1D tridiagonal Laplacian operator for the Z direction and
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
     * @param boundZ The boundary conditions in the z dimension.
     * @param sizeOfBX3 A 3-column matrix providing scratch space for [Solution, SuperPrime, RHSPrime].
     * @param isSingular true if the laplacian is singular.
     * @param helmholtzShift set to a non zero value to solve (L - sigma I)x =  b where the helmholtzShift is sigma.
     */
    EigenDecompThomas(const Eigen<T> &eigen, const AxisSegmentHost<SegmentT> &boundZ, Mat<T> &sizeOfBX3, bool isSingular, T helmholtzShift = 0);

    /**
     * @brief Constructs the hybrid solver and manages its own internal memory.
     * @param boundary The boundary conditions.
     * @param hands At least 3 handles.
     * @param sizeOfBX3 allocated memory for the right hand side, and the thomas calculations.  It should have as many rows
     * as there are rows in L, and 3 columns.
     * @param helmholtzShift set to a non zero value to solve (L - sigma I)x =  b where the helmholtzShift is sigma.
     */
    template<typename SegX, typename SegY>
    EigenDecompThomas(const BoundaryConfigHost<T, SegX, SegY, SegmentT>& boundary, Handles &hands, Mat<T> sizeOfBX3, T helmholtzShift = 0);




    /**
     * @brief Constructs the hybrid solver and manages its own internal memory.
     * @param boundary The boundary conditions.
     * @param hands At least 3 handles.
     * @param helmholtzShift set to a non zero value to solve (L - sigma I)x =  b where the helmholtzShift is sigma.
     */
    template<class SegX, class SegY>
    EigenDecompThomas(const BoundaryConfigHost<T, SegX, SegY, SegmentT> &boundary, Handles &hands, T helmholtzShift = 0);

    void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const override;
};


// Deduction guide for the constructor without a Mat workspace
template<typename Real, typename SegX, typename SegY, typename SegZ>
EigenDecompThomas(const BoundaryConfigHost<Real, SegX, SegY, SegZ>&, Handles&, Real)
    -> EigenDecompThomas<Real, SegZ>;

// Deduction guide for the constructor with a Mat workspace
template<typename Real, typename SegX, typename SegY, typename SegZ>
EigenDecompThomas(const BoundaryConfigHost<Real, SegX, SegY, SegZ>&, Handles&, Mat<Real>, Real)
    -> EigenDecompThomas<Real, SegZ>;

#endif //CUDABANDED_EIGENDECOMPTHOMAS_CUH