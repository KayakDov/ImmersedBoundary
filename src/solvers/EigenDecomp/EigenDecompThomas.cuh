//
// Created by usr on 2/19/26.
//

#ifndef CUDABANDED_EIGENDECOMPTHOMAS_CUH
#define CUDABANDED_EIGENDECOMPTHOMAS_CUH
#include "EigenDecomp3d.cuh"


/**
 * @class EigenDecompThomas
 * @brief A hybrid 3D Poisson solver using Eigen-decomposition (X, Y) and the Thomas Algorithm (Z).
 * * This class implements a semi-direct solver for the discrete 3D Laplacian. It optimizes
 * the Fast Diagonalization Method by only diagonalizing two dimensions (typically X and Y).
 * The third dimension (Z) is solved in the transformed eigen-space as a set of independent
 * tridiagonal systems using the Thomas Algorithm (TDMA).
 * * This approach is particularly useful when:
 * 1. The Z-dimension has boundary conditions or grid spacing that make full diagonalization difficult.
 * 2. You want to reduce the number of high-latency matrix multiplications (multEZ) required.
 * * @tparam T Floating-point type (float or double).
 */
template<typename T>
class EigenDecompThomas : public EigenDecomp3d<T> {

    /** @brief GPU workspace for the Thomas solver's modified super-diagonal coefficients. */
    Tensor<T> workSpaceSuperPrime;

    /** @brief GPU workspace for the Thomas solver's modified intermediate RHS values. */
    Tensor<T> workSpaceRHSPrime;

    /** @brief Grid spacing in the Z-direction, used to build the tridiagonal coefficients. */
    double deltaZ;

    /**
     * @brief Overridden to be a no-op (or specialized) since Z-direction is handled by Thomas solve.
     * @note In this hybrid model, the E_z transform is replaced by the tridiagonal solve in setUTilde.
     */
    void multEZ(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE)  const override;

    /**
     * @brief Solves the tridiagonal systems in the eigen-space.
     * * Applies the Thomas algorithm across the "depths" of the tensor. For each (i, j)
     * in the eigen-transformed XY plane, it solves the system:
     * \f[ ( \lambda_x[i] + \lambda_y[j] + L_z ) \tilde{u} = \tilde{f} \f]
     * where \f$ L_z \f$ is the 1D tridiagonal Laplacian operator for the Z-direction.
     * * @param src Input tensor in transformed eigen-space (f-tilde).
     * @param dst Output tensor in transformed eigen-space (u-tilde).
     * @param hand CUDA handle for kernel execution.
     */
    void setUTilde(const Tensor<T> &src, Tensor<T> &dst, Handle &hand) const override;

    /**
     * @brief Internal constructor for resource delegation.
     */
    EigenDecompThomas(const GridDim &dim, Handle *hand3, const Real3d &delta, Mat<T> sizeOfBX3, Event *event);


public:
    /**
     * @brief Constructs the hybrid solver using existing matrix workspaces.
     * * @param rowsXRowsP1 Matrix for X-direction eigenvectors. It will be overwritten. [rows x rows+1]
     * @param colsXColsP1 Matrix for Y-direction eigenvectors. It will be overwritten. [cols x cols+1]
     * @param depthsXDepthsP1 Matrix for Z-direction (Thomas) setup.
     * @param sizeOfBX3 A 3-column matrix providing scratch space for [Solution, SuperPrime, RHSPrime].
     * @param hand3 Pointer to array of Handles for multi-stream execution.
     * @param delta Grid spacing (dx, dy, dz).
     * @param event Event for stream synchronization.
     */
    EigenDecompThomas(Mat<T> &rowsXRowsP1, Mat<T> &colsXColsP1, Mat<T> &depthsXDepthsP1, Mat<T> &sizeOfBX3, Handle *hand3, const Real3d &delta, Event *event);

    /**
     * @brief Constructs the hybrid solver and manages its own internal memory.
     * * @param dim Grid dimensions.
     * @param hand3 Pointer to array of Handles.
     * @param delta Grid spacing.
     * @param event Event for stream synchronization.
     */
    EigenDecompThomas(const GridDim &dim, Handle *hand3, const Real3d &delta, Event *event);

};

#endif //CUDABANDED_EIGENDECOMPTHOMAS_CUH