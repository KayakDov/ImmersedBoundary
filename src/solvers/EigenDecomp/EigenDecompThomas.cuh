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
protected:
    /** @brief GPU workspace for the Thomas solver's modified super-diagonal coefficients. */
    Tensor<T> workSpaceSuperPrime;

    /** @brief GPU workspace for the Thomas solver's modified intermediate RHS values. */
    Tensor<T> workSpaceRHSPrime;

    /** @brief Grid spacing in the Z-direction, used to build the tridiagonal coefficients. */
    double deltaX;

    void multiplyEF(Handle &hand, const Tensor<T> &src, const Tensor<T> &dst, bool transposeE) const override;
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

public:
    /**
     * @brief Constructs the hybrid solver using existing matrix workspaces.
     * @param eigen The eigen vectors and values.
     * @param deltaX The distance between x grid points.
     * @param sizeOfBX3 A 3-column matrix providing scratch space for [Solution, SuperPrime, RHSPrime].
     */
    EigenDecompThomas(const LaplacianEigen<T> &eigen, double deltaX, Mat<T> &sizeOfBX3);

    /**
     * @brief Constructs the hybrid solver and manages its own internal memory.
     * @param boundary The boundary conditions.
     * @param deltaX The distance between x grid points.
     * @param hand3 Pointer to array of Handles.
     * @param event2 Event for stream synchronization.
     * @param sizeOfBX3 allocated memory for the right hand side, and the thomas calculations.  It should have as amny rows
     * as there are rows in L, and 3 columns.
     */
    EigenDecompThomas(const BoundaryConfig<T>& boundary, double deltaX, Handle *hand3, Event *event2, Mat<T> sizeOfBX3);

    /**
     * @brief Constructs the hybrid solver and manages its own internal memory.
     * @param boundary The boundary conditions.
     * @param deltaX The distance between x grid points.
     * @param hand3 Pointer to array of Handles.
     * @param event2 Event for stream synchronization.
     */
    EigenDecompThomas(const BoundaryConfig<T>& boundary, double deltaX, Handle *hand3, Event *event2);

};

#endif //CUDABANDED_EIGENDECOMPTHOMAS_CUH