//
// Created by usr on 2/19/26.
//

#ifndef CUDABANDED_EIGENDECOMP3D_CUH
#define CUDABANDED_EIGENDECOMP3D_CUH
#include "EigenDecompSolver.h"
#include "Tensor.h"


template<typename T>
class EigenDecomp3d: public EigenDecompSolver<T> {


    /**
     * @brief Compute ũ = f̃ / (λ_x + λ_y + λ_z).
     *
     * @param f  Input in eigen-space.
     * @param u  Output solution in eigen-space.
     * @param hand CUDA cuBLAS/cusolver handle.
     */
    virtual void setUTilde(const Tensor<T> &f, Tensor<T> &u, Handle &hand) const;

    /**
     * @brief Multiply using E_i or E_iᵀ batched across layers.
     *
     * @param i Which eigenbasis to use (0=x,1=y,2=z).
     * @param transposeEigen Use E_iᵀ instead of E_i.
     * @param transpose Swap roles of left/right inputs in cuBLAS.  Set to true if the verctors in a1 need to be
     * transposed.  Otherwise, set to false.
     * @param a1 Input matrix batch.
     * @param dst1 Output matrix batch.
     * @param stride Matrix stride.
     * @param hand CUDA handle.
     * @param batchCount Number of batches.
     */
    void multE(size_t i, bool transposeEigen, bool transpose,
               const Mat<T> &a1, Mat<T> &dst1, size_t stride,
               Handle &hand, size_t batchCount) const;

    /** @brief Apply E_x or E_xᵀ across all z-layers. */
    void multEX(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const;

    /** @brief Apply E_y or E_yᵀ across all z-layers. */
    void multEY(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const;

    /** @brief Apply E_z or E_zᵀ across all x-y slices. */
    virtual void multEZ(const Mat<T> &src, Mat<T> &dst, Handle &hand, bool transposeE) const;

    /**
     * @brief Apply full transform:
     *        f → E_zᵀ E_yᵀ E_xᵀ f    (forward)
     *        or
     *        u ← E_x E_y E_z ũ      (inverse)
     *
     * @param hand CUDA handle.
     * @param src Input 3D tensor.   Will be overwritten.
     * @param dst Output 3D tensor.
     * @param transposeE Whether to apply Eᵀ instead of E.
     */
    void multiplyEF(Handle &hand, const Tensor<T> &src, Tensor<T> &dst, bool transposeE) const;

    /**
     * Sets the eigen vectors and values.
     * @param hand3 Handles used to make the settings.
     * @param delta The distance between the grid points.
     * @param event3 3 Events to control the stream flow.
     */
    void setEigens(Handle *hand3, Real3d delta, Event *event3);

public:


    /**
     * @brief Creates an eigen decomposition solver for a 3D staggered MAC grid.
     * * @param rowsXRowsP1 Matrix workspace for X-direction operations. Dimensions: [rows x (rows + 1)].
     * @param colsXColsP1 Matrix workspace for Y-direction operations. Dimensions: [cols x (cols + 1)].
     * @note Optimization: If rows == cols, pass the same matrix as rowsXRows.
     * @param depthsXDepthsP1 Matrix workspace for Z-direction operations. Dimensions: [layers x (layers + 1)].
     * @note Optimization: If layers == rows or layers == cols, you may pass the corresponding
     * matrix used for those dimensions.
     * @param sizeOfB Workspace vector. Must be the same size as the Eulerian Pressure grid (the system RHS).
     * @param hand3 Pointer to an array of at least three Handles for concurrent 3D stream processing.
     * @param delta The grid spacing (dx, dy, dz).
     * @param event Pointer to an Event object or array used for multistream synchronization.
     */
    EigenDecomp3d(Mat<T> &rowsXRowsP1, Mat<T> &colsXColsP1, Mat<T> &depthsXDepthsP1, SimpleArray<T> sizeOfB, Handle *hand3, Real3d delta, Event *event);

    /**
     * Creates an eigen deocmposoiton solver for a laplacian built from a 3d grid.
     * @param dim The dimensions of the solver.
     * @param hand3 3 contexts for parallel streaming.
     * @param delta The distance between grid points.
     * @param event an event for controlling stream dependency.
     */
    EigenDecomp3d(const GridDim& dim, Handle *hand3, const Real3d& delta, Event *event);

    /**
     * Creates an eigen deocmposoiton solver for a laplacian built from a 3d grid.
     * @param dim The dimensions of the solver.
     * @param hand3 3 contexts for parallel streaming.
     * @param delta The distance between grid points.
     * @param sizeOfB A scratch space the size of the RHS.  This will be overwritten.
     * @param event an event for controlling stream dependency.
     */
    EigenDecomp3d(const GridDim& dim, Handle *hand3, const Real3d& delta, SimpleArray<T> sizeOfB, Event *event);

    /**
     * Solves the system.
     * @param x The solution will be placed here.
     * @param b The RHS should be here.
     * @param hand The context.
     */
    void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const;

};


#endif //CUDABANDED_EIGENDECOMP3D_CUH