//
// Created by usr on 2/19/26.
//

#ifndef CUDABANDED_EIGENDECOMP3D_CUH
#define CUDABANDED_EIGENDECOMP3D_CUH
#include "EigenDecompSolver.h"


template<typename T>
class EigenDecomp3d: public EigenDecompSolver<T> {
protected:

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
     * @param transposeOperand Swap roles of left/right inputs in cuBLAS.  Set to true if the verctors in a1 need to be
     * transposed.  Otherwise, set to false.
     * @param operand1 Input matrix batch.
     * @param dst1 Output matrix batch.
     * @param stride Matrix stride.
     * @param hand CUDA handle.
     * @param batchCount Number of batches.
     */
    void multE(size_t i, bool transposeEigen, bool transposeOperand, const Mat<T> &operand1, Mat<T> &dst1, size_t stride, Handle &hand, size_t batchCount) const;

    /** @brief Apply E_x or E_xᵀ across all z-layers. */
    void multEX(const Mat<T> &src1, Mat<T> dst1, Handle &hand, bool transposeE) const;

    /** @brief Apply E_y or E_yᵀ across all z-layers. */
    void multEY(const Mat<T> &src1, Mat<T> dst1, Handle &hand, bool transposeE) const;

    /** @brief Apply E_z or E_zᵀ across all x-y slices. */
    void multEZ(const Mat<T> &src1, Mat<T> dst1, Handle &hand, bool transposeE) const;

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
    virtual void multiplyEF(Handle &hand, const Tensor<T> &src, const Tensor<T> &dst, bool transposeE) const;



public:


    /**
     * @brief Creates an eigen decomposition solver for a 3D staggered MAC grid.
     * @param eigen The eigens.
     * @param sizeOfB Workspace vector. Must be the same size as the Eulerian Pressure grid (the system RHS).
     */
    EigenDecomp3d(const LaplacianEigen<T> &eigen, SimpleArray<T> sizeOfB);

    /**
     * Creates an eigen deocmposoiton solver for a laplacian built from a 3d grid.
     * @param boundary The boundary conditions.
     * @param hand3 3 contexts for parallel streaming
     * @param event2 an event for controlling stream dependency.
     */
    EigenDecomp3d(BoundaryConfig<T> boundary, Handle *hand3, Event *event2);




    /**
     * Creates an eigen deocmposoiton solver for a laplacian built from a 3d grid.
     * @param boundary The boundary conditions.
     * @param hand3 3 contexts for parallel streaming.
     * @param sizeOfB A scratch space the size of the RHS.  This will be overwritten.
     * @param event2 an event for controlling stream dependency.
     */
    EigenDecomp3d(BoundaryConfig<T> boundary, Handle *hand3, Event *event2, SimpleArray<T> sizeOfB);

    /**
     * Solves the system.
     * @param x The solution will be placed here.
     * @param b The RHS should be here.
     * @param hand The context.
     */
    void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const;

};


#endif //CUDABANDED_EIGENDECOMP3D_CUH