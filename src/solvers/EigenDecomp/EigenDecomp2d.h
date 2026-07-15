/**
 * @file EigenDecomp2d.h
 * @brief Declares the public API for EigenDecomp2d.h.
 * @ingroup eigen_decomp
 *
 * @details
 * Solver interfaces consume the GPU container layer and CUDA library handles to implement direct, iterative, and eigen-decomposition based algorithms.
 */



#ifndef CUDABANDED_EIGENDECOMP2D_H
#define CUDABANDED_EIGENDECOMP2D_H
#include "EigenDecompSolver.h"


template<typename T>
class EigenDecomp2d: public EigenDecompSolver<T> {
private:
    /**
     * @brief Compute ũ = f̃ / (λ_x + λ_y + λ_z).
     *
     * @param src  Input in eigen-space.
     * @param dst  Output solution in eigen-space.
     * @param hand CUDA cuBLAS/cusolver handle.
     */
    void eValsLInvMult(const SimpleArray<T> &src, SimpleArray<T> &dst, Handle &hand) const;

public:

    /**
     * Creates a solver that owns its own memory.
     * @param boundary The boundary conditions.
     * @param hand2 2 contexts.
     * @param event an empty event.
     * @param helmholtzShift set to a non zero value to solve (L - sigma I)x =  b where the helmholtShift is sigma.
     *
     */
    template<class BoundaryConfigT>
    EigenDecomp2d(const BoundaryConfigT &boundary, Handle *hand2, Event &event, T helmholtzShift = 0);


    /**
     * Be sure that b is in the column space of L.  Otheriwise you will receive a projection onto the column space
     * that will not actually solve Lx = b.
     * @param x The solution will be put here.
     * @param b The rhs of the equation Lx = b.
     * @param hand
     */
    void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const override;
};

#endif //CUDABANDED_EIGENDECOMP2D_H