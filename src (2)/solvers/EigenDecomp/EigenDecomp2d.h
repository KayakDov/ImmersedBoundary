/**
 * @file EigenDecomp2d.h
 * @brief Eigen-decomposition solver for 2D problems.
 * @ingroup eigen_decomp
 */



#ifndef CUDABANDED_EIGENDECOMP2D_H
#define CUDABANDED_EIGENDECOMP2D_H
#include "EigenDecompSolver.h"


template<typename T>
class EigenDecomp2d: public EigenDecompSolver<T> {
private:
    /**
     * @brief Compute ũ = f̃ / (λ_x + λ_y).
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
     * @param hands At least 2 handles.
     * @param helmholtzShift set to a non zero value to solve (L - sigma I)x =  b where the helmholtzShift is sigma.
     *
     */
    template<class BoundaryConfigT>
    EigenDecomp2d(const BoundaryConfigT &boundary, Handles &hands, T helmholtzShift = 0);


    /**
     * Be sure that b is in the column space of L.  Otherwise you will receive a projection onto the column space
     * that will not actually solve Lx = b.
     * @param x The solution will be put here.
     * @param b The rhs of the equation Lx = b.
     * @param hand
     */
    void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const override;
};

#endif //CUDABANDED_EIGENDECOMP2D_H