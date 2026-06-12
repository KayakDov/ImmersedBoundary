

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
     */
    template<typename BoundaryConfigT>
    EigenDecomp2d(const BoundaryConfigT &boundary, Handle *hand2, Event &event);


    /**
     * Be sure that b is in the column space of L.  Otheriwise you will receive a projection onto the column space
     * that will not actually solve Lx = b.
     * @param x The solution will be put here.
     * @param b The rhs of the equation Lx = b.
     * @param hand
     */
    void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const;
};

#endif //CUDABANDED_EIGENDECOMP2D_H