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
     * @param src  Input in eigen-space.
     * @param dst  Output solution in eigen-space.
     * @param hand CUDA cuBLAS/cusolver handle.
     */
    virtual void multLEigenValInverse(const SimpleArray<T> &src, SimpleArray<T> &dst, Handle &hand) const;



public:


    /**
     * @brief Creates an eigen decomposition solver for a 3D staggered MAC grid.
     * @param eigen The eigens.
     * @param sizeOfB Workspace vector. Must be the same size as the Eulerian Pressure grid (the system RHS).
     * @param isSingular Is the laplacian singular (all boundary conditions are Neumann)
     */
    EigenDecomp3d(const poisson::Eigen<T> &eigen, SimpleArray<T> sizeOfB, bool isSingular);

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
     * Be sure that b is in the column space of L.  Otheriwise you will receive a projection onto the column space
     * that will not actually solve Lx = b.
     * @param x The solution will be put here.
     * @param b The rhs of the equation Lx = b.
     * @param hand
     */
    void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const;

};


#endif //CUDABANDED_EIGENDECOMP3D_CUH