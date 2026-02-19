

#ifndef CUDABANDED_EIGENDECOMP2D_H
#define CUDABANDED_EIGENDECOMP2D_H
#include "EigenDecompSolver.h"


template<typename T>
class EigenDecomp2d: public EigenDecompSolver<T> {
private:
    /**
     * @brief Compute ũ = f̃ / (λ_x + λ_y + λ_z).
     *
     * @param f  Input in eigen-space.
     * @param u  Output solution in eigen-space.
     * @param hand CUDA cuBLAS/cusolver handle.
     */
    void setUTilde(const Mat<T> &f, Mat<T> &u, Handle &hand) const;

    /**
     * Sets the eigen values.
     * @param hand2 Contexts for mutli threeading.
     * @param delta The distance between the gris points.
     * @param event An event to control stream flow.
     */
    void setEigens(Handle *hand2, Real2d delta, Event &event);

public:
    /**
     * @brief Creates an eigen decomposition solver for a 2D staggered MAC grid.
     * * This solver uses discrete sine/cosine transforms to invert the Laplacian.
     * * @param rowsXRowsP1 Matrix workspace for row-wise operations. Dimensions: [rows x rows + 1].
     * @param colsXColsP1 Matrix workspace for column-wise operations. Dimensions: [cols x cols + 1].
     * @note Optimization: If the grid is square (rows == cols), pass the same matrix
     * reference used for rowsXRows to reduce memory footprint.
     * @param sizeOfB Workspace vector. Must be the same size as the Eulerian Pressure grid (the system RHS).
     * @param hand2 Pointer to an array of at least two Handles for stream management.
     * @param delta The grid spacing (dx, dy).
     * @param event Event object used to synchronize and control multistreaming execution.
     */
    EigenDecomp2d(SquareMat<T> &rowsXRowsP1, SquareMat<T> &colsXColsP1, SimpleArray<T> &sizeOfB, Handle *hand2, Real2d delta, Event &event);

    /**
     * Creates a solver that owns its own memory.
     * @param dim The dimensions of the grid.
     * @param hand2 2 contexts.
     * @param delta The distance between grid points.
     * @param event
     */
    EigenDecomp2d(GridDim dim, Handle *hand2, Real2d delta, Event &event);


    void solve(SimpleArray<T> &x, const SimpleArray<T> &b, Handle &hand) const;
};

#endif //CUDABANDED_EIGENDECOMP2D_H