//
// Created by usr on 3/10/26.
//

#ifndef CUDABANDED_EIGENDECOMPFORFORTRAN_H
#define CUDABANDED_EIGENDECOMPFORFORTRAN_H
#include "EigenDecomp2d.h"
#include "EigenDecompThomas.cuh"

/**
 * A wrapper for an eigen decomposition solver of any type with reusable resources for calling the solve method.
 * @tparam Real
 */
template<typename Real>
class EigenDecompForFortran {
    std::unique_ptr<EigenDecompSolver<Real>> eds = nullptr;//This may hold any type of eigen solver, 2d, 3d, or Thomas.
    Handle hand;
    SimpleArray<Real> x, b, adjToB;
public:
    /**
     * Constructs the Eigen decomposition solver.
     * @param rows The number of rows in the grid.
     * @param cols The number of columns in the grid.
     * @param layers The number of layers in the grid.
     * @param dx The distance between grid points in the x direction (columns) in the grid.
     * @param dy The distance between grid points in the y direction (rows) in the grid.
     * @param dz The distance between grid points in the z direction (layers) in the grid.
     * @param thomas True if Thomas algorythm should be used for the z direction, false otherwise.
     * @param sizeOfBForX Allocated gpu space.  It should be at least rows * cols * layers number of elements.  It will be overwritten.
     * @param sizeOfBForRHS Allocated gpu space.  It should be at least rows * cols * layers number of elements.  It will be overwritten.
     * @param sizeOfBForBAdj Allocated gpu space.  It should be at least rows * cols * layers number of elements.  It will be overwritten.
     *
     */
    EigenDecompForFortran(size_t rows, size_t cols, size_t layers, double dx, double dy, double dz, bool leftIsNeumann,
                          bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool backIsNeumann,
                          bool frontIsNeumann, Real leftVal, Real rightVal, Real topVal, Real bottomVal, Real frontVal,
                          Real backVal, bool isStaggered, bool thomas, SimpleArray<Real> sizeOfBForX, SimpleArray<Real> sizeOfBForRHS,
                          SimpleArray<Real> sizeOfBForBAdj);
    /**
     * Solves the equation L x = b.
     * @param xHost The solution overwrites this array.
     * @param bHost The rhs of the equation is input here.
     */
    void solve(Real* xHost, Real* bHost);
};

#endif //CUDABANDED_EIGENDECOMPFORFORTRAN_H
