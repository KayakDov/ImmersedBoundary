#include <chrono>

#include "Poisson/EigenDecompSolver.h"
#include "BiCGSTAB/BiCGSTAB.cuh"

/**
 * @brief Construct and immediately solve the Poisson problem.
 *
 * The constructor:
 *   1. Builds eigenbases for Lx, Ly, Lz.
 *   2. Applies forward transform to f to obtain f̃.
 *   3. Solves diagonal system to obtain ũ.
 *   4. Applies inverse transform to obtain x (the output).
 *
 * Orientation of the boundaries is as follows.  The front and back boundaries have there first row agains the top,
 * and first column against the left.  The left and right boundaries each have their first row against the top, and
 * first column against the back.  The top and bottom boundaries each have their first row against the back
 * and first column against the left.
 *
 * Each pair of matrices that are stored together have the 2nd matrix of the pair stored beneath the first,
 * so when thought of as a single matrix with two submatrices, the first half of each column belongs to the first
 * sub matrix, and the second half of each column belongs to the second sub matrix.
 *
 * This constructor is meant to be run as a fortran method.
 *
 * @param frontBack A pointer to the device front and back boundaries.  The back boundary matrix should be below the
 * front boundary matrix.  This will not be changed.
 * @param fbLd The leading dimension of the frontBack matrix.  The distance between the first element of each column.
 * @param leftRight This will not be changed.
 * @param lrLd
 * @param topBottom this will not be changed.
 * @param tbLd
 * @param x Output buffer for the solution. No padding is permitted.
 * @param xStride The distance between elements of the output data.
 * @param height Height of the grid.
 * @param width Width of the grid.
 * @param depth Depth of the grid.
 * @param f Right-hand-side of the Poisson equation (will be overwritten).  No padding is permitted.
 * @param fStride The distance between elements of the f vector.
 * @param rowsXRows A space to work in.  Will be changed.
 * @param rowsXRowsLd
 * @param colsXCols A space to work in.  Will be changed.
 * @param colsXColsLd
 * @param depthsXDepths A space to work in.  Will be changed.
 * @param depthsXDepthsLd
 * @param maxDimX3 A space to work in.  Will be changed.
 * @param maxDimX3Ld
 */
template<typename T>
void solveDecomp(
    T* frontBack, const size_t fbLd,
    T* leftRight, const size_t lrLd,
    T* topBottom, const size_t tbLd,
    T* f, const size_t fStride,
    T* x, const size_t xStride,
    const size_t height, const size_t width, const size_t depth,
    T* rowsXRows, const size_t rowsXRowsLd,
    T* colsXCols, const size_t colsXColsLd,
    T* depthsXDepths, const size_t depthsXDepthsLd,
    T *maxDimX3, const size_t maxDimX3Ld
) {
    const size_t n = height * width * depth;
    auto frontBackMat = Mat<T>::create(2*height, width, fbLd, frontBack);
    auto leftRightMat = Mat<T>::create(2*height, depth, lrLd, leftRight);
    auto topBottomMat = Mat<T>::create(2*depth, width, tbLd, topBottom);
    const CubeBoundary cb(frontBackMat, leftRightMat, topBottomMat);
    std::array<Handle, 3> hands{};
    auto xVec = Vec<T>::create(n, xStride, x);
    auto fVec = Vec<T>::create(n, fStride, f);
    auto xMat = SquareMat<T>::create(width, colsXColsLd, colsXCols);
    auto yMat = SquareMat<T>::create(height, rowsXRowsLd, rowsXRows);
    auto zMat = SquareMat<T>::create(depth, depthsXDepthsLd, depthsXDepths);
    auto maxDimX3Mat = Mat<T>::create(n, 3, depthsXDepthsLd, maxDimX3);

    cudaDeviceSynchronize();
    EigenDecompSolver(cb, xVec, fVec, yMat, xMat, zMat, maxDimX3Mat, hands);
    cudaDeviceSynchronize();
}


/**
 * Solves Ax = b
 * @param A The banded matrix.  Each column represents a diagonal of a sparse matrix.  Shorter diagonals will have
 * trailing padding, but never leading padding.  There should be as many columns as there are diagonals in the
 * square sparse matrix, and as many rows as there are rows in the square sparse matrix.  This matrix will not be changed.
 * @param aLd The leading dimension of A.  It is the distance between the first elements of each column.  Must be
 * at least the number of rows in A, but may be more if there's padding.
 * @param inds The ith element is the diagonal index of the ith column in A.  Super diagonals have positive indices,
 * and subdiagonals have negative indices.  The absolute value of the index is the distance of the diagonal from the
 * primary diagonal.  This vector will not be changed.
 * @param indsStride  The distance between elements of inds.  This is usually 1.
 * @param numInds The number of diagonals.
 * @param b The RHS of Ax=b.  This vector will be overwritten with the solution, x.
 * @param bStride The distance between elements of b.
 * @param bSize The number of elements in b, x, and the number of rows in A.
 * @param prealocatedSizeX7 should have bSize rows and 7 columns.  Will be overwritten.
 * @param prealocatedLd The distance between the first elements of each column of prealocatedSizeX7.
 * @param maxIterations The maximum number of iterations.
 * @param tolerance What's close enough to 0.
 */
template<typename T>
void solveBiCGSTAB(
    T* A,
    const size_t aLd,
    int32_t* inds,
    const size_t indsStride,
    const size_t numInds,
    T* b,
    const size_t bStride,
    const size_t bSize,
    T* prealocatedSizeX7, //size x 7 matrix
    const size_t prealocatedLd,
    size_t maxIterations,
    T tolerance
) {
    Mat<T> preAlocatedMat = Mat<T>::create(bSize, 7, prealocatedLd, prealocatedSizeX7);
    Vec<T> bVec = Vec<T>::create(bSize, bStride, b);
    const auto ABanded = BandedMat<T>::create(bSize, numInds, aLd, A, inds, indsStride); //rows cols pointer VecIndices
    BiCGSTAB<T>::solve(ABanded,bVec, &preAlocatedMat, tolerance, maxIterations);
}
