#include <chrono>

#include "solvers/EigenDecompSolver.h"
#include "solvers/BiCGSTAB.cuh"

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
 * @param frontBackPtr A pointer to the device front and back boundaries.  The back boundary matrix should be below the
 * front boundary matrix.  This will not be changed.
 * @param fbLd The leading dimension of the frontBack matrix.  The distance between the first element of each column.
 * @param leftRightPtr This will not be changed.
 * @param lrLd
 * @param topBottomPtr this will not be changed.
 * @param tbLd
 * @param xPtr Output buffer for the solution. No padding is permitted.
 * @param xStride The distance between elements of the output data.
 * @param height Height of the grid.
 * @param width Width of the grid.
 * @param depth Depth of the grid.
 * @param fPtr Right-hand-side of the Poisson equation (will be overwritten).  No padding is permitted.
 * @param fStride The distance between elements of the f vector.
 * @param rowsXRowsPtr A space to work in.  Will be changed.
 * @param rowsXRowsLd
 * @param colsXColsPtr A space to work in.  Will be changed.
 * @param colsXColsLd
 * @param depthsXDepthsPtr A space to work in.  Will be changed.
 * @param depthsXDepthsLd
 * @param maxDimX3Ptr A space to work in.  Will be changed.
 * @param maxDimX3Ld
 */
template<typename T>
void solveDecomp(
    size_t frontBackPtr, const size_t fbLd,
    size_t leftRightPtr, const size_t lrLd,
    size_t topBottomPtr, const size_t tbLd,
    size_t fPtr, const size_t fStride,
    size_t xPtr, const size_t xStride,
    const size_t height, const size_t width, const size_t depth,
    size_t rowsXRowsPtr, const size_t rowsXRowsLd,
    size_t colsXColsPtr, const size_t colsXColsLd,
    size_t depthsXDepthsPtr, const size_t depthsXDepthsLd,
    size_t maxDimX3Ptr, const size_t maxDimX3Ld
) {
    const size_t n = height * width * depth;

    const CubeBoundary cb = CubeBoundary<T>::create(
        reinterpret_cast<T *>(frontBackPtr), fbLd,
        reinterpret_cast<T *>(leftRightPtr), lrLd,
        reinterpret_cast<T *>(topBottomPtr), tbLd,
        height, width, depth
    );
    Handle hands[3];
    auto xVec = Vec<T>::create(n, xStride, reinterpret_cast<T *>(xPtr));
    auto fVec = Vec<T>::create(n, fStride, reinterpret_cast<T *>(fPtr));
    auto xMat = SquareMat<T>::create(width, colsXColsLd, reinterpret_cast<T *>(colsXColsPtr));
    auto yMat = SquareMat<T>::create(height, rowsXRowsLd, reinterpret_cast<T *>(rowsXRowsPtr));
    auto zMat = SquareMat<T>::create(depth, depthsXDepthsLd, reinterpret_cast<T *>(depthsXDepthsPtr));
    auto maxDimX3Mat = Mat<T>::create(n, 3, depthsXDepthsLd, reinterpret_cast<T *>(maxDimX3Ptr));
    auto sizeOfB = SimpleArray<T>::create(fVec.size(), hands[0]);

    cudaDeviceSynchronize();
    PoissonRHS<T> poisson(cb, fVec, hands[2]);
    EigenDecompSolver3d(yMat, xMat, zMat, maxDimX3Mat, sizeOfB, hands).solve(xVec, fVec, hands[0]);
    cudaDeviceSynchronize();
}


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
 * @param frontBackPtr A pointer to the device front and back boundaries.  The back boundary matrix should be below the
 * front boundary matrix.  This will not be changed.
 * @param fbLd The leading dimension of the frontBack matrix.  The distance between the first element of each column.
 * @param leftRightPtr This will not be changed.
 * @param lrLd
 * @param topBottomPtr this will not be changed.
 * @param tbLd
 * @param xPtr Output buffer for the solution. No padding is permitted.
 * @param xStride The distance between elements of the output data.
 * @param height Height of the grid.
 * @param width Width of the grid.
 * @param depth Depth of the grid.
 * @param fPtr Right-hand-side of the Poisson equation (will be overwritten).  No padding is permitted.
 * @param fStride The distance between elements of the f vector.
 * @param rowsXRowsPtr A space to work in.  Will be changed.
 * @param rowsXRowsLd
 * @param colsXColsPtr A space to work in.  Will be changed.
 * @param colsXColsLd
 * @param depthsXDepthsPtr A space to work in.  Will be changed.
 * @param depthsXDepthsLd
 * @param maxDimX3Ptr A space to work in.  Will be changed.
 * @param maxDimX3Ld
 */
void inline solveDecompFloat(
    size_t frontBackPtr, const size_t fbLd,
    size_t leftRightPtr, const size_t lrLd,
    size_t topBottomPtr, const size_t tbLd,
    size_t fPtr, const size_t fStride,
    size_t xPtr, const size_t xStride,
    const size_t height, const size_t width, const size_t depth,
    size_t rowsXRowsPtr, const size_t rowsXRowsLd,
    size_t colsXColsPtr, const size_t colsXColsLd,
    size_t depthsXDepthsPtr, const size_t depthsXDepthsLd,
    size_t maxDimX3Ptr, const size_t maxDimX3Ld
) {
    return solveDecomp<float>(
        frontBackPtr, fbLd,
        leftRightPtr, lrLd,
        topBottomPtr, tbLd,
        fPtr, fStride,
        xPtr, xStride,
        height, width, depth,
        rowsXRowsPtr, rowsXRowsLd,
        colsXColsPtr, colsXColsLd,
        depthsXDepthsPtr, depthsXDepthsLd,
        maxDimX3Ptr, maxDimX3Ld
    );
}

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
 * @param frontBackPtr A pointer to the device front and back boundaries.  The back boundary matrix should be below the
 * front boundary matrix.  This will not be changed.
 * @param fbLd The leading dimension of the frontBack matrix.  The distance between the first element of each column.
 * @param leftRightPtr This will not be changed.
 * @param lrLd
 * @param topBottomPtr this will not be changed.
 * @param tbLd
 * @param xPtr Output buffer for the solution. No padding is permitted.
 * @param xStride The distance between elements of the output data.
 * @param height Height of the grid.
 * @param width Width of the grid.
 * @param depth Depth of the grid.
 * @param fPtr Right-hand-side of the Poisson equation (will be overwritten).  No padding is permitted.
 * @param fStride The distance between elements of the f vector.
 * @param rowsXRowsPtr A space to work in.  Will be changed.
 * @param rowsXRowsLd
 * @param colsXColsPtr A space to work in.  Will be changed.
 * @param colsXColsLd
 * @param depthsXDepthsPtr A space to work in.  Will be changed.
 * @param depthsXDepthsLd
 * @param maxDimX3Ptr A space to work in.  Will be changed.
 * @param maxDimX3Ld
 */
void inline solveDecompDouble(
    size_t frontBackPtr, const size_t fbLd,
    size_t leftRightPtr, const size_t lrLd,
    size_t topBottomPtr, const size_t tbLd,
    size_t fPtr, const size_t fStride,
    size_t xPtr, const size_t xStride,
    const size_t height, const size_t width, const size_t depth,
    size_t rowsXRowsPtr, const size_t rowsXRowsLd,
    size_t colsXColsPtr, const size_t colsXColsLd,
    size_t depthsXDepthsPtr, const size_t depthsXDepthsLd,
    size_t maxDimX3Ptr, const size_t maxDimX3Ld
) {
    return solveDecomp<double>(
        frontBackPtr, fbLd,
        leftRightPtr, lrLd,
        topBottomPtr, tbLd,
        fPtr, fStride,
        xPtr, xStride,
        height, width, depth,
        rowsXRowsPtr, rowsXRowsLd,
        colsXColsPtr, colsXColsLd,
        depthsXDepthsPtr, depthsXDepthsLd,
        maxDimX3Ptr, maxDimX3Ld
    );
}

/**
 * Solves Ax = b
 * @param APtr The banded matrix.  Each column represents a diagonal of a sparse matrix.  Shorter diagonals will have
 * trailing padding, but never leading padding.  There should be as many columns as there are diagonals in the
 * square sparse matrix, and as many rows as there are rows in the square sparse matrix.  This matrix will not be changed.
 * @param aLd The leading dimension of A.  It is the distance between the first elements of each column.  Must be
 * at least the number of rows in A, but may be more if there's padding.
 * @param indsPtrInt32_t The ith element is the diagonal index of the ith column in A.  Super diagonals have positive indices,
 * and subdiagonals have negative indices.  The absolute value of the index is the distance of the diagonal from the
 * primary diagonal.  This vector will not be changed.
 * @param indsStride  The distance between elements of inds.  This is usually 1.
 * @param numInds The number of diagonals.
 * @param bPtr The RHS of Ax=b.  This vector will be overwritten with the solution, x.
 * @param bStride The distance between elements of b.
 * @param bSize The number of elements in b, x, and the number of rows in A.
 * @param prealocatedSizeX7Ptr should have bSize rows and 7 columns.  Will be overwritten.
 * @param prealocatedLd The distance between the first elements of each column of prealocatedSizeX7.
 * @param maxIterations The maximum number of iterations.
 * @param tolerance What's close enough to 0.
 */
template<typename T>
void solveBiCGSTAB(
    size_t APtr,
    const size_t aLd,
    size_t indsPtrInt32_t,
    const size_t indsStride,
    const size_t numInds,
    size_t bPtr,
    const size_t bStride,
    const size_t bSize,
    size_t prealocatedSizeX7Ptr, //size x 7 matrix
    const size_t prealocatedLd,
    size_t maxIterations,
    T tolerance
) {
    Handle hand4[4]{};
    auto heightX7 = Mat<T>::create(bSize, 7, prealocatedLd, reinterpret_cast<T *>(prealocatedSizeX7Ptr));
    auto a9 = Vec<T>::create(9, hand4[0]);
    Vec<T> bVec = Vec<T>::create(bSize, bStride, reinterpret_cast<T *>(bPtr));
    const auto ABanded = BandedMat<T>::create(bSize, numInds, aLd, reinterpret_cast<T *>(APtr),
                                              reinterpret_cast<int32_t *>(indsPtrInt32_t),
                                              indsStride); //rows cols pointer VecIndices
        // Handle hand;
    // std::cout << "ABanded = \n" << GpuOut<T>(ABanded, hand) << std::endl;
    // std::cout << "indices = \n" << GpuOut<int32_t>(ABanded._indices, hand) << std::endl;
    // std::cout << "bVec = \n" << GpuOut<T>(bVec, hand) << std::endl;
    // std::cout << "prealocated = \n" << GpuOut<T>(preAlocatedMat, hand) << std::endl;

    BCGBanded<T>::solve(hand4, ABanded, bVec, bVec, &heightX7, &a9, tolerance, maxIterations);
}
// (
//     Handle* hand4,
//     const BandedMat<T> &A,
//     Vec<T>& result,
//     const Vec<T> &b,
//     Mat<T> *allocatedBHeightX7,
//     Vec<T>* allocated9,
//     const T tolerance,
//     const size_t maxIterations
// )