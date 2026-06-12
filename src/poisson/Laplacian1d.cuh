
#ifndef CUDABANDED_LAPLACIAN1D_H
#define CUDABANDED_LAPLACIAN1D_H

#include <stddef.h>
#include <cstdint>

// Forward dependencies required for by-value tracking
#include "BoundaryConfig.cuh"
#include "deviceArrays/headers/Support/Streamable.h" // Replace with actual path to SimpleArray/Handle if different
#include "deviceArrays/headers/sparse/BandedMat.h"
#include "deviceArrays/headers/SquareMat.h"
#include "deviceArrays/headers/sparse/TriDiagonal.cuh"

/**
 * The one dimensional Laplacians for a 2 or 3d grid.  Note, if a pair of Laplacians have the same dimension size and
 * boundary conditions, then the pointers may point to the same memory.
 * @tparam T
 */
template<typename T>
class Laplacian1d :public XYZ<TriDiagonal<T>>{

public:

    /**
     *
     * @param boundary The boundary for the 1d laplacians.
     * @param hand The context.
     */
    template<typename BoundaryConfigT>
    Laplacian1d(const BoundaryConfigT &boundary, Handle &hand);

    /**
     * The square matrix of the given dimension.  This method allocated memory.
     * @param dim 0 for x, 1 for y, 2 for z.
     * @param hand
     * @return A square 1d laplacian matrix.
     */
    SquareMat<T> dense(size_t dim, Handle &hand);

    /**
     * The square matrix of the given dimension.  This method allocated memory.
     * @param dim 0 for x, 1 for y, 2 for z.
     * @param squareMatGoesHere The square matrix to be populated.
     * @param hand
     * @return A square 1d laplacian matrix.
     */
    void dense(size_t dim, SquareMat<T>& squareMatGoesHere, Handle &hand);

    /**
     * Creates a 1 dimensional laplacian.
     * @tparam AxisSegmentT Either a UniformSegment or a VariableSegment.
     * @param segment The boundary conditions and spacing of the matrix.
     * @param mat The laplacian will fill this matrix.
     * @param hand
     */
    template<typename  AxisSegmentT>
    static void create(const AxisSegmentT &segment, TriDiagonal<T> mat, Handle &hand);

    /**
     * Creates a 1 dimensional laplacian.
     * This method allocated memory.
     * @tparam AxisSegmentT Either a UniformSegment or a VariableSegment.
     * @param segment The boundary conditions and spacing of the matrix.
     * @param hand
     */
    template<class AxisSegmentT>
    TriDiagonal<T> create(const AxisSegmentT &segment, Handle &hand);
};

#endif //CUDABANDED_LAPLACIAN1D_H
