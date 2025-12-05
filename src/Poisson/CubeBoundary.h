#pragma once

#include "deviceArrays/headers/Mat.h"
#include <cuda_runtime.h>
#include <algorithm>

/**
 * @brief Represents the boundary faces of a 3D cube using three paired matrices.
 * 
 * The six faces of a 3D cube are stored in three Mat<T> objects: 
 * frontBack (Front/Back), leftRight (Left/Right), and topBottom (Top/Bottom). 
 * Each pair stores the second face immediately after the first face in memory.
 *
 * The layout of each matrix is as follows.  The first column of the left and right matrices is at the back.  The first
 * row of the left and right matrices is at the top.  The first column of the top and bottom matrices is at the back.
 * The first column of the top and bottom matrices is adjacent to the left matrix.
 * 
 * @tparam T The data type of the cube elements.
 */
template<typename T>
class CubeBoundary {
public:
    /// @brief Front/Back, Left/Right, and Top/Bottom boundary matrices
    Mat<T> frontBack, leftRight, topBottom;

    /**
     * @brief Constructs a CubeBoundary from given boundary matrices.
     * 
     * @param frontBack Matrix with front (first half rows) and back (second half rows) faces
     * @param leftRight Matrix with left and right faces
     * @param topBottom Matrix with top and bottom faces
     */
    CubeBoundary(Mat<T>& frontBack, Mat<T>& leftRight, Mat<T>& topBottom);

    /**
     * @brief Returns the total number of elements across all boundary matrices.
     * @return Total number of elements
     */
    [[nodiscard]] size_t size() const;

    /**
     * @brief Returns the number of internal elements inside the cube.
     * @return Internal cube size
     */
    [[nodiscard]] size_t internalSize() const;

    /**
     * @brief Frees memory of all Mat<T> views in this boundary.
     */
    void freeMem();

    /**
     * @brief Generates a sample cube boundary with front=1, back=0, and linear transitions.
     * @param dim Cube side length
     * @param stream CUDA stream for memory operations
     * @return CubeBoundary instance
     */
    static CubeBoundary<T> ZeroTo1(size_t dim, cudaStream_t stream);

    /**
     * @brief Generates a sample cube boundary with front=1, back=0, and linear transitions.
     * @param height Cube height
     * @param width Cube width
     * @param depth Cube depth
     * @param stream CUDA stream for memory operations
     * @return CubeBoundary instance
     */
    static CubeBoundary<T> ZeroTo1(size_t height, size_t width, size_t depth, cudaStream_t stream);
};
