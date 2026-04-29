#ifndef CUDABANDED_KRONECKERTRIPLET_H
#define CUDABANDED_KRONECKERTRIPLET_H
#include "deviceArrays/headers/Mat.h"

/**
 * @class KroneckerTriplet
 * @brief Computes operations involving the Kronecker product of three matrices.
 *
 * This class facilitates operations representing A ⊗ B ⊗ C, specifically designed
 * for 3D spectral decompositions where the matrices correspond to 1D eigenbases.
 *
 * Memory layout assumption: Flattened indices represent a 3D grid where the
 * row (X) changes fastest, then the layer (Z), and the column (Y) changes slowest.
 *
 * @tparam T Floating-point type (e.g., float, double).
 */
template <typename T>
class KroneckerTriplet {
    /**
     * @brief A structure holding the three matrices (x, y, z) that form the triplet.
     * mat.x applies to the fastest changing dimension (rows),
     * mat.z applies to the middle dimension (layers),
     * mat.y applies to the slowest changing dimension (columns).
     */
    const XYZ<Mat<T>>& mat;

    /**
     * @brief Wrapper for batched cuBLAS matrix multiplication.
     *
     * Computes combinations of (kMat * operand) or (operand * kMat), with optional
     * transpositions on the Kronecker matrix kMat.
     *
     * @param kMat The 1D operator matrix for a specific dimension.
     * @param transposeThis If true, uses kMatᵀ instead of kMat.
     * @param transposeOperand If true, computes (operand * kMat). If false, computes (kMat * operand).
     * @param operand1 The input tensor/matrix data.
     * @param dst1 The output tensor/matrix data.
     * @param stride Memory stride between batches.
     * @param hand The CUDA handle for asynchronous execution.
     * @param batchCount The number of matrices in the batch.
     */
    void mult1D(Mat<T> kMat, bool transposeThis, bool transposeOperand, const Mat<T> &operand1, Mat<T> &dst1,
                size_t stride, Handle &hand, size_t batchCount) const;

    /**
     * @brief Applies the X-dimension matrix to the rows of the tensor.
     * @param other Input tensor.
     * @param result Output tensor.
     * @param transposeThis If true, applies the transpose of the X matrix.
     * @param hand CUDA handle.
     */
    void multRows(const Mat<T> &other, Mat<T> result, bool transposeThis, Handle &hand);

    /**
     * @brief Applies the Y-dimension matrix to the columns of the tensor.
     * @param other Input tensor.
     * @param result Output tensor.
     * @param transposeThis If true, applies the transpose of the Y matrix.
     * @param hand CUDA handle.
     */
    void multCols(const Tensor<T> &other, Tensor<T> result, bool transposeThis, Handle &hand);

    /**
     * @brief Applies the Z-dimension matrix to the layers of the tensor.
     * @param other Input tensor.
     * @param result Output tensor.
     * @param transposeThis If true, applies the transpose of the Z matrix.
     * @param hand CUDA handle.
     */
    void multDepths(const Tensor<T> &other, Tensor<T> result, bool transposeThis, Handle &hand);

public:
    /**
     * @brief Constructs a KroneckerTriplet from three constituent matrices.
     *
     * @param mat An XYZ structure containing the matrices.
     * mat.x multiplies the rows (fastest dimension).
     * mat.y multiplies the columns (slowest dimension).
     * mat.z multiplies the depths/layers (middle dimension).
     * Pass an empty matrix for z to create a 2D Kronecker pair.
     */
    KroneckerTriplet(const XYZ<Mat<T>>& mat);

    /**
     * @brief Explicitly forms the full dense matrix result of the Kronecker triplet product.
     *
     * Computes `result = mat.y ⊗ mat.z ⊗ mat.x`.
     *
     * @param result A pre-allocated matrix to store the final Kronecker product.
     * @param yDimMultZDimBuffer A pre-allocated intermediate buffer to store `mat.y ⊗ mat.z`.
     * @param hand CUDA handle.
     */
    void product(Mat<T> &result, Mat<T> &yDimMultZDimBuffer, Handle &hand);

    /**
     * @brief Explicitly forms the full dense matrix result, allocating necessary memory.
     *
     * @param hand CUDA handle.
     * @return The fully evaluated dense Kronecker product matrix.
     */
    Mat<T> product(Handle& hand);

    /**
     * @brief Retrieves the logical 3D grid dimensions required for vector multiplication.
     * @return A GridDim object representing (cols, rows, layers) corresponding to the matrices.
     */
    GridDim dim();

    /**
     * @brief Multiplies the Kronecker triplet implicitly with a 3D vector (flattened array).
     *
     * Conceptually computes `result = (mat.y ⊗ mat.z ⊗ mat.x) * other` without forming
     * the full operator matrix, exploiting the tensor product structure for efficiency.
     *
     * @param other The input 3D data flattened into a 1D array.
     * @param result The output array.
     * @param transposeThis If true, applies the transpose of the entire triplet operator.
     * @param resultSizeBuffer An intermediate scratch buffer equal in size to `result`.
     * @param hand CUDA handle.
     */
    void mult(const SimpleArray<T> &other, SimpleArray<T> &result, bool transposeThis, SimpleArray<T> resultSizeBuffer, Handle &hand);

    /**
     * @brief Multiplies the Kronecker triplet implicitly with a batch of vectors (a matrix).
     *
     * @param other The input matrix where each column is a flattened 3D array.
     * @param result The output matrix.
     * @param transposeThis If true, applies the transpose of the entire triplet operator.
     * @param resultHeightBuffer An intermediate scratch buffer equal to the height of `result`.
     * @param hand CUDA handle.
     */
    void mult(Mat<T> other, Mat<T> result, bool transposeThis, SimpleArray<T> resultHeightBuffer, Handle &hand);
};

#endif //CUDABANDED_KRONECKERTRIPLET_H