#ifndef CUDABANDED_KRONECKERTRIPLET_H
#define CUDABANDED_KRONECKERTRIPLET_H
#include "deviceArrays/headers/SquareMat.h"

/**
 * @class KroneckerTriplet
 * @brief Computes operations involving the Kronecker product of three matrices.
 *
 * This class facilitates operations representing Y ⊗ Z ⊗ X, specifically designed
 * for 3D spectral decompositions where the matrices correspond to 1D eigenbases.
 *
 * Memory layout assumption: Flattened indices represent a 3D grid where the
 * row (X) changes fastest, then the layer (Z), and the column (Y) changes slowest.
 *
 * @tparam T Floating-point type (e.g., float, double).
 */
template <typename T>
class KroneckerTriplet : public XYZ<Mat<T>> {

    const GridDim dim;

public:
    /**
     * @brief Applies the X-dimension matrix to the rows of the tensor.
     * @param other Input tensor.
     * @param result Output tensor.
     * @param transposeThis If true, applies the transpose of the X matrix.
     * @param hand CUDA handle.
     */
    void multRows(const SimpleArray<T> &other, SimpleArray<T> result, bool transposeThis, Handle &hand);

    /**
     * @brief Applies the Y-dimension matrix to the columns of the tensor.
     * @param other Input tensor.
     * @param result Output tensor.
     * @param transposeThis If true, applies the transpose of the Y matrix.
     * @param hand CUDA handle.
     */
    void multCols(const SimpleArray<T> &other, SimpleArray<T> result, bool transposeThis, Handle &hand);

    /**
     * @brief Applies the Z-dimension matrix to the layers of the tensor.
     * @param other Input tensor.
     * @param result Output tensor.
     * @param transposeThis If true, applies the transpose of the Z matrix.
     * @param hand CUDA handle.
     */
    void multDepths(const SimpleArray<T> &other, SimpleArray<T> result, bool transposeThis, Handle &hand);

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

    KroneckerTriplet(Mat<T> x, Mat<T> y, Mat<T> z);

    /**
     * @brief Explicitly forms the full dense matrix result of the Kronecker triplet product.
     *
     * Computes `result = mat.y ⊗ mat.z ⊗ mat.x`.
     *
     * @param result A pre-allocated matrix to store the final Kronecker product.
     * @param xDimMultZDimBuffer A pre-allocated intermediate buffer to store `mat.y ⊗ mat.z`.
     * @param hand CUDA handle.
     */
    void product(Mat<T> &result, Mat<T> &xDimMultZDimBuffer, Handle &hand);

    /**
     * @brief Explicitly forms the full dense matrix result, allocating necessary memory.
     *
     * @param hand CUDA handle.
     * @return The fully evaluated dense Kronecker product matrix.
     */
    Mat<T> product(Handle& hand);

    void mult(const SimpleArray<T> &other, SimpleArray<T> &result, bool transposeThis, Handle &hand);

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

    void mult(const Mat<T> &other, Mat<T> &result, bool transposeThis, Handle &hand);

    /**
     * @brief Multiplies the Kronecker triplet implicitly with a batch of vectors (a matrix).
     *
     * @param other The input matrix where each column is a flattened 3D array.
     * @param result The output matrix.
     * @param transposeThis If true, applies the transpose of the entire triplet operator.
     * @param hand CUDA handle.
     */
    void mult(const Mat<T> &other, Mat<T> &result, bool transposeThis, SimpleArray<T> &resultHeightBuffer, Handle &hand);

    static KroneckerTriplet<T> xOperator(const GridDim &gridDim, const Mat<T> &forRows);

    static KroneckerTriplet<T> yOperator(const GridDim &gridDim, const Mat<T> &forCols);

    static KroneckerTriplet<T> zOperator(const GridDim &gridDim, const Mat<T> &forLayers);
};

#endif //CUDABANDED_KRONECKERTRIPLET_H