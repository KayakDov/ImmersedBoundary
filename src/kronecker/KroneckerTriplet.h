#ifndef CUDABANDED_KRONECKERTRIPLET_H
#define CUDABANDED_KRONECKERTRIPLET_H
#include "deviceArrays/headers/SquareMat.h"

/**
 * @class KroneckerTriplet
 * @brief Computes operations involving the Kronecker product of three matrices.
 *
 * This class facilitates operations representing X ⊗ Z ⊗ Y, specifically designed
 * for 3D spectral decompositions where the matrices correspond to 1D eigenbases.
 *
 * Memory layout assumption: Flattened indices represent a 3D grid where the
 * row (Y) changes fastest, then the layer (Z), and the column (X) changes slowest.
 *
 * @tparam T Floating-point type (e.g., float, double).
 */
template <typename T>
class KroneckerTriplet : public XYZ<SquareMat<T>> {

    const GridDim dim;

public:
    /**
     * @brief Applies the X-dimension matrix to the rows of the tensor.
     * @param other Input tensor.
     * @param result Output tensor.
     * @param transposeThis If true, applies the transpose of the X matrix.
     * @param hand CUDA handle.
     */
    void multRows(const SimpleArray<T> &other, SimpleArray<T> result, bool transposeThis, Handle &hand)  const;

    /**
     * @brief Applies the Y-dimension matrix to the columns of the tensor.
     * @param other Input tensor.
     * @param result Output tensor.
     * @param transposeThis If true, applies the transpose of the Y matrix.
     * @param hand CUDA handle.
     */
    void multCols(const SimpleArray<T> &other, SimpleArray<T> result, bool transposeThis, Handle &hand) const;

    /**
     * @brief Applies the Z-dimension matrix to the layers of the tensor.
     * @param other Input tensor.
     * @param result Output tensor.
     * @param transposeThis If true, applies the transpose of the Z matrix.
     * @param hand CUDA handle.
     */
    void multDepths(const SimpleArray<T> &other, SimpleArray<T> result, bool transposeThis, Handle &hand) const;

    /**
     * @brief Constructs a KroneckerTriplet from an XYZ structure of constituent matrices.
     *
     * @param mat An XYZ structure containing the matrices.
     * mat.x multiplies the X dimension (slowest).
     * mat.y multiplies the Y dimension (fastest).
     * mat.z multiplies the Z dimension (middle).
     */
    KroneckerTriplet(const XYZ<SquareMat<T>> &mat);

    /**
     * @brief Constructs a KroneckerTriplet from three individual constituent matrices.
     *
     * @param x Operator for the slowest-changing dimension (X).
     * @param y Operator for the fastest-changing dimension (Y).
     * @param z Operator for the middle-changing dimension (Z).
     */
    KroneckerTriplet(SquareMat<T> x, SquareMat<T> y, SquareMat<T> z);

    /**
     * @brief Explicitly forms the full dense matrix result of the Kronecker triplet product.
     *
     * Computes `result = mat.x ⊗ mat.z ⊗ mat.y`.
     *
     * @param result A pre-allocated matrix to store the final Kronecker product.
     * @param xDimMultZDimBuffer A pre-allocated intermediate buffer to store `mat.x ⊗ mat.z`.
     * @param hand CUDA handle.
     */
    void product(Mat<T> &result, Mat<T> &xDimMultZDimBuffer, Handle &hand) const;

    /**
     * @brief Explicitly forms the full dense matrix result, allocating necessary memory.
     *
     * @param hand CUDA handle.
     * @return The fully evaluated dense Kronecker product matrix.
     */
    Mat<T> product(Handle& hand) const;

    /**
     * @brief Multiplies the Kronecker triplet implicitly with a 1D vector (flattened array).
     *
     * This convenience method allocates its own temporary scratch buffer internally.
     *
     * @param other The input 3D data flattened into a 1D array.
     * @param result The output array.
     * @param transposeThis If true, applies the transpose of the entire triplet operator.
     * @param hand CUDA handle.
     */
    void mult(const SimpleArray<T> &other, SimpleArray<T> &result, bool transposeThis, Handle &hand) const;

    /**
     * @brief Multiplies the Kronecker triplet implicitly with a 3D vector (flattened array).
     *
     * Conceptually computes `result = (mat.x ⊗ mat.z ⊗ mat.y) * other` without forming
     * the full operator matrix, exploiting the tensor product structure for efficiency.
     *
     * @param other The input 3D data flattened into a 1D array.
     * @param result The output array.
     * @param transposeThis If true, applies the transpose of the entire triplet operator.
     * @param resultSizeBuffer An intermediate scratch buffer equal in size to `result`.
     * @param hand CUDA handle.
     */
    void mult(const SimpleArray<T> &other, SimpleArray<T> &result, bool transposeThis, const SimpleArray<T> &resultSizeBuffer, Handle &hand) const;

    /**
     * @brief Multiplies the Kronecker triplet implicitly with a batch of vectors (a matrix).
     *
     * This convenience method allocates its own temporary column scratch buffer internally.
     *
     * @param other The input matrix where each column is a flattened 3D array.
     * @param result The output matrix.
     * @param transposeThis If true, applies the transpose of the entire triplet operator.
     * @param hand CUDA handle.
     */
    void mult(const Mat<T> &other, Mat<T> &result, bool transposeThis, Handle &hand) const;

    /**
     * @brief Multiplies the Kronecker triplet implicitly with a batch of vectors (a matrix).
     *
     * @param other The input matrix where each column is a flattened 3D array.
     * @param result The output matrix.
     * @param transposeThis If true, applies the transpose of the entire triplet operator.
     * @param resultHeightBuffer A scratch buffer with height equal to the output vector length.
     * @param hand CUDA handle.
     */
    void mult(const Mat<T> &other, Mat<T> &result, bool transposeThis, SimpleArray<T> &resultHeightBuffer, Handle &hand) const;

    /**
     * @brief Factory method for a triplet where only the X dimension is active.
     * @param gridDim The dimensions of the 3D grid.
     * @param forRows The operator to apply to the X dimension.
     * @return A KroneckerTriplet with the specified X operator and identities for Y and Z.
     */
    static KroneckerTriplet<T> xOperator(const GridDim &gridDim, const SquareMat<T> &forRows);

    /**
     * @brief Factory method for a triplet where only the Y dimension is active.
     * @param gridDim The dimensions of the 3D grid.
     * @param forCols The operator to apply to the Y dimension.
     * @return A KroneckerTriplet with the specified Y operator and identities for X and Z.
     */
    static KroneckerTriplet<T> yOperator(const GridDim &gridDim, const SquareMat<T> &forCols);

    /**
     * @brief Factory method for a triplet where only the Z dimension is active.
     * @param gridDim The dimensions of the 3D grid.
     * @param forLayers The operator to apply to the Z dimension.
     * @return A KroneckerTriplet with the specified Z operator and identities for X and Y.
     */
    static KroneckerTriplet<T> zOperator(const GridDim &gridDim, const SquareMat<T> &forLayers);
    /**
     * @brief Constructs a 2D Kronecker operator.
     * @param X Operator for the slowest-changing dimension (X).
     * @param Y Operator for the fastest-changing dimension (Y).
     */
    KroneckerTriplet(const SquareMat<T> &X, const SquareMat<T> &Y);

    /**
     * @brief Factory method for a pair where only the X dimension is active.
     */
    static KroneckerTriplet<T> xOperator2d(const GridDim &gridDim, const SquareMat<T> &forRows);

    /**
     * @brief Factory method for a pair where only the Y dimension is active.
     */
    static KroneckerTriplet<T> yOperator2d(const GridDim &gridDim, const SquareMat<T> &forCols);

    /**
     * @brief Overrides the multiplication to optimize for 2D by skipping the Z dimension.
     *
     * @param other Input data.
     * @param result Output data.
     * @param transposeThis Transpose flag.
     * @param resultSizeBuffer Intermediate scratch buffer.
     * @param hand CUDA handle.
     */
    void mult2d(const SimpleArray<T> &other, SimpleArray<T> &result, bool transposeThis, const SimpleArray<T> &resultSizeBuffer, Handle &hand) const;
};

#endif //CUDABANDED_KRONECKERTRIPLET_H