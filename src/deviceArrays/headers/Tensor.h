/**
 * @file Tensor.h
 * @brief Defines the Tensor class for 3D GPU-stored arrays.
 *
 * This class represents a 3-dimensional tensor (rows x cols x layers) and provides
 * methods to access layers, columns, and individual elements. It inherits from Mat<T>
 * and supports GPU storage and operations.
 *
 * Memory layout has each layer above subsequent layers.  So if this were a matrix, then the first column would include
 * the first columns of each layer.
 */

#ifndef BICGSTAB_TENSOR_H
#define BICGSTAB_TENSOR_H

#include "../defFiles/DeviceData.cu"
#include "Mat.h"


/**
 * @class Tensor
 * @brief Represents a 3D tensor stored on GPU.
 *
 * Tensor is a final class that inherits from Mat<T> and provides additional
 * access methods for 3D data. It supports row/column/depth access and allows
 * retrieving individual elements as Singleton<T>.
 *
 *Warning: many methods are not yet implemented.
 *
 * Memory layout is that each layer is beneath the layer above it when visualized in two dimensions,
 * so ld >= rows*layers, and to map from flatIndex -> (row, col, layer) we have
 * idx -> (idx % height, idx / (height * depth), (idx % (height * depth)) / height)
 *
 * @tparam T The data type of tensor elements (e.g., float, double, int32_t).
 */
template <typename T>
class Tensor final : public GpuArray<T> {
    friend Vec<T>;
    using GpuArray<T>::kernelPrep;
    using GpuArray<T>::col;
    using GpuArray<T>::row;
private:

    /**
     * @brief Private constructor for internal use.
     *
     * Constructs a Tensor with the given dimensions and shared pointer to data.
     * Users should use the static create() method instead.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param layers Number of layers (depth).
     * @param ld Leading dimension of the underlying storage.
     * @param _ptr Shared pointer to the underlying GPU memory.
     */
    Tensor(size_t rows, size_t cols, size_t layers, size_t ld, std::shared_ptr<T> _ptr);

public:
    /**
     * This matrix is the tensor laid out with one layer above another.
     */
    Mat<T> utilityMatrix;

    const size_t _layers;

    /**
     * This method is a bit dangerous to call without understanding data layout, so bet to see if one of the other layer
     * methods meets your needs first.
     * @return A matrix extracted from this tensor.
     */
    Mat<T> subMatrix(size_t startRow, size_t startCol, size_t startLayer, size_t height, size_t width, size_t ld);

    /**
     * @brief Factory method to create a Tensor of given dimensions.
     *
     * Allocates GPU memory for a tensor of size rows x cols x layers.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param layers Number of layers (depth).
     * @param stream Optional CUDA stream for GPU operations.
     * @return A new Tensor<T> object.
     */
    static Tensor<T> create(size_t rows, size_t cols, size_t layers, cudaStream_t stream);

    /**
     * @brief Returns a specific layer of the tensor as a Mat<T>.
     *
     * @param deptIndex Layer index (0-based).
     * @return Mat<T> representing the requested layer.
     */
    Mat<T> layerRowCol(size_t deptIndex);

    /**
     * @brief Returns a specific layer of the tensor as a Mat<T>.  This layer is perpendicular
     * to the standard layer, and is the set cross product of a column and a depth.
     *
     * For indexing, the column index is the layer and the row index is the row index.
     *
     * @param colIndex The index of the column.(0-based).
     * @return Mat<T> representing the requested layer.
     */
    Mat<T> layerColDepth(size_t colIndex);

    /**
     * @brief Returns a column-depth vector at the given row and column.
     *
     * @param row Row index (0-based).
     * @param col Column index (0-based).
     * @return Vec<T> representing the depth vector at the specified position.
     */
    Vec<T> depth(size_t row, size_t col);
    /**
     * @brief Returns a column-depth vector at the given row and column.
     *
     * @param row Row index (0-based).
     * @param col Column index (0-based).
     * @return Vec<T> representing the depth vector at the specified position.
     */
    Vec<T> row(size_t row, size_t layer);
    /**
     * @brief Returns a column-depth vector at the given row and column.
     *
     * @param row Row index (0-based).
     * @param col Column index (0-based).
     * @return Vec<T> representing the depth vector at the specified position.
     */
    Vec<T> col(size_t col, size_t layer);

    /**
     * @brief Returns a single element of the tensor as a Singleton<T>.
     *
     * @param row Row index (0-based).
     * @param col Column index (0-based).
     * @param layer Layer index (0-based).
     * @return Singleton<T> representing the single element at the specified location.
     */
    Singleton<T> get(size_t row, size_t col, size_t layer);

    /**
     * @return The size of a layer
     */
    [[nodiscard]] size_t layerSize() const;

    DeviceData3d<T> toKernel3d() const;

    /**
     * The data for this destined for a kernel.
     * @return
     */
    DeviceData3d<T> toKernel3d();

    operator DeviceData3d<T>();
    operator DeviceData3d<T>() const;

    KernelPrep kernelPrep() const;

    KernelPrep kernelPrep();

    /**
     *
     * @return The size of this entire tensor.
     */
    [[nodiscard]] size_t size() const override;

    void set(const T *hostData, cudaStream_t stream) override;

    void get(T *hostData, cudaStream_t stream) const override;

    void set(const GpuArray<T> &src, cudaStream_t stream) override;

    void get(GpuArray<T> &dst, cudaStream_t stream) const override;

    void set(std::istream &input_stream, bool isText, bool isColMjr, Handle *hand) override;

    std::ostream &get(std::ostream &output_stream, bool isText, bool printColMajor, Handle &hand) const override;
};

#endif //BICGSTAB_TENSOR_H
