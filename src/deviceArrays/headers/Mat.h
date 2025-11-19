
#ifndef BICGSTAB_MAT_H
#define BICGSTAB_MAT_H

#include "Singleton.h"
#include "Tensor.h"
#include "GpuArray.h"
#include "Vec.h"

/**
 * @brief Abstract base class for GPU-backed matrices.
 *
 * Inherits from GpuArray<T> and provides additional matrix operations:
 * multiplication, addition/subtraction, transposition, submatrix extraction,
 * column/row access, and normalization. Designed to work with CUDA streams and
 * optional Handle objects.
 *
 * @tparam T Type of elements in the matrix (e.g., float, double, int).
 */
template <typename T>
class Mat : public GpuArray<T> {
    using GpuArray<T>::mult;

    friend Tensor<T>;

protected:
    /**
     * @brief Protected constructor. Use static create() method or derived classes.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param ld Leading dimension of the matrix.
     * @param _ptr Shared pointer to device memory for the matrix.
     */
    Mat(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> _ptr);

public:


    /**
     * @brief Retrieves or creates a target matrix with the specified dimensions.
     *
     * This method either uses an existing `Mat<T>` pointer if provided or creates
     * a new `Mat<T>` instance with the given dimensions. The newly created
     * `Mat<T>` instance is managed through a unique pointer and returned to the caller.
     *
     * @param rows The number of rows for the target matrix.
     * @param cols The number of columns for the target matrix.
     * @param result A pointer to an existing `Mat<T>` instance. If not null, this
     *               instance will be used directly.
     * @param out_ptr_unique A unique pointer to manage the newly created `Mat<T>` instance,
     *                       if no existing matrix (`result`) is provided.
     *
     * @return A pointer to the target `Mat<T>` matrix, either an existing instance
     *         (`result`) or a newly created one.
     */
    static Mat<T> *_get_or_create_target(size_t rows, size_t cols, Mat<T> *result,
                                  std::unique_ptr<Mat<T>> &out_ptr_unique);

    /**
     * @copydoc GpuArray::size
     */
    [[nodiscard]] size_t size() const override;
    /**
     * @copydoc GpuArray::bytes
     */
    [[nodiscard]] size_t bytes() const;
    /**
     * @copydoc GpuArray::set
     */
    void set(const T* src, cudaStream_t stream) override;
    /**
     * @copydoc GpuArray::get
     */
    void get(T* dst, cudaStream_t stream) const override;
    /**
     * @copydoc GpuArray::set
     */
    void set(const GpuArray<T>& src, cudaStream_t stream) override;
    /**
     * @copydoc GpuArray::get
     */
    void get(GpuArray<T>& dst, cudaStream_t stream) const override;
    /**
     * @copydoc GpuArray::set
     */
    void set(std::istream &input_stream, bool isText, bool isColMjr, Handle *hand) override;
    /**
     * @copydoc GpuArray::get
     */
    std::ostream &get(std::ostream &output_stream, bool isText, bool printColMajor, Handle &hand) const override;
    /**
     * @brief Access an element of the matrix as a Singleton.
     * @param row Row index.
     * @param col Column index.
     * @return Singleton wrapping the element at (row, col).
     */
    Singleton<T> get(size_t row, size_t col);
    /**
     * @brief Multiply this matrix with another matrix (result in new matrix or optional output).
     * @param other Matrix to multiply with.
     * @param result Optional pre-allocated matrix to store the result.
     * @param handle Optional Handle object for GPU streams.
     * @param alpha Optional scalar multiplier for this matrix.
     * @param beta Optional scalar multiplier for the result matrix.
     * @param transposeA Whether to transpose this matrix.
     * @param transposeB Whether to transpose the other matrix.
     * @return New matrix representing the result of multiplication.
     */
    virtual Mat<T> mult(const Mat<T>& other, Mat<T>* result, Handle* handle, const Singleton<T> *alpha, const
                        Singleton<T> *beta, bool transposeA, bool transposeB) const;
    /**
     * @brief Multiply this matrix with a vector.
     * @copydoc GpuArray::mult
     */
    virtual void mult(const Vec<T> &other, Vec<T> &result, Handle *handle, const Singleton<T> *alpha, const Singleton<T> *beta, bool
                      transpose) const;
    /**
     * @brief Multiply this matrix with a vector (operator overload).
     * @copydoc Mat::mult(const Vec<T>&, Vec<T>*, Handle*, const Singleton<T>*, const Singleton<T>*, bool) const
     */
    virtual Vec<T> operator*(const Vec<T>& other) const;

    /**
     * @brief Multiply this matrix with another matrix (operator overload).
     * @copydoc Mat::mult
     */
    virtual Mat<T> operator*(const Mat<T>& other) const;

    /**
     * @brief Add another matrix to this one.
     * @note Not yet implemented.
     */
    virtual Mat<T> plus(const Mat<T>& x, Mat<T>* result, const Singleton<T>* alpha, const Singleton<T>* beta, bool transposeA, bool transposeB, Handle* handle);

    /**
     * @brief Subtract another matrix from this one.
     * @note Not yet implemented.
     */
    virtual Mat<T> minus(const Mat<T>& x, Mat<T>* result, const Singleton<T>* alpha, const Singleton<T>* beta , bool transposeA, bool transposeB, Handle* handle);

    /**
     * @brief Multiply this matrix by a scalar in-place.
     * @note Not yet implemented.
     */
    virtual void mult(const Singleton<T>& alpha, Handle* handle);

    /**
     * @brief Transpose this matrix into a provided result matrix.
     * @note Not yet implemented.
     */
    virtual void transpose(Mat<T>& result, Handle* handle) const;

    /**
     * @brief Transpose this matrix, optionally using pre-allocated memory.
     * @note Not yet implemented.
     */
    virtual void transpose(Handle* handle, Mat<T>* preAlocatedMem);

    /**
     * @brief Static factory method to create a new matrix.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return New Mat<T> instance with allocated GPU memory.
     */
    static Mat<T> create(size_t rows, size_t cols);

    /**
     * @brief Extract a submatrix from this matrix.
     * @param startRow Row index of top-left corner.
     * @param startCol Column index of top-left corner.
     * @param height Number of rows in the submatrix.
     * @param width Number of columns in the submatrix.
     * @return Submatrix as a new Mat<T>.
     * @note Not yet implemented.
     */
    [[nodiscard]] virtual Mat<T> subMat(size_t startRow, size_t startCol, size_t height, size_t width) const;


    /**
     * @brief Normalize the columns of the matrix, setting a specific row to 1.
     * @param setRowTo1 Index of the row that should be set to 1.
     * @param handle Optional Handle object for GPU streams.
     * @note Not yet implemented.
     */
    virtual void normalizeCols(size_t setRowTo1, Handle* handle);


    /**
     * Creates a tensor that is a window into this data.
     * @param layers The number of layers in the tensor.  Be sure that layers is divisible by _rows.
     * @return a tensor that is a window into this data.
     */
    Tensor<T> tensor(size_t layers);


    static void batchMult(const Mat &a1, size_t strideA, const Mat &b1,
                          size_t strideB, Mat &c1, size_t strideC,
                          bool transposeA, bool transposeB,
                          Handle &hand, size_t batchCount,
                          const Singleton<T> &alpha = Singleton<T>::ONE, const Singleton<T> &beta = Singleton<T>::ZERO);


    KernelPrep kernelPrep();

    /**
     * @brief Retrieves a pointer to the underlying GPU memory data.
     *
     * This method provides access to the raw pointer of the memory allocated in the GPU
     * for the current array instance. The returned pointer can be used for direct read
     * or write operations on the GPU memory.
     *
     * @return A pointer to the GPU memory holding the array's data.
     * The pointer is valid as long as the GpuArray instance remains valid.
     */
    DeviceData2d<T> toKernel2d();
    DeviceData2d<T> toKernel2d() const;


    __host__ __device__ operator DeviceData2d<T>();
    __host__ __device__ operator DeviceData2d<T>() const;
};
#endif //BICGSTAB_MAT_H