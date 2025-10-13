
#ifndef BICGSTAB_MAT_H
#define BICGSTAB_MAT_H

#include "deviceArrays.h"

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
     * @copydoc GpuArray::size
     */
    [[nodiscard]] size_t size() const override;
    /**
     * @copydoc GpuArray::bytes
     */
    [[nodiscard]] size_t bytes() const override;
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
    void set(std::istream& input_stream, bool isText, bool readColMjr, cudaStream_t stream) override;
    /**
     * @copydoc GpuArray::get
     */
    void get(std::ostream& output_stream, bool isText, bool printColMjr, cudaStream_t stream) const override;
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
    virtual Vec<T> mult(const Vec<T>& other, Vec<T>* result, Handle* handle, const Singleton<T> *alpha, const
                        Singleton<T> *beta, bool transpose) const;
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
     * @brief Extract a column vector from the matrix.
     * @param index Column index.
     * @return Column as a Vec<T>.
     * @note Not yet implemented.
     */
    virtual Vec<T> col(size_t index);

    /**
     * @brief Extract a row vector from the matrix.
     * @param index Row index.
     * @return Row as a Vec<T>.
     * @note Not yet implemented.
     */
    virtual Vec<T> row(size_t index);

    /**
     * @brief Normalize the columns of the matrix, setting a specific row to 1.
     * @param setRowTo1 Index of the row that should be set to 1.
     * @param handle Optional Handle object for GPU streams.
     * @note Not yet implemented.
     */
    virtual void normalizeCols(size_t setRowTo1, Handle* handle);
};
#endif //BICGSTAB_MAT_H