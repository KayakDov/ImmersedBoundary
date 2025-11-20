// --- GPUArray.h ---
// This file declares the classes and functions used in deviceArrays.cu.
// It is included by main.cu to let the compiler know what exists.
#ifndef GPUARRAY_H
#define GPUARRAY_H

#include "../defFiles/DeviceData.cu"
#include <memory> // For std::shared_ptr
#include "handle.h"
#include <iomanip>
#include "DeviceMemory.h"
#include "KernelPrep.cuh"


template <typename T> class Vec;
template <typename T> class Mat;
template <typename T> class Singleton;
template <typename T> class Tensor;
template <typename T> class SquareMat;
template <typename T> class BandedMat;

inline void cudaFreeDeleter(void* ptr) {
    if (ptr) cudaFree(ptr);
}

/**
 * @brief A class representing a GPU-accelerated multidimensional array.
 *
 * The GpuArray class provides functionality for creating, managing, and performing
 * computations on multidimensional arrays in GPU memory. It is designed for high
 * performance computational tasks that benefit from GPU acceleration.
 *
 * The class ensures efficient memory allocation, transfer, and synchronization
 * between host and device. It supports various operations, such as element-wise
 * computations, linear algebra operations, and reductions, optimized for GPU execution.
 *
 * @note Proper initialization of GPU devices and context is required for using this class.
 * It is designed to work with CUDA-enabled devices.
 *
 * @warning Attempting to operate on uninitialized or invalid GpuArray instances
 * may result in undefined behavior.
 */
template <typename T>
class GpuArray {
    friend Mat<T>;
    friend BandedMat<T>;
    friend SquareMat<T>;
private:

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


    operator DeviceData2d<T>();
    operator DeviceData2d<T>() const;
public:
    /**
     * @brief Represents the number of rows in a GpuArray instance.
     *
     * This constant indicates the row dimension of the matrix represented
     * by the GpuArray. It determines the number of elements accessible
     * along the row axis of the structure.
     *
     * @note This value is immutable and set during the construction of the
     * GpuArray instance.
     */
    const size_t _rows;
    /**
     * @brief Represents the number of columns in a GpuArray instance.
     *
     * This constant specifies the column dimension of the matrix represented
     * by the GpuArray. It determines the number of elements accessible along
     * the column axis of the structure.
     *
     * @note This value is immutable and is set during the construction of the
     * GpuArray instance.
     */
    const size_t _cols;

    /**
     * @brief Represents the leading dimension of a GpuArray instance.
     *
     * This constant specifies the stride in memory between consecutive columns
     * of the GpuArray.  This value will be greater than or equal to the height of the array,
     * the difference made up by elements of a larger array or padding.
     *
     * @note This value is immutable and is determined during the construction
     * of the GpuArray instance.
     */
    const size_t _ld;
protected:
    /**
     * @brief Shared pointer managing the memory associated with the GpuArray instance.
     *
     * This pointer is responsible for managing the lifetime of the GPU memory block
     * containing the data represented by the GpuArray.
     *
     * @note The pointer is immutable once set during the construction of the
     * GpuArray instance and is shared to facilitate memory reuse and resource
     * management among related instances or derived classes.
     */
    std::shared_ptr<T> _ptr;
    /**
     * @brief Constructs a GpuArray instance with specified dimensions, leading dimension,
     * and memory pointer.
     *
     * This constructor initializes a GpuArray instance with the specified number of
     * rows, columns, leading dimension (stride between consecutive columns (2d) or elements (1d) in memory),
     * and a shared pointer to the memory block representing the array data.
     * It is intended for use in scenarios where the array's parameters and memory
     * allocation are pre-determined.
     *
     * @param rows The number of rows in the GpuArray.
     * @param cols The number of columns in the GpuArray.
     * @param ld The leading dimension (column stride (2d) or element stride (1d) of the GpuArray.
     *           This value determines the memory layout of the array.
     * @param _ptr A shared pointer to the allocated memory containing the data
     *             for the GpuArray.
     */
    GpuArray(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> _ptr);


    /**
     * @brief Performs matrix multiplication with optional transposition, scaling factors, and result storage.
     *
     * This method multiplies the current GpuArray instance with another GpuArray, applying optional
     * transposition on the operands. The result is scaled by the alpha and beta parameters and stored
     * in the provided result GpuArray. The operation is performed using GPU-accelerated libraries.
     *
     * @param other The GpuArray instance to multiply with the current GpuArray.
     * @param result A pointer to the GpuArray where the result of the multiplication will be stored.
     * @param handle A handle to manage GPU computation contexts. If null, a new handle will be created.
     * @param alpha A pointer to a Singleton object representing the scaling factor for the matrix product.
     * @param beta A pointer to a Singleton object representing the scaling factor for the result matrix.
     * @param transposeA A boolean flag indicating whether to transpose the current GpuArray before multiplication.
     * @param transposeB A boolean flag indicating whether to transpose the other GpuArray before multiplication.
     *
     * @throws std::invalid_argument If the type T is unsupported or the dimensions of the matrices are incompatible.
     *
     * @note The provided GpuArray instances must be properly initialized. Undefined behavior may occur if they are not.
     * @warning Ensure that handle, alpha, beta, and result pointers are properly allocated and valid before calling.
     */
    virtual void mult(const GpuArray<T> & other, GpuArray<T> * result, Handle *handle, const Singleton<T> *alpha, const
                      Singleton<T> *beta, bool transposeA, bool transposeB) const;

    
public:
    /**
     * @brief Destructor for the GpuArray class. A shared pointer manages gpu memory,so it will be freed when the shared
     * pointer pointing to that memory frees it.
     *
     * Cleans up resources associated with the GpuArray instance. Ensures that any
     * allocated GPU memory or other resources tied to the object are properly released.
     *
     * This destructor is defaulted and relies on proper handling of object lifetime
     * to avoid resource leaks. Users should ensure no pending operations or dependencies
     * exist when destroying an instance of GpuArray.
     *
     * @note Proper synchronization may be required before object destruction in cases
     * where asynchronous GPU operations are in progress.
     *
     * @warning Failure to manage the lifecycle of GPU-related resources properly can
     * lead to undefined behavior or memory leaks.
     */
    virtual ~GpuArray();

    /**
     * @brief Returns the size of the data structure.
     *
     * This pure virtual function must be implemented by derived classes to provide
     * the size or the number of elements contained within the respective data structure.
     *
     * @return The number of elements in the data structure.
     *
     * @note The function is marked [[nodiscard]] to emphasize that the return value
     * should not be ignored, as it represents critical information about the underlying data.
     */
    [[nodiscard]] virtual size_t size() const = 0;

    /**
     * @brief Retrieves the size in bytes of the memory required.  Note that because of pitch, the memory occupied may
     * be greater.
     *
     * @return The size in bytes as a `size_t` value.
     */
    [[nodiscard]] virtual size_t bytes() const;

    /**
     * @brief Virtual method to copy data from host memory to device memory.
     *
     * This function facilitates the transfer of data from the host to the device
     * for processing. The implementation of this method should perform the memory
     * copy operation while ensuring synchronization using a specified CUDA stream.
     *
     * If the size of this array is different than the size of the source array, behavior is undefined.
     *
     * @param hostData Pointer to the data in host memory.
     * @param stream The CUDA stream used to execute the memory transfer operation.
     *
     * @note Proper synchronization of the CUDA stream is essential to avoid undefined
     * behavior during concurrent operations.
     *
     * @warning Ensure that the memory referenced by `hostData` is valid and accessible
     * during the function call to prevent runtime errors.
     */
    virtual void set(const T* hostData, cudaStream_t stream) = 0;

    /**
     * @brief Transfers the contents of the device array to host memory.
     *
     * The `get` method copies the data from the device array to a host memory buffer
     * provided by the user. The method utilizes the specified CUDA stream for the
     * asynchronous data transfer.
     *
     * @param hostData Pointer to the host memory buffer where the device data will be stored.
     *                 The user must ensure that the buffer is allocated with sufficient size
     *                 to hold the data being transferred.
     * @param stream CUDA stream to be used for the data transfer operation. This allows for
     *               pipelined and asynchronous memory operations with other processing
     *               on the device.
     */
    virtual void get(T* hostData, cudaStream_t stream) const = 0;

    /**
     * @brief Virtual function to set the content of the GPU-accelerated array.
     *
     * This method transfers the data from the source GPU array into the current instance
     * using the specified CUDA stream. It is designed to handle efficient data
     * synchronization and transfer between different GPU arrays.
     *
     * If the size of this array is different than the size of the source array, the behavior is undefined.
     *
     * @param src The source GpuArray object to copy data from.
     * @param stream The CUDA stream used for the data transfer operation.
     *
     * @note Proper synchronization mechanisms must be in place when using streams
     * to ensure data integrity, especially in asynchronous operations.
     *
     * @warning Attempting to use an invalid or uninitialized GpuArray as the source
     * or providing an invalid CUDA stream may lead to undefined behavior.
     */
    virtual void set(const GpuArray<T>& src, cudaStream_t stream ) = 0;

    /**
     * @brief Retrieves the content of the GPU array and transfers it to the specified destination.
     *
     * This method transfers the data of the GPU array to the provided destination array
     * using the specified CUDA stream for asynchronous execution. The destination array (`dst`)
     * must be appropriately sized to accommodate the data being retrieved.
     *
     * If the src array has a different size than this array, the behavior is undefined.
     *
     * The transfer ensures synchronization with other GPU operations within the same stream,
     * enabling coordinated computations and memory transfers.
     *
     * @param dst A reference to the destination GpuArray where the content will be transferred.
     * @param stream The CUDA stream used to perform the transfer operation asynchronously.
     */
    virtual void get(GpuArray<T>& dst, cudaStream_t stream) const = 0;

    /**
     * @brief Sets the data of the object using the provided input stream.
     *
     * This method reads data from the given input stream and configures the object
     * accordingly. The format of the input data and its layout can be specified using
     * additional parameters. GPU stream synchronization can also be managed using
     * a CUDA stream.
     *
     * @param input_stream The input stream from which the data is read.
     * @param isText A boolean indicating whether the input data is in text format (true) or binary format (false).
     * @param isColMjr
     * @param hand
     */
    virtual void set(std::istream &input_stream, bool isText, bool isColMjr, Handle *hand) = 0;


    /**
     * @brief Pure virtual function to retrieve data from the GPU array.
     *
     * This function is used to transfer data from the GPU array to a specified output stream. The
     * format and layout of the transferred data can be controlled via the provided parameters.
     * The operation is executed asynchronously using the specified CUDA stream.
     *
     * @param output_stream The output stream to which the data will be written.
     * @param isText Specifies whether the data should be written in a textual format.
     *               If true, data is formatted as text; otherwise, it will be written in a binary format.
     * @param printColMajor Indicates whether the data should be stored in column-major order.
     *                 If true, data is arranged column-major; otherwise, row-major order is used.
     * @param hand The CUDA stream that will be used to execute the data retrieval operation asynchronously.
     *
     * @note Proper synchronization is required if other operations depend on the retrieved data.
     * Ensure the CUDA stream is active and valid when calling this function.
     *
     * @warning Attempting to retrieve data from an uninitialized or invalid GPU array
     * may result in undefined behavior.
     */
    virtual std::ostream &get(std::ostream &output_stream, bool isText, bool printColMajor, Handle &hand) const = 0;

    /**
     * @brief Fills the GpuArray with the specified value on a given CUDA stream.
     *
     * This method assigns the specified value to all elements in the GpuArray
     * using a GPU kernel for parallel execution. It ensures efficient processing
     * over large arrays by leveraging GPU hardware and a provided CUDA stream for
     * asynchronous execution.
     *
     * @tparam T The data type of the elements in the GpuArray.
     * @param val The value with which to fill the array.
     * @param stream The CUDA stream used for kernel execution, enabling asynchronous
     * processing and overlapping computations with other operations.
     *
     * @warning Ensure that the array is properly allocated in GPU memory before
     * calling this method. Any errors during kernel execution, such as access
     * violations or invalid configurations, will need to be addressed separately.
     */
    virtual void fill(T val, cudaStream_t stream);


    /**
     * The shared pointer to the gpu data.
     * @return The shared pointer to the data.
     */
    std::shared_ptr<T> ptr() const;


    /**
     * Severs the connection to the gou memory.  If this is the last array using that block of memory, then the memory
     * is freed.
     */
    void freeMem();


    /**
     * https://en.wikipedia.org/wiki/Kronecker_product
     * @brief Computes the Kronecker product C = A (x) B + C, where A is *this and B is *other.
     *
     * The result matrix C will have dimensions (heightA * heightB) x (widthA * widthB).
     *
     * @param other The second matrix (B) in the Kronecker product.
     * @param result Pointer to the GpuArray where the result (C) will be added.
     * @param stream
     */
    void multKronecker(const GpuArray& other, GpuArray& result, cudaStream_t stream) const;

    /**
     * Allows this to be passed as a pointer to its data on the device.
     */
    explicit operator T*();
    /**
     * Allows this to be passed as a const pointer to its data on the device.
     */
    explicit operator const T*() const;

    /**
     *
     * @return A pointer to the gpu data.
     */
    [[nodiscard]] T* data();

    /**
     * A const pointer to the GPU data.
     * @return A const pointer to the GPU data.
     */
    const T *data() const;

    /**
     * Prepares default grid and block sizes for a kernel.
     * @return
     */
    [[nodiscard]] virtual KernelPrep kernelPrep() const;

    /**
     * @brief Extract a column vector from the matrix.
     * @param index Column index.
     * @return Column as a Vec<T>.
     * @note Not yet implemented.
     */
    virtual Vec<T> col(size_t index);

    Vec<T> col(size_t index) const;

    /**
     * @brief Extract a row vector from the matrix.
     * @param index Row index.
     * @return Row as a Vec<T>.
     * @note Not yet implemented.
     */
    virtual Vec<T> row(size_t index);

    Vec<T> row(size_t index) const;

    /**
     * @brief Gets the desired diagonal.  An index of 0 is the primary diagonal, a positive index indicates the start
     * column of the super diagonal and a negative index is the start row of the sub diagonal.
     *
     * @param index
     * @return The diagonal, as a vector.
     */
    virtual Vec<T> diag(int32_t index);

    Vec<T> diag(int32_t index) const;


    /**
     * Creates a vector that is a window into part of this GpuArray, or the underlying data.
     * Note: before using this method, make sure row, col, or diag don't meet your needs, as they are safer.
     * @param offset The vector starts here relative to the first element of this matrix.
     * @param ld The step size between vector elements.
     * @param size The number of elements in the new vector.
     * @return The new vector created that is a window into this matrix.
     */
    Vec<T> vec(size_t offset, size_t ld, size_t size);

    Vec<T> vec(size_t offset, size_t ld, size_t size) const;
};

#endif // GPUARRAY_H