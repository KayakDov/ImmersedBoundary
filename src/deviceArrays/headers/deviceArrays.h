// --- deviceArrays.h ---
// This file declares the classes and functions used in deviceArrays.cu.
// It is included by main.cu to let the compiler know what exists.
#ifndef DEVICEARRAYS_H
#define DEVICEARRAYS_H

#include <vector> // For std::vector
#include <memory> // For std::shared_ptr
#include <stdexcept> // For std::runtime_error
#include <cuda_runtime.h> // For CUDA runtime API
#include <fstream> // For file I/O
#include <string> // For std::string
#include "handle.h"
#include <iostream>
#include <iomanip>
#include "DeviceMemory.h"
#include <array>


template <typename T> class GpuArray;
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
 * An array stored on the GPU.
 * @tparam T The type of data that will be stored on the GPU.
 */
template <typename T>
/**
 * @brief A class representing a GPU-accelerated multi-dimensional array.
 *
 * The GpuArray class provides functionality for creating, managing, and performing
 * computations on multi-dimensional arrays in GPU memory. It is designed for high
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
class GpuArray {
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
    Mat<T>* _get_or_create_target(size_t rows, size_t cols, Mat<T>* result, std::unique_ptr<Mat<T>>& out_ptr_unique) const;

    /**
     * @brief Retrieves an existing GPU vector or creates a new one with the specified length.
     *
     * This method checks whether the given result pointer is valid. If it is, the existing
     * vector is returned. Otherwise, a new GPU vector is created with the specified length
     * and stream, and the unique pointer is updated to manage the new vector's lifetime.
     *
     * @param length The length of the GPU vector to be created if no valid result is provided.
     * @param result A pointer to an existing GPU vector; if valid, it will be returned.
     * @param out_ptr_unique A unique pointer that will be initialized with a new GPU vector
     *                       if the result pointer is null.
     * @param stream The CUDA stream used for creating the GPU vector.
     * @return A pointer to the existing or newly created GPU vector.
     */
    Vec<T>* _get_or_create_target(size_t length, Vec<T> *result, std::unique_ptr<Vec<T>> &out_ptr_unique, cudaStream_t stream) const;

    /**
     * @brief Retrieves an existing target or creates a new target of type Singleton.
     *
     * This method checks if the target `result` is already provided. If `result` is not
     * null, it returns the existing target. Otherwise, it creates a new instance of
     * `Singleton` using the given CUDA stream and assigns it to `out_ptr_unique`.
     * The newly created object is then returned.
     *
     * @param result Pointer to an existing `Singleton` instance, if available. If null,
     * a new instance will be created.
     * @param out_ptr_unique Reference to a `std::unique_ptr` to store the newly created
     * `Singleton` instance if `result` is null.
     * @param stream The CUDA stream used for initializing the `Singleton` instance
     * when creating a new target.
     * @return Pointer to the existing or newly created `Singleton` instance.
     */
    Singleton<T>* _get_or_create_target(Singleton<T> *result, std::unique_ptr<Singleton<T>> &out_ptr_unique, cudaStream_t stream) const;

    /**
     * @brief Retrieves an existing target or creates a new one with the specified default value.
     *
     * This method fetches a given target if it exists, or initializes and returns a new
     * instance of the target using the provided default value and handle. If no target exists
     * (`result` is null), a new instance is created using `std::unique_ptr` and returned.
     *
     * @param defaultVal The default value used for initializing the new target, if required.
     * @param hand The handle providing additional context such as a stream for target creation.
     * @param result Pointer to an existing target instance, or null if no instance exists.
     * @param out_ptr_unique A unique pointer that holds ownership of a newly created target
     * if `result` is null.
     * @return A pointer to an existing or newly created target of type `Singleton<T>`.
     */
    const Singleton<T>* _get_or_create_target(T defaultVal, Handle& hand, const Singleton<T>* result, std::unique_ptr<Singleton<T>>& out_ptr_unique) const;

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
    virtual void mult(const GpuArray<T>& other, GpuArray<T>* result, Handle* handle, const Singleton<T> *alpha, const Singleton<T> *beta, bool transposeA, bool transposeB) const;

    
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
     * @brief Retrieves the size in bytes of the memory required or occupied.
     *
     * This method is used to determine the amount of memory, in bytes, associated
     * with the object or data structure implementing this interface.
     *
     * @return The size in bytes as a `size_t` value.
     */
    [[nodiscard]] virtual size_t bytes() const = 0;

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
     * @param stream The CUDA stream used for data transfer and operations, ensuring asynchronous execution.
     */
    virtual void set(std::istream& input_stream, bool isText, bool isColMjr, cudaStream_t stream) = 0;


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
     * @param isColMjr Indicates whether the data should be stored in column-major order.
     *                 If true, data is arranged column-major; otherwise, row-major order is used.
     * @param stream The CUDA stream that will be used to execute the data retrieval operation asynchronously.
     *
     * @note Proper synchronization is required if other operations depend on the retrieved data.
     * Ensure the CUDA stream is active and valid when calling this function.
     *
     * @warning Attempting to retrieve data from an uninitialized or invalid GPU array
     * may result in undefined behavior.
     */
    virtual void get(std::ostream& output_stream, bool isText, bool isColMjr, cudaStream_t stream) const = 0;

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
     * @brief Retrieves a pointer to the underlying GPU memory data.
     *
     * This method provides access to the raw pointer of the memory allocated in the GPU
     * for the current array instance. The returned pointer can be used for direct read
     * or write operations on the GPU memory.
     *
     * @return A pointer to the GPU memory holding the array's data.
     * The pointer is valid as long as the GpuArray instance remains valid.
     */
    T* data();

    /**
     * The shared pointer to the gpu data.
     * @return The shared pointer to the data.
     */
    std::shared_ptr<T> ptr() const;

    /**
     * @brief Retrieves the pointer to the underlying data stored in GPU memory.
     *
     * This method provides access to the raw GPU memory containing the array elements.
     * It is useful for low-level operations that need direct access to the data buffer.
     *
     * @return A constant pointer to the GPU memory holding the array data.
     *         The pointer remains valid as long as the GpuArray instance is not destroyed
     *         or resized.
     *
     * @warning Direct manipulation of the returned pointer should be done with caution
     *          to avoid data corruption or undefined behavior.
     */
    [[nodiscard]] const T* data() const;

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
};

/**
 * @brief Input formatted data (space-separated values) into gpuArray1D<T> from a stream.
 * @tparam T Element type
 * @param is Input stream
 * @param arr Array to fill
 * @return Input stream
 */
template <typename T>
std::istream& operator>>(std::istream& is, Vec<T>& arr) {

    std::vector<T> hostData(arr.size());
    for (size_t i = 0; i < hostData.size(); ++i) {
        is >> hostData[i];
        if (!is) {
            is.setstate(std::ios::badbit);
            break;
        }
    }
    arr.set(hostData.data());
    return is;
}

/**
 * @brief Prints a gpuArray1D<T> to a stream.
 * @tparam T Element type
 * @param os Output stream
 * @param arr The gpuArray1D to print
 * @return Output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const Vec<T>& arr) {
    std::vector<T> hostData(arr.size());
    Handle hand;
    arr.get(hostData.data(), hand.stream);
    hand.synch();

    for (size_t i = 0; i < hostData.size(); ++i) {
        os << hostData[i];
        if (i + 1 < hostData.size()) {
            os << " ";
        }
    }
    os << "\n";
    return os;
}

/**
 * @brief Prints a gpuArray2D<T> to a stream with improved formatting.
 *
 * This operator assumes a column-major memory layout and prints the matrix
 * row by row for a more readable output. It uses iomanip to format the
 * floating-point numbers to a fixed precision and set width.
 *
 * @tparam T Element type
 * @param os Output stream
 * @param arr The gpuArray2D to print
 * @return Output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const Mat<T>& arr) {
    cudaDeviceSynchronize();
    std::vector<T> hostData(arr.size());

    Handle hand;
    arr.get(hostData.data(), hand.stream);
    hand.synch();

    for (size_t r = 0; r < arr._rows; ++r) {
        for (size_t c = 0; c < arr._cols; ++c) {
            // Contiguous column-major access
            os << hostData[c * arr._rows + r] << " ";
        }
        os << "\n";
    }
    return os;
}


/**
 * @brief Reads formatted data (column by column) into gpuArray2D<T> from a stream.
 *
 * This operator assumes the input stream is formatted in column-major order,
 * meaning it will read all elements of the first column, then the second, and so on.
 *
 * @tparam T Element type
 * @param is Input stream
 * @param arr Array to fill
 * @return Input stream
 */
template <typename T>
std::istream& operator>>(std::istream& is, Mat<T>& arr) {
    std::vector<T> hostData(arr.size());
    size_t ld = arr.getLD();

    for (size_t c = 0; c < arr._cols; ++c) {
        for (size_t r = 0; r < arr._rows; ++r) {
            // Reading data in column-major order from the stream
            is >> hostData[c * ld + r];
            if (!is) {
                is.setstate(std::ios::badbit);
                return is;
            }
        }
    }

    arr.set(hostData.data());
    return is;
}

template <typename T>
/**
 * @brief Overloads the stream insertion operator for GpuArray instances.
 *
 * This function provides formatted output for GpuArray objects by determining
 * whether the array instance is one-dimensional (Vec) or two-dimensional (Mat).
 * It dynamically casts the input object to the appropriate type and delegates
 * stream formatting to the corresponding type's stream operator.
 *
 * @tparam T The data type of the elements in the GpuArray.
 * @param os The output stream to which the array's content will be written.
 * @param arr The GpuArray instance to be formatted and written to the stream.
 * @return A reference to the output stream with the formatted array content.
 *
 * @throws std::runtime_error If the function fails to determine whether the input
 * array is of type Vec or Mat, indicating the array's type is unsupported.
 *
 * @note The underlying implementation assumes that valid dynamic casting can
 * differentiate between one-dimensional and two-dimensional arrays derived
 * from the GpuArray base class.
 */
std::ostream& operator<<(std::ostream& os, const GpuArray<T>& arr) {
    
    if (auto ptr1d = dynamic_cast<const Vec<T>*>(&arr)) return os << *ptr1d;
    else if (auto ptr2d = dynamic_cast<const Mat<T>*>(&arr)) return os << *ptr2d;
    else throw std::runtime_error("Unable to detect the type of array, 1d or 2d.");
    return os;
}


#endif // DEVICEARRAYS_H