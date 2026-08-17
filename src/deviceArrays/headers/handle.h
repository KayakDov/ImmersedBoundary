/**
 * @file handle.h
 * @brief Defines RAII wrappers for CUDA streams and cuBLAS/cuSOLVER/cuSPARSE handles.
 * @ingroup device_arrays
 *
 * @details
 * These classes form the core GPU container layer. Public containers generally own or view CUDA device memory through shared ownership, while explicit factory functions allocate new storage.
 */

#ifndef BICGSTAB_HANDLE_H
#define BICGSTAB_HANDLE_H

#include <cusolverDn.h>
#include <memory>
#include <cusparse.h>

/**
 * @brief Checks a CUDA runtime error and throws a std::runtime_error if an error occurred.
 * @param err CUDA error code to check.
 * @param file Source file where the check occurred.
 * @param line Line number in the source file.
 *
 * This function is the backend of the CHECK_CUDA_ERROR macro.
 */
void checkCudaErrors(cudaError_t err, const char *file, int line);

/**
 * @brief Checks a cuBLAS status code and throws std::runtime_error on failure.
 * @param status cuBLAS status code to check.
 * @param file Source file where the check occurred.
 * @param line Line number where the check occurred.
 */
void checkCublasErrors(cublasStatus_t status, const char *file, int line);

/**
 * @brief Checks a cuSOLVER status code and throws std::runtime_error on failure.
 * @param status cuSOLVER status code to check.
 * @param file Source file where the check occurred.
 * @param line Line number where the check occurred.
 */
void checkSolverErrors(cusolverStatus_t status, const char *file, int line);

/**
 * @brief Checks a cuSPARSE status code and throws std::runtime_error on failure.
 * @param status cuSPARSE status code to check.
 * @param file Source file where the check occurred.
 * @param line Line number where the check occurred.
 */
void checkSparseErrors(cusparseStatus_t status, const char *file, int line);

/**
 * @brief Macro to check a CUDA runtime error and throw a runtime exception if needed.
 * @param err CUDA runtime function call to check.
 *
 * Expands to a call to checkCudaErrors passing the current file and line number.
 */
#define CHECK_CUDA_ERROR(err) checkCudaErrors(err, __FILE__, __LINE__)
#define CHECK_CUBLAS_ERROR(status) checkCublasErrors(status, __FILE__, __LINE__)
#define CHECK_SOLVER_ERROR(status) checkSolverErrors(status, __FILE__, __LINE__)
#define CHECK_SPARSE_ERROR(status) checkSparseErrors(status, __FILE__, __LINE__)

/**
 * @brief Custom deleter functor for cublasHandle_t.
 * The implementation is in handle.cu.
 */
struct CublasDeleter {
    void operator()(cublasHandle_t handle) const;
};

/**
 * @brief Custom deleter functor for cusolverDnHandle_t.
 * The implementation is in handle.cu.
 */
struct CusolverDeleter {
    void operator()(cusolverDnHandle_t handle) const;
};

/**
 * @brief Custom deleter functor for cusparseHandle_t.
 *
 * The implementation is in handle.cu.
 */
struct CusparseDeleter {
    void operator()(cusparseHandle_t handle) const;
};

inline void ensure_device_impl(int target_device) {

    thread_local int32_t cached_device = -1;

    if (cached_device != target_device) {
        CHECK_CUDA_ERROR(cudaSetDevice(target_device));
        cached_device = target_device;
    }
}

/** Owning smart pointer for a cuBLAS handle. */
using CublasHandlePtr = std::unique_ptr<std::remove_pointer<cublasHandle_t>::type, CublasDeleter>;
/** Owning smart pointer for a cuSOLVER dense handle. */
using CusolverHandlePtr = std::unique_ptr<std::remove_pointer<cusolverDnHandle_t>::type, CusolverDeleter>;
/** Owning smart pointer for a cuSPARSE handle. */
using CusparseHandlePtr = std::unique_ptr<std::remove_pointer<cusparseHandle_t>::type, CusparseDeleter>;



/**
 * @brief Wrapper class for managing cuBLAS, cuSOLVER, and CUDA streams.
 *
 * Handle encapsulates:
 * - A cublasHandle_t for cuBLAS operations
 * - A cusolverDnHandle_t for cuSOLVER operations
 * - A cudaStream_t for asynchronous execution
 *
 * The class handles proper initialization, stream association, and cleanup.
 * Ownership of the stream can either belong to the Handle instance or be external.
 */
class Handle {
private:
    CublasHandlePtr handlePtr;
    CusolverHandlePtr solverHandlePtr;
    CusparseHandlePtr sparseHandlePtr;
    cudaStream_t stream;

    size_t gpuIndex;

public:
    void ensureDevice() const { ensure_device_impl(gpuIndex); }


    /**
     * @brief Constructs a Handle with a user-provided CUDA stream.
     * @param user_stream Optional user-defined CUDA stream. If nullptr, a new stream is created.
     *
     * The constructed Handle will either own the stream (and destroy it on destruction)
     * or simply reference a user-provided stream.
     *
     * @throws std::runtime_error if handle creation or stream setup fails.
     */
    explicit Handle(cudaStream_t user_stream, size_t gpuIndex = 0);

    Handle();

    Handle(size_t gpuIndex);


    int device() const { return gpuIndex; }

    /**
     * @brief Get or create a Handle instance.
     * @param handle Pointer to an existing Handle. If non-null, it is returned as-is.
     * @param out_ptr_unique Reference to a unique_ptr where a new Handle will be stored if needed.
     * @return Pointer to a valid Handle instance (either the input or a newly created one).
     */
    static Handle *_get_or_create_handle(Handle *handle, std::unique_ptr<Handle> &out_ptr_unique);

    /**
     * @brief Destructor. Destroys cuBLAS and cuSOLVER handles.
     *
     * If the Handle owns the CUDA stream, synchronizes and destroys it as well.
     */
    ~Handle();

    /**
     * @brief Synchronizes all operations on the associated CUDA stream.
     *
     * Ensures that all pending GPU work submitted to this stream has completed.
     *
     * @throws std::runtime_error if stream synchronization fails.
     */
    void synch() const;

    operator cublasHandle_t() const;

    operator cublasHandle_t();

    operator cudaStream_t() const;

    operator cusolverDnHandle_t() const;

    operator cusparseHandle_t() const;

private:
    bool isOwner = false; ///< True if the Handle owns the CUDA stream and should destroy it
};

#endif //BICGSTAB_HANDLE_H
