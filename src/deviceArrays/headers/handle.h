#ifndef BICGSTAB_HANDLE_H
#define BICGSTAB_HANDLE_H

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <memory>

/**
 * @brief Checks a CUDA runtime error and throws a std::runtime_error if an error occurred.
 * @param err CUDA error code to check.
 * @param file Source file where the check occurred.
 * @param line Line number in the source file.
 *
 * This function is the backend of the CHECK_CUDA_ERROR macro.
 */
void checkCudaErrors(cudaError_t err, const char* file, int line);
void checkCublasErrors(cublasStatus_t status, const char* file, int line);
/**
 * @brief Macro to check a CUDA runtime error and throw a runtime exception if needed.
 * @param err CUDA runtime function call to check.
 *
 * Expands to a call to checkCudaErrors passing the current file and line number.
 */
#define CHECK_CUDA_ERROR(err) checkCudaErrors(err, __FILE__, __LINE__)
#define CHECK_CUBLAS_ERROR(status) checkCublasErrors(status, __FILE__, __LINE__)

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
public:
    cublasHandle_t handle;          ///< cuBLAS handle
    cusolverDnHandle_t cusolverHandle; ///< cuSOLVER handle
    cudaStream_t stream;            ///< CUDA stream associated with the handles

    /**
     * @brief Default constructor. Creates a new CUDA stream and initializes cuBLAS/cuSOLVER handles.
     */
    Handle();

    /**
     * @brief Constructs a Handle with a user-provided CUDA stream.
     * @param user_stream Optional user-defined CUDA stream. If nullptr, a new stream is created.
     *
     * The constructed Handle will either own the stream (and destroy it on destruction)
     * or simply reference a user-provided stream.
     *
     * @throws std::runtime_error if handle creation or stream setup fails.
     */
    explicit Handle(cudaStream_t user_stream);

    /**
     * @brief Get or create a Handle instance.
     * @param handle Pointer to an existing Handle. If non-null, it is returned as-is.
     * @param out_ptr_unique Reference to a unique_ptr where a new Handle will be stored if needed.
     * @return Pointer to a valid Handle instance (either the input or a newly created one).
     */
    static Handle* _get_or_create_handle(Handle* handle, std::unique_ptr<Handle>& out_ptr_unique);

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

private:
    bool isOwner = false; ///< True if the Handle owns the CUDA stream and should destroy it
};

#endif //BICGSTAB_HANDLE_H
