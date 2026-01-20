#ifndef BICGSTAB_HANDLE_H
#define BICGSTAB_HANDLE_H

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <memory>
#include <cusparse.h>

/**
 * @brief Checks a CUDA runtime error and throws a std::runtime_error if an error occurred.
 * @param err CUDA error code to check.
 * @param file Source file where the check occurred.
 * @param line Line number in the source file.
 *
 *Handle must always be passed by reference!  TODO:figure out how to make this not necessary.
 *
 * This function is the backend of the CHECK_CUDA_ERROR macro.
 */
void checkCudaErrors(cudaError_t err, const char *file, int line);

void checkCublasErrors(cublasStatus_t status, const char *file, int line);

void checkSolverErrors(cusolverStatus_t status, const char *file, int line);

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

/** * @brief Custom deleter functor for cublasHandle_t.
 * The implementation is in handle.cu.
 */
struct CublasDeleter {
    void operator()(cublasHandle_t handle) const;
};

/** * @brief Custom deleter functor for cusolverDnHandle_t.
 * The implementation is in handle.cu.
 */
struct CusolverDeleter {
    void operator()(cusolverDnHandle_t handle) const;
};

struct CusparseDeleter {
    void operator()(cusparseHandle_t handle) const;
};

// --- Type Aliases for Smart Pointers ---
// We use unique_ptr with the raw pointer type and the custom deleter struct.
using CublasHandlePtr = std::unique_ptr<std::remove_pointer<cublasHandle_t>::type, CublasDeleter>;
using CusolverHandlePtr = std::unique_ptr<std::remove_pointer<cusolverDnHandle_t>::type, CusolverDeleter>;
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
public:



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
