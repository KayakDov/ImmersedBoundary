#include "../headers/handle.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <cublas_v2.h> // Make sure this is included for cublasStatus_t

void CublasDeleter::operator()(cublasHandle_t handle) const {
    if (handle) CHECK_CUBLAS_ERROR(cublasDestroy(handle));
}

void CusolverDeleter::operator()(cusolverDnHandle_t handle) const {
    if (handle) CHECK_SOLVER_ERROR(cusolverDnDestroy(handle));
}

void CusparseDeleter::operator()(cusparseHandle_t handle) const {
    if (handle) CHECK_SPARSE_ERROR(cusparseDestroy(handle));
}



Handle::Handle() : Handle(nullptr) {}

Handle::Handle(cudaStream_t user_stream) {

    CHECK_CUDA_ERROR(cudaFree(0));//TODO: see if things still work when I delete this.

    cublasHandle_t rawHandle;
    CHECK_CUBLAS_ERROR(cublasCreate(&rawHandle));
    handlePtr = CublasHandlePtr(rawHandle, CublasDeleter());

    cusolverDnHandle_t rawSHandle;
    CHECK_SOLVER_ERROR(cusolverDnCreate(&rawSHandle));
    solverHandlePtr = CusolverHandlePtr(rawSHandle, CusolverDeleter());

    cusparseHandle_t rawSparseHandle;
    CHECK_SPARSE_ERROR(cusparseCreate(&rawSparseHandle));
    sparseHandlePtr = CusparseHandlePtr(rawSparseHandle, CusparseDeleter());

    if (user_stream == nullptr) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
        this->isOwner = true;
    } else {
        this->isOwner = false;
        this->stream = user_stream;
    }

    CHECK_CUBLAS_ERROR(cublasSetStream(handlePtr.get(), this->stream));
    CHECK_SOLVER_ERROR(cusolverDnSetStream(solverHandlePtr.get(), this->stream));
    CHECK_SPARSE_ERROR(cusparseSetStream(sparseHandlePtr.get(), this->stream));

    cublasSetPointerMode(handlePtr.get(), CUBLAS_POINTER_MODE_DEVICE);
    cusparseSetPointerMode(sparseHandlePtr.get(), CUSPARSE_POINTER_MODE_DEVICE);
}
Handle* Handle::_get_or_create_handle(Handle* handle, std::unique_ptr<Handle>& out_ptr_unique) {
    if (handle) return handle;
    else {
        out_ptr_unique = std::make_unique<Handle>();
        return out_ptr_unique.get();
    }
}


Handle::~Handle() {
    if (this->isOwner) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
}

void Handle::synch() const {
    CHECK_CUDA_ERROR(cudaStreamSynchronize(this->stream));
}

Handle::operator struct cusolverDnContext*() const {
    return solverHandlePtr.get();
}


// Add this to handle.cu
Handle::operator cusparseHandle_t() const {
    return sparseHandlePtr.get();
}

// Clean up the stream operator to use the standard type
Handle::operator cudaStream_t() const {
    return stream;
}

// Add these to handle.cu to satisfy GpuArray and Mat links
Handle::operator cublasHandle_t() const {
    return handlePtr.get();
}

Handle::operator cublasHandle_t() {
    return handlePtr.get();
}



namespace {
    /**
     * @brief Helper function to construct the full error message and throw a runtime exception.
     * @param errorDetail The specific error message (e.g., "CUDA Error: invalid argument").
     * @param file The file where the error occurred.
     * @param line The line number where the error occurred.
     */
    void throwError(const std::string& errorDetail, const char* file, int line) {
        std::string errMsg = errorDetail + " at " + file + ":" + std::to_string(line);
        throw std::runtime_error(errMsg);
    }
} // namespace

void checkCudaErrors(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess)
        throwError("CUDA Error: " + std::string(cudaGetErrorString(err)), file, line);

}

void checkCublasErrors(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throwError("CUBLAS Error (Status Code " + std::to_string(status) + ")", file, line);
}

void checkSolverErrors(cusolverStatus_t status, const char* file, int line) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throwError("CUSOLVER Error (Status Code " + std::to_string(status) + ")", file, line);
}

void checkSparseErrors(cusparseStatus_t status, const char *file, int line) {
    if (status != CUSPARSE_STATUS_SUCCESS)
        throwError("CUSPARSE Error (Status Code " + std::string(cusparseGetErrorString(status)) + ")", file, line);
}
