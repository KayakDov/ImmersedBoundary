#include "../headers/handle.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <cublas_v2.h> // Make sure this is included for cublasStatus_t

void CublasDeleter::operator()(cublasHandle_t handle) const {
    if (handle) {
        cublasStatus_t status = cublasDestroy(handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            // We use std::cerr instead of throwing to prevent std::terminate during unique_ptr cleanup.
            std::cerr << "Warning: Failed to destroy cuBLAS handle. Status: " << status << std::endl;
        }
    }
}

void CusolverDeleter::operator()(cusolverDnHandle_t handle) const {
    if (handle) {
        cusolverStatus_t status = cusolverDnDestroy(handle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            std::cerr << "Warning: Failed to destroy cuSOLVER handle. Status: " << status << std::endl;
        }
    }
}



Handle::Handle() : Handle(nullptr) {}

Handle::Handle(cudaStream_t user_stream) {

    CHECK_CUDA_ERROR(cudaFree(0));

    cublasHandle_t rawHandle;
    CHECK_CUBLAS_ERROR(cublasCreate(&rawHandle));
    handlePtr = CublasHandlePtr(rawHandle, CublasDeleter());

    cusolverDnHandle_t rawSHandle;
    CHECK_SOLVER_ERROR(cusolverDnCreate(&rawSHandle));
    solverHandlePtr = CusolverHandlePtr(rawSHandle, CusolverDeleter());

    if (user_stream == nullptr) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
        this->isOwner = true;
    } else {
        this->isOwner = false;
        this->stream = user_stream;
    }

    CHECK_CUBLAS_ERROR(cublasSetStream(handlePtr.get(), this->stream));
    CHECK_SOLVER_ERROR(cusolverDnSetStream(solverHandlePtr.get(), this->stream));
    cublasSetPointerMode(handlePtr.get(), CUBLAS_POINTER_MODE_DEVICE);
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

Handle::operator cublasHandle_t() const {
    return handlePtr.get();
}

Handle::operator cublasHandle_t() {
    return handlePtr.get();
}

Handle::operator struct CUstream_st*() const {
    return stream;
}

Handle::operator struct cusolverDnContext*() const {
    return solverHandlePtr.get();
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