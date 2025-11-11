#include "../headers/handle.h"

#include <iostream>
#include <memory>
#include <stdexcept>

Handle::Handle(cudaStream_t user_stream) {
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Failed to create cuBLAS handle");

    if (cusolverDnCreate(&cusolverHandle) != CUSOLVER_STATUS_SUCCESS) {
        cublasDestroy(handle);
        throw std::runtime_error("Failed to create cuSOLVER handle");
    }

    if (user_stream == nullptr) {
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            cublasDestroy(handle);
            throw std::runtime_error("Failed to create CUDA stream");
        }
        this->isOwner = true;
    } else {
        this->isOwner = false;
        this->stream = user_stream;
    }

    if (cublasSetStream(handle, this->stream) != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        if (this->isOwner) cudaStreamDestroy(this->stream);
        throw std::runtime_error("Failed to set CUBLAS stream");
    }
    if (cusolverDnSetStream(cusolverHandle, this->stream) != CUSOLVER_STATUS_SUCCESS) {
        cublasDestroy(handle);
        cusolverDnDestroy(cusolverHandle);
        if (this->isOwner) cudaStreamDestroy(this->stream);
        throw std::runtime_error("Failed to set cuSOLVER stream");
    }
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
}
Handle* Handle::_get_or_create_handle(Handle* handle, std::unique_ptr<Handle>& out_ptr_unique) {
    if (handle) return handle;
    else {
        out_ptr_unique = std::make_unique<Handle>();
        return out_ptr_unique.get();
    }
}

Handle::Handle() : Handle(nullptr) {}

Handle::~Handle() {
    cublasDestroy(handle);
    cusolverDnDestroy(cusolverHandle);
    if (this->isOwner) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        cudaStreamDestroy(stream);
    }
}

void Handle::synch() const {
    CHECK_CUDA_ERROR(cudaStreamSynchronize(this->stream));
}

Handle::operator cublasHandle_t() const {
    return handle;
}

Handle::operator cublasHandle_t() {
    return handle;
}

Handle::operator struct CUstream_st*() const {
    return stream;
}

void checkCudaErrors(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#include <cublas_v2.h> // Make sure this is included for cublasStatus_t

// New function to check CUBLAS errors
void checkCublasErrors(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        // CUBLAS doesn't have a simple string getter like cudaGetErrorString,
        // so we print the code or map it manually.
        std::cerr << "CUBLAS Error: " << status << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}