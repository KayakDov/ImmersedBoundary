#include "deviceArrays.h"

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
    if (this->isOwner) cudaStreamDestroy(stream);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void Handle::synch() const {
    CHECK_CUDA_ERROR(cudaStreamSynchronize(this->stream));
}