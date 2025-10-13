
#ifndef BICGSTAB_HANDLE_H
#define BICGSTAB_HANDLE_H
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <memory>

void checkCudaErrors(cudaError_t err, const char* file, int line);
#define CHECK_CUDA_ERROR(err) checkCudaErrors(err, __FILE__, __LINE__)

class Handle {
public:
    cublasHandle_t handle;
    cusolverDnHandle_t cusolverHandle{};
    cudaStream_t stream;

    Handle();

    explicit Handle(cudaStream_t user_stream);

    static Handle* _get_or_create_handle(Handle* handle, std::unique_ptr<Handle>& out_ptr_unique);

    ~Handle();

    void synch() const;

private:
    bool isOwner = false; // Flag to indicate if the class owns the stream and should destroy it.
};

#endif //BICGSTAB_HANDLE_H