#include <utility>

#include "../headers/vec.h"
#include "../headers/singleton.h"
#include <curand_kernel.h> // For curandState
#include "../headers/DeviceMemory.h"

#include "deviceArrays/headers/deviceArraySupport.h"


template<typename T>
Vec<T>::Vec(const size_t size, const std::shared_ptr<T> ptr, const size_t stride) : GpuArray<T>(
    static_cast<size_t>(1), size, stride, ptr) {
}

template<typename T>
Vec<T> Vec<T>::create(size_t length, cudaStream_t stream) {
    T *rawPtr = nullptr;
    cudaMallocAsync(&rawPtr, length * sizeof(T), stream);
    return Vec<T>(length, std::shared_ptr<T>(rawPtr, cudaFreeDeleter), 1);
}

template<typename T>
Vec<T> Vec<T>::subVec(const size_t offset, const size_t length, const size_t stride) const {
    return Vec<T>(
        length,
        std::shared_ptr<T>(this->_ptr, const_cast<T *>(this->data() + offset * this->_ld * stride)),
        stride * this->_ld
    );
}

template<typename T>
void Vec<T>::mult(
    const Mat<T> &other,
    Vec<T> &result,
    Handle *handle,
    const Singleton<T> *alpha,
    const Singleton<T> *beta,
    bool transpose
) const {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    std::unique_ptr<Singleton<T> > temp_a_ptr;
    const Singleton<T> *a = this->_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T> > temp_b_ptr;
    const Singleton<T> *b = this->_get_or_create_target(static_cast<T>(0), *h, beta, temp_b_ptr);

    other.mult(*this, result, handle, a, b, !transpose);
}

template<typename T>
void Vec<T>::mult(
    const Vec<T> &other,
    Singleton<T> &result,
    Handle *handle
) const {
    if (this->_cols != other._cols)
        throw std::invalid_argument("Vector lengths do not match for dot product.");

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);


    if constexpr (std::is_same_v<T, float>)
        cublasSdot(h->handle, this->_cols, this->data(), this->_ld, other.data(), other._ld, result.data());
    else if constexpr (std::is_same_v<T, double>)
        cublasDdot(h->handle, this->_cols, this->data(), this->_ld, other.data(), other._ld, result.data());
    else static_assert(!std::is_same_v<T, float> && !std::is_same_v<T, double>, "Vec::add unsupported type.");
}


template<typename T>
Vec<T> Vec<T>::operator*(const Mat<T> &other) const {
    Vec<T> result = Vec<T>::create(this->_cols, nullptr);
    this->mult(other, result);
    return result;
}

template<typename T>
T Vec<T>::operator*(const Vec<T> &other) const {
    Singleton<T> result = Singleton<T>::create();
    this->mult(other, result);
    return result.get();
}

template<typename T>
size_t Vec<T>::size() const {
    return this->_cols;
}

template<typename T>
size_t Vec<T>::bytes() const {
    return this->_cols * this->_ld * sizeof(T);
}

template<typename T>
void Vec<T>::set(const T *hostData, cudaStream_t stream) {
    if (this->_ld == 1) cudaMemcpyAsync(this->_ptr.get(), hostData, bytes(), cudaMemcpyHostToDevice, stream);
    else
        cudaMemcpy2DAsync(
            this->_ptr.get(), this->_ld * sizeof(T),
            hostData, sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyHostToDevice, stream
        );
}

template<typename T>
void Vec<T>::get(T *hostData, cudaStream_t stream) const {
    if (this->_ld == 1)
        cudaMemcpyAsync(hostData, this->_ptr.get(), bytes(), cudaMemcpyDeviceToHost, stream);
    else
        cudaMemcpy2DAsync(
            hostData, sizeof(T),
            this->_ptr.get(), this->_ld * sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyDeviceToHost, stream
        );
}

template<typename T>
void Vec<T>::set(const GpuArray<T> &src, cudaStream_t stream) {
    if (this->_ld == 1 && src._ld == 1) {
        cudaMemcpyAsync(this->_ptr.get(), src.data(), bytes(), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpy2DAsync(
            this->_ptr.get(), this->_ld * sizeof(T),
            src.data(), src._ld * sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyDeviceToDevice, stream
        );
    }
}

template<typename T>
void Vec<T>::get(GpuArray<T> &dst, cudaStream_t stream) const {
    if (this->_ld == 1 && dst._ld == 1) {
        cudaMemcpyAsync(dst.data(), this->_ptr.get(), bytes(), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpy2DAsync(
            dst.data(), dst._ld * sizeof(T),
            this->_ptr.get(), this->_ld * sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyDeviceToDevice, stream
        );
    }
}

template<typename T>
void Vec<T>::set(std::istream &input_stream, bool isText, bool isColMjr, cudaStream_t stream) {
    StreamSet<T> helper(this->_rows, this->_cols, input_stream);
    while (helper.hasNext()) {
        helper.readChunk(isText);
        Vec<T> subVec = this->subVec(
            helper.getColsProcessed(),
            helper.getChunkWidth(),
            1
        );
        subVec.set(helper.getBuffer().data(), stream);
        helper.updateProgress();
    }
}

template<typename T>
void Vec<T>::get(std::ostream &output_stream, bool isText, bool, cudaStream_t stream) const {
    StreamGet<T> helper(this->_rows, this->_cols, output_stream);
    while (helper.hasNext()) {
        Vec<T> subArray = this->subVec(
            helper.getColsProcessed(),
            helper.getChunkWidth(),
            1
        );
        subArray.get(helper.getBuffer().data(), stream);
        helper.writeChunk(isText);
        helper.updateProgress();
    }
}

template<typename T>
__global__ void fill1dKernel(T *__restrict__ a, const size_t size, const T val, const size_t stride) {
    if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size)
        a[idx * stride] = val;
}


template<typename T>
void Vec<T>::fill(T val, cudaStream_t stream) {
    if (this->_ld == 1 && (val == static_cast<T>(0) || sizeof(T) == 1))
        cudaMemset(this->data(), val, size() * sizeof(T));
    else {
        constexpr size_t BLOCK_SIZE = 256;
        size_t num_blocks = (this->size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill1dKernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(this->data(), this->size(), val, this->_ld);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template<typename T>
Singleton<T> Vec<T>::get(size_t i) {
    return Singleton<T>(std::shared_ptr<T>(this->_ptr, this->data() + i * this->_ld));
}

template<typename T>
void Vec<T>::add(const Vec<T> &x, const Singleton<T> *alpha, Handle *handle) {
    if (this->_cols != x._cols)
        throw std::invalid_argument("Vector lengths do not match for add.");

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Singleton<T> > temp_a_ptr;
    const Singleton<T> *a = this->_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);

    if constexpr (std::is_same_v<T, float>)
        cublasSaxpy(h->handle, this->_cols, a->data(), x.data(), x._ld, this->data(), this->_ld);
    else if constexpr (std::is_same_v<T, double>)
        cublasDaxpy(h->handle, this->_cols, a->data(), x.data(), x._ld, this->data(), this->_ld);
    else throw std::invalid_argument("Vec::add unsupported type.");
}

template<typename T>
void Vec<T>::sub(const Vec<T> &x, const Singleton<T> *alpha, Handle *handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    Singleton<T> a = Singleton<T>::create(h->stream);
    a.set(-alpha->get(), h->stream);
    this->add(x, &a, h);
}

template<typename T>
void Vec<T>::mult(const Singleton<T> &alpha, Handle *handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    if constexpr (std::is_same_v<T, float>)
        cublasSscal(h->handle, this->_cols, alpha.data(), this->data(), this->_ld);
    else if constexpr (std::is_same_v<T, double>)
        cublasDscal(h->handle, this->_cols, alpha.data(), this->data(), this->_ld);
    else throw std::invalid_argument("Unsupported type.");
}


extern "C" __global__ void setup_kernel_float(curandState *state, unsigned long long seed, size_t size, size_t stride) {
    if (unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < size) curand_init(seed, id, 0, &state[id]);
}

extern "C" __global__ void
setup_kernel_double(curandState *state, unsigned long long seed, size_t size, size_t stride) {
    if (unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < size) curand_init(seed, id, 0, &state[id]);
}

__global__ void fillRandomKernel_float(float *array, size_t size, size_t stride, curandState *state) {
    if (const size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < size)
        array[id * stride] = curand_uniform(&state[id]);
}

__global__ void fillRandomKernel_double(double *array, size_t size, size_t stride, curandState *state) {
    if (unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < size)
        array[id * stride] = curand_uniform_double(&state[id]);
}

template<typename T>
void Vec<T>::fillRandom(Handle *handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((this->_cols + threadsPerBlock.x - 1) / threadsPerBlock.x);

    curandState *rawDevStates = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&rawDevStates, this->_cols * sizeof(curandState)));

    std::unique_ptr<curandState, decltype(&cudaFreeDeleter)>
            devStates(rawDevStates, &cudaFreeDeleter);

    if constexpr (std::is_same_v<T, float>) {
        setup_kernel_float<<<numBlocks, threadsPerBlock, 0, h->stream>>>(devStates.get(), 0, this->size(), this->_ld);
        h->synch();
        fillRandomKernel_float<<<numBlocks, threadsPerBlock, 0, h->stream>>>(
            this->data(), this->_cols, this->_ld, devStates.get());
    } else if constexpr (std::is_same_v<T, double>) {
        setup_kernel_double<<<numBlocks, threadsPerBlock, 0, h->stream>>>(devStates.get(), 0, this->size(), this->_ld);
        h->synch();
        fillRandomKernel_double<<<numBlocks, threadsPerBlock, 0, h->stream>>>(
            this->data(), this->_cols, this->_ld, devStates.get());
    } else throw std::invalid_argument("Unsupported type.");
}


template<typename T>
__global__ void EBEPowKernel(T *data, size_t size, size_t stride, const T *t, const T *n) {
    if (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size) {
        size_t offset = idx * stride;
        data[offset] = (*t) * pow(data[offset], *n);
    }
}

template<typename T>
void Vec<T>::EBEPow(const Singleton<T> &t, const Singleton<T> &n, cudaStream_t stream) {
    if (this->_cols == 0) return;

    int blockSize = 256;
    int gridSize = (this->_cols + blockSize - 1) / blockSize;

    EBEPowKernel<<<gridSize, blockSize, 0, stream>>>(
        this->data(), this->_cols, this->_ld,
        t.data(), n.data()
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
}


template<typename T>
__global__ void setSumKernel(T *d_result, const T *d_a, const T *d_b, const T *alpha, const T *beta,
                             const size_t size) {
    if (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size)
        d_result[idx] = *alpha * d_a[idx] + *beta * d_b[idx];
}

template<typename T>
void Vec<T>::setSum(const Vec<T> &a, const Vec<T> &b, const Singleton<T> &alpha, const Singleton<T> &beta,
                    Handle *handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    constexpr int THREADS_PER_BLOCK = 256;
    int numBlocks = (this->size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    setSumKernel<<<numBlocks, THREADS_PER_BLOCK, 0, h->stream>>>(
        this->data(), // Destination: 'this' vector
        a.data(), // Input 1: 'a' vector
        b.data(), // Input 2: 'b' vector
        alpha.data(), // Scalar alpha (passed by value)
        beta.data(), // Scalar beta (passed by value)
        this->size() // Size
    );
}


template<typename T>
__global__ void setDifferenceKernel(T *d_result, const T *d_a, const T *d_b, const T *alpha, const T *beta,
                                    const size_t size) {
    if (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size)
        d_result[idx] = *alpha * d_a[idx] - *beta * d_b[idx];
}

template<typename T>
void Vec<T>::setDifference(const Vec<T> &a, const Vec<T> &b, const Singleton<T> &alpha, const Singleton<T> &beta,
                           Handle *handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    constexpr int THREADS_PER_BLOCK = 256;
    int numBlocks = (this->size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    setDifferenceKernel<<<numBlocks, THREADS_PER_BLOCK, 0, h->stream>>>(
        this->data(), // Destination: 'this' vector
        a.data(), // Input 1: 'a' vector
        b.data(), // Input 2: 'b' vector
        alpha.data(), // Scalar alpha (passed by value)
        beta.data(), // Scalar beta (passed by value)
        this->size() // Size
    );
}


template class Vec<float>;
template class Vec<double>;
template class Vec<size_t>;
template class Vec<int32_t>;
template class Vec<unsigned char>;
