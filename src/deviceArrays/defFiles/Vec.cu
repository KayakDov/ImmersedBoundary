#include <utility>

#include "../headers/Vec.h"
#include "../headers/Singleton.h"
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
Vec<T> Vec<T>::create(size_t length, size_t stride, T *pointer) {
    return Vec<T>(length, nonOwningGpuPtr(pointer), stride);
}

template<typename T>
Vec<T> Vec<T>::subVec(const size_t offset, const size_t length, const size_t stride) const {
    return Vec<T>(
        length,
        std::shared_ptr<T>(this->_ptr, const_cast<T *>(this->toKernel1d() + offset * this->_ld * stride)),
        stride * this->_ld
    );
}

template <typename T>
Vec<T>* Vec<T>::_get_or_create_target(size_t length, Vec<T>* result, std::unique_ptr<Vec<T>>& out_ptr_unique, cudaStream_t stream) {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Vec<T>>(Vec<T>::create(length, stream));
        return out_ptr_unique.get();
    }
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
        cublasSdot(*h, this->_cols, this->toKernel1d(), this->_ld, other.toKernel1d(), other._ld, result.toKernel1d());
    else if constexpr (std::is_same_v<T, double>)
        cublasDdot(*h, this->_cols, this->toKernel1d(), this->_ld, other.toKernel1d(), other._ld, result.toKernel1d());
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
        cudaMemcpyAsync(this->data(), src.data(), bytes(), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpy2DAsync(
            this->data(), this->_ld * sizeof(T),
            src.data(), src._ld * sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyDeviceToDevice, stream
        );
    }
}

template<typename T>
void Vec<T>::get(GpuArray<T> &dst, cudaStream_t stream) const {
    if (this->_ld == 1 && dst._ld == 1) {
        cudaMemcpyAsync(dst.data(), this->data(), bytes(), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpy2DAsync(
            dst.data(), dst._ld * sizeof(T),
            this->data(), this->_ld * sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyDeviceToDevice, stream
        );
    }
}

template<typename T>
void Vec<T>::set(std::istream &input_stream, bool isText, bool isColMjr, Handle* hand) {

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(hand, temp_hand_ptr);

    StreamSet<T> helper(this->_rows, this->_cols, input_stream);
    while (helper.hasNext()) {
        helper.readChunk(isText);
        Vec<T> subVec = this->subVec(
            helper.getColsProcessed(),
            helper.getChunkWidth(),
            1
        );
        subVec.set(helper.getBuffer().data(), *h);
        helper.updateProgress();
    }
}

template<typename T>
std::ostream &Vec<T>::get(std::ostream &output_stream, bool isText, bool printColMajor, Handle &hand) const {
    StreamGet<T> helper(this->_rows, this->_cols, output_stream);
    while (helper.hasNext()) {
        Vec<T> subArray = this->subVec(
            helper.getColsProcessed(),
            helper.getChunkWidth(),
            1
        );
        subArray.get(helper.getBuffer().data(), hand);
        helper.writeChunk(isText);
        helper.updateProgress();
    }
    return output_stream;
}

template<typename T>
__global__ void fill1dKernel(DeviceData1d<T> a, const T val) {
    if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < a.cols) a[idx] = val;
}


template<typename T>
void Vec<T>::fill(T val, cudaStream_t stream) {
    if (this->_ld == 1 && (val == static_cast<T>(0) || sizeof(T) == 1))
        cudaMemset(this->toKernel1d(), val, size() * sizeof(T));
    else {
        KernelPrep kp = kernelPrep();
        fill1dKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(this->toKernel1d(), val);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template<typename T>
Singleton<T> Vec<T>::get(size_t i) {
    return Singleton<T>(std::shared_ptr<T>(this->_ptr, this->toKernel1d() + i * this->_ld));
}

template<typename T>
void Vec<T>::add(const Vec<T> &x, const Singleton<T> *alpha, Handle *handle) {
    if (this->_cols != x._cols)
        throw std::invalid_argument("Vector lengths do not match for add.");

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Singleton<T> > temp_a_ptr;
    const Singleton<T> *a = Singleton<T>::_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);

    if constexpr (std::is_same_v<T, float>)
        cublasSaxpy(*h, this->_cols, a->toKernel1d(), x.toKernel1d(), x._ld, this->toKernel1d(), this->_ld);
    else if constexpr (std::is_same_v<T, double>)
        cublasDaxpy(*h, this->_cols, a->toKernel1d(), x.toKernel1d(), x._ld, this->toKernel1d(), this->_ld);
    else throw std::invalid_argument("Vec::add unsupported type.");
}

template<typename T>
void Vec<T>::sub(const Vec<T> &x, const Singleton<T> *alpha, Handle *handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    Singleton<T> a = Singleton<T>::create(*h);
    a.set(-alpha->get(), *h);
    this->add(x, &a, h);
}

template<typename T>
void Vec<T>::mult(const Singleton<T> &alpha, Handle *handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    if constexpr (std::is_same_v<T, float>)
        cublasSscal(*h, this->_cols, alpha.toKernel1d(), this->toKernel1d(), this->_ld);
    else if constexpr (std::is_same_v<T, double>)
        cublasDscal(*h, this->_cols, alpha.toKernel1d(), this->toKernel1d(), this->_ld);
    else throw std::invalid_argument("Unsupported type.");
}


extern "C" __global__ void setup_kernel_float(curandState *state, unsigned long long seed, size_t size, size_t stride) {
    if (unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < size) curand_init(seed, id, 0, &state[id]);
}

extern "C" __global__ void
setup_kernel_double(curandState *state, unsigned long long seed, size_t size, size_t stride) {
    if (unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < size) curand_init(seed, id, 0, &state[id]);
}

__global__ void fillRandomKernel_float(DeviceData1d<float> array, curandState *state) {
    if (const size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < array.cols)
        array[id] = curand_uniform(&state[id]);
}

__global__ void fillRandomKernel_double(DeviceData1d<double> array, curandState *state) {
    if (unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < array.cols)
        array[id] = curand_uniform_double(&state[id]);
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
        setup_kernel_float<<<numBlocks, threadsPerBlock, 0, *h>>>(devStates.get(), 0, this->size(), this->_ld);
        h->synch();
        fillRandomKernel_float<<<numBlocks, threadsPerBlock, 0, *h>>>(
            this->toKernel1d(), devStates.get());
    } else if constexpr (std::is_same_v<T, double>) {
        setup_kernel_double<<<numBlocks, threadsPerBlock, 0, *h>>>(devStates.get(), 0, this->size(), this->_ld);
        h->synch();
        fillRandomKernel_double<<<numBlocks, threadsPerBlock, 0, *h>>>(
            this->toKernel1d(),  devStates.get());
    } else throw std::invalid_argument("Unsupported type.");
}


template<typename T>
__global__ void EBEPowKernel(DeviceData1d<T> data, const T *t, const T *n) {
    if (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < data.cols)
        data[idx] = (*t) * pow(data[idx], *n);
}

template<typename T>
void Vec<T>::EBEPow(const Singleton<T> &t, const Singleton<T> &n, cudaStream_t stream) {
    if (this->_cols == 0) return;

    KernelPrep kp = this->kernelPrep();

    EBEPowKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
        this->toKernel1d(),
        t.data(), n.data()
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
}


template<typename T>
__global__ void setSumKernel(DeviceData1d<T> result, const DeviceData1d<T> a, const DeviceData1d<T> b, const T *alpha, const T *beta) {
    if (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < result.cols)
        result[idx] = *alpha * a[idx] + *beta * b[idx];
}

template<typename T>
void Vec<T>::setSum(const Vec<T> &a, const Vec<T> &b, const Singleton<T> &alpha, const Singleton<T> &beta,
                    Handle *handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    KernelPrep kp = this->kernelPrep();

    setSumKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, *h>>>(
        this->toKernel1d(), // Destination: 'this' vector
        a.toKernel1d(), // Input 1: 'a' vector
        b.toKernel1d(), // Input 2: 'b' vector
        alpha.data(), // Scalar alpha (passed by value)
        beta.data() // Scalar beta (passed by value)
    );
}


template<typename T>
__global__ void setDifferenceKernel(DeviceData1d<T> d_result, const DeviceData1d<T> d_a, const DeviceData1d<T> d_b, const T* alpha, const T *beta
                                    ) {
    if (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < d_result.cols)
        d_result[idx] = *alpha * d_a[idx] - *beta * d_b[idx];
}

template<typename T>
void Vec<T>::setDifference(const Vec<T> &a, const Vec<T> &b, const Singleton<T> &alpha, const Singleton<T> &beta,
                           Handle *handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle *h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    KernelPrep kp = this->kernelPrep();
    setDifferenceKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, *h>>>(
        this->toKernel1d(), // Destination: 'this' vector
        a.toKernel1d(), // Input 1: 'a' vector
        b.toKernel1d(), // Input 2: 'b' vector
        alpha.data(), // Scalar alpha (passed by value)
        beta.data()
    );
}

template<typename T>
Tensor<T> Vec<T>::tensor(size_t height, size_t layers) {
    const size_t ld = height * layers;
    const size_t rows = size()/ld;
    return Tensor<T>(height, rows, layers, ld, this->_ptr);
}

template<typename T>
DeviceData1d<T> Vec<T>::toKernel1d() {
    return DeviceData1d<T>(this->size(), this->_ld, this->_ptr.get());
}

template<typename T>
DeviceData1d<T> Vec<T>::toKernel1d() const {
    return DeviceData1d<T>(this->size(), this->_ld, this->_ptr.get());
}

template<typename T>
Vec<T>::operator DeviceData1d<T>() {
    return toKernel1d();
}

template<typename T>
Vec<T>::operator DeviceData1d<T>() const {
    return toKernel1d();
}

template<typename T>
KernelPrep Vec<T>::kernelPrep() {
    return KernelPrep(this->size());
}

template<typename T>
Vec<T> GpuArray<T>::vec(size_t offset, size_t ld, size_t size) {
    return Vec<T>(size, std::shared_ptr<T>(this->_ptr, this->_ptr.get() + offset), ld);
}

template <typename T>
Vec<T> GpuArray<T>::col(const size_t index){
    if (index >= this->_cols) throw std::out_of_range("Out of range");
    return this->vec(index * this->_ld, 1, this->_rows);
}
template <typename T>
Vec<T> GpuArray<T>::col(const size_t index) const {
    return (const_cast<GpuArray<T>*>(this))->col(index);
}
template <typename T>
Vec<T> GpuArray<T>::row(const size_t index){
    if (index > this->_rows) throw std::out_of_range("Out of range");
    return this->vec(index, this->_ld, this->_cols);
}
template <typename T>
Vec<T> GpuArray<T>::row(const size_t index) const{
    return (const_cast<GpuArray<T>*>(this))->row(index);
}
template<typename T>
Vec<T> GpuArray<T>::diag(int32_t index) {

    if (index >= 0) {
        if (index >= this-> _cols) throw std::out_of_range("Out of range");
        const size_t size = std::min(this->_rows, this->_cols - index);
        return this->vec(index * this->_ld, this->_ld + 1, size);
    } else {
        if (-index >= this->_rows) throw std::out_of_range("Out of range.");
        const size_t size = std::min(this->_cols, this->_rows + index);
        return this->vec(-index, this->_ld + 1, size);
    }
}
template<typename T>
Vec<T> GpuArray<T>::diag(int32_t index) const {
    return (const_cast<GpuArray<T>*>(this))->diag(index);
}

template class Vec<float>;
template class Vec<double>;
template class Vec<size_t>;
template class Vec<int32_t>;
template class Vec<unsigned char>;

template Vec<float> GpuArray<float>::col(size_t) const;
template Vec<double> GpuArray<double>::col(size_t) const;
template Vec<size_t> GpuArray<size_t>::col(size_t) const; // Maps to 'unsigned long' in the error
template Vec<int> GpuArray<int>::col(size_t) const;
template Vec<unsigned char> GpuArray<unsigned char>::col(size_t) const;

// Instantiate GpuArray::row(size_t) const for all necessary types
template Vec<float> GpuArray<float>::row(size_t) const;
template Vec<double> GpuArray<double>::row(size_t) const;
template Vec<size_t> GpuArray<size_t>::row(size_t) const; // Maps to 'unsigned long' in the error
template Vec<int> GpuArray<int>::row(size_t) const;
template Vec<unsigned char> GpuArray<unsigned char>::row(size_t) const;