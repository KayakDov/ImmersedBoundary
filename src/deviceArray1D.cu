#include "deviceArrays.h"



template <typename T>
CuArray1D<T>::CuArray1D(size_t length)
    : CuArray<T>(1, length, 1) {
    void* rawPtr = nullptr;
    cudaMalloc(&rawPtr, length * sizeof(T));
    this->_ptr = std::shared_ptr<void>(rawPtr, cudaFreeDeleter);
}

template <typename T>
CuArray1D<T>::CuArray1D(const CuArray1D<T>& superArray, size_t offset, size_t length, size_t stride)
    : CuArray<T>(1, length, stride * superArray.getLD()) {
    this->_ptr = std::shared_ptr<void>(
        superArray._ptr,
        static_cast<char*>(superArray._ptr.get()) + offset * superArray.getLD() * sizeof(T)
    );
}

template <>
CuArray1D<float> CuArray1D<float>::mult(
    const CuArray2D<float>& other,
    CuArray1D<float>* result,
    Handle* handle,
    float alpha,
    float beta,
    bool transpose
) const {

    CuArray1D<float>* resPtr = result ? result: new CuArray1D<float>(other._cols);
    
    other.mult(*this, resPtr, handle, alpha, beta, !transpose);  

    if (!result) {
        CuArray1D<float> temp = *resPtr;
        delete resPtr;
        return temp;
    }
    
    return *resPtr;
}

template <>
CuArray1D<double> CuArray1D<double>::mult(
    const CuArray2D<double>& other,
    CuArray1D<double>* result,
    Handle* handle,
    double alpha,
    double beta,
    bool transpose
) const {

    CuArray1D<double>* resPtr = result ? result: new CuArray1D<double>(other._cols);
    
    other.mult(*this, resPtr, handle, alpha, beta, !transpose);

    if (!result) {
        CuArray1D<double> temp = *resPtr;
        delete resPtr;
        return temp;
    }
    
    return *resPtr;
}

template <>
float CuArray1D<float>::mult(
    const CuArray1D<float>& other,
    Handle* handle
) const {
    if (this->_cols != other._cols)
        throw std::invalid_argument("Vector lengths do not match for dot product.");
    
    
    Handle* h = handle ? handle : new Handle();
    float resultHost = 0.0f;
    
    cublasSdot(h->handle, this->_cols, this->data(), this->getLD(), other.data(), other.getLD(), &resultHost);
    

    if (!handle){
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
         delete h;
    }
    
    return resultHost;
}

template <>
double CuArray1D<double>::mult(
    const CuArray1D<double>& other,
    Handle* handle
) const {
    if (this->_cols != other._cols)
        throw std::invalid_argument("Vector lengths do not match for dot product.");
    
    Handle* h = handle ? handle : new Handle();
    double resultHost = 0.0;
    
    cublasDdot(h->handle, this->_cols, this->data(), this->getLD(), other.data(), other.getLD(), &resultHost);

    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
    
    return resultHost;
}

template <>
CuArray1D<float> CuArray1D<float>::operator*(const CuArray2D<float>& other) const {
    return this->mult(other);
}
template <>
CuArray1D<double> CuArray1D<double>::operator*(const CuArray2D<double>& other) const {
    return this->mult(other);
}
template <>
float CuArray1D<float>::operator*(const CuArray1D<float>& other) const {
    return this->mult(other);
}
template <>
double CuArray1D<double>::operator*(const CuArray1D<double>& other) const {
    return this->mult(other);
}


template <typename T>
CuArray1D<T>::CuArray1D(const CuArray2D<T>& extractFrom, int index, IndexType indexType):
CuArray<T>(
    1,
    indexType == IndexType::Row ? extractFrom._cols : extractFrom._rows,
    indexType == IndexType::Row ? extractFrom.getLD() : 1
) {
    if ((indexType == IndexType::Column && static_cast<size_t>(index) >= extractFrom._cols) || (indexType == IndexType::Row && static_cast<size_t>(index) >= extractFrom._rows))
        throw std::out_of_range("Out of range");
    size_t offset = indexType == IndexType::Row ? static_cast<size_t>(index) : static_cast<size_t>(index) * extractFrom.getLD();
    this->_ptr = std::shared_ptr<void>(
        extractFrom.getPtr(),
        const_cast<void*>(reinterpret_cast<const void*>(reinterpret_cast<const char*>(extractFrom.data()) + offset * sizeof(T)))
    );
}

template <typename T>
size_t CuArray1D<T>::size() const {
    return this->_cols;
}

template <typename T>
size_t CuArray1D<T>::bytes() const {
    return this->_cols * this->_ld * sizeof(T);
}

template <typename T>
void CuArray1D<T>::set(const T* hostData, cudaStream_t stream) {
    if (this->_ld == 1) cudaMemcpyAsync(this->_ptr.get(), hostData, bytes(), cudaMemcpyHostToDevice, stream);
    else cudaMemcpy2DAsync(
            this->_ptr.get(), this->_ld * sizeof(T),
            hostData, sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyHostToDevice, stream
        );
}

template <typename T>
void CuArray1D<T>::get(T* hostData, cudaStream_t stream) const {
    if (this->_ld == 1)
        cudaMemcpyAsync(hostData, this->_ptr.get(), bytes(), cudaMemcpyDeviceToHost, stream);
    else cudaMemcpy2DAsync(
            hostData, sizeof(T),
            this->_ptr.get(), this->_ld * sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyDeviceToHost, stream
        );
}

template <typename T>
void CuArray1D<T>::set(const CuArray<T>& src, cudaStream_t stream) {
    if (this->_ld == 1 && src.getLD() == 1) {
        cudaMemcpyAsync(this->_ptr.get(), src.data(), bytes(), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpy2DAsync(
            this->_ptr.get(), this->_ld * sizeof(T),
            src.data(), src.getLD() * sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyDeviceToDevice, stream
        );
    }
}

template <typename T>
void CuArray1D<T>::get(CuArray<T>& dst, cudaStream_t stream) const {
    if (this->_ld == 1 && dst.getLD() == 1) {
        cudaMemcpyAsync(dst.data(), this->_ptr.get(), bytes(), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpy2DAsync(
            dst.data(), dst.getLD() * sizeof(T),
            this->_ptr.get(), this->_ld * sizeof(T),
            sizeof(T), this->_cols,
            cudaMemcpyDeviceToDevice, stream
        );
    }
}

template <typename T>
void CuArray1D<T>::set(std::istream& input_stream, cudaStream_t stream) {
    StreamSet<T> helper(this->_rows, this->_cols, input_stream);
    while (helper.hasNext()) {
        helper.readChunk();
        CuArray1D<T> subArray(
            *this,
            helper.getColsProcessed(),
            helper.getNextChunkColNumber()
        );
        subArray.set(helper.getHostBuffer(), stream);
        helper.updateProgress();
    }
}

template <typename T>
void CuArray1D<T>::get(std::ostream& output_stream, cudaStream_t stream) const {
    StreamGet<T> helper(this->_rows, this->_cols, output_stream);
    while (helper.hasNext()) {
        CuArray1D<T> subArray(
            *this,
            helper.getColsProcessed(),
            helper.getNextChunkColNumber()
        );
        subArray.get(helper.getHostBuffer(), stream);
        helper.writeChunk();
        helper.updateProgress();
    }
}

template <>
void CuArray1D<float>::add(const CuArray1D<float>& x, float alpha, Handle* handle) {
    if (this->_cols != x._cols) 
        throw std::invalid_argument("Vector lengths do not match for add.");
    
    Handle* h = handle ? handle : new Handle();
    
    cublasSaxpy(h->handle, this->_cols, &alpha, x.data(), x.getLD(), this->data(), this->getLD());
    
    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}

template <>
void CuArray1D<double>::add(const CuArray1D<double>& x, double alpha, Handle* handle) {
    if (this->_cols != x._cols)
        throw std::invalid_argument("Vector lengths do not match for add.");
        
    Handle* h = handle ? handle : new Handle();
    
    cublasDaxpy(h->handle, this->_cols, &alpha, x.data(), x.getLD(), this->data(), this->getLD());
    
    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}

template <>
void CuArray1D<float>::sub(const CuArray1D<float>& x, float alpha, Handle* handle) {
    this->add(x, -alpha, handle);
}

template <>
void CuArray1D<double>::sub(const CuArray1D<double>& x, double alpha, Handle* handle) {
    this->add(x, -alpha, handle);
}

template <>
void CuArray1D<float>::mult(float alpha, Handle* handle) {
    if (this->_cols == 0) {
        return;
    }
    
    Handle* h = handle ? handle : new Handle();
    
    cublasSscal(h->handle, this->_cols, &alpha, this->data(), this->getLD());
    
    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}

template <>
void CuArray1D<double>::mult(double alpha, Handle* handle) {
    if (this->_cols == 0) {
        return;
    }
    
    Handle* h = handle ? handle : new Handle();
    
    cublasDscal(h->handle, this->_cols, &alpha, this->data(), this->getLD());
    
    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}


extern "C" __global__ void setup_kernel_float(curandState* state, unsigned long long seed, size_t size, size_t stride) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        curand_init(seed, id, 0, &state[id * stride]);
    }
}

extern "C" __global__ void setup_kernel_double(curandState* state, unsigned long long seed, size_t size, size_t stride) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        curand_init(seed, id, 0, &state[id * stride]);
    }
}

__global__ void fillRandomKernel_float(float* array, size_t size, size_t stride, curandState* state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        array[id * stride] = curand_uniform(&state[id * stride]);
    }
}

__global__ void fillRandomKernel_double(double* array, size_t size, size_t stride, curandState* state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        array[id * stride] = curand_uniform_double(&state[id * stride]);
    }
}

template <>
void CuArray1D<float>::fillRandom(Handle* handle) {
    
    Handle* h = handle ? handle : new Handle();
    
    dim3 threadsPerBlock(256);
    dim3 numBlocks((this->_cols + threadsPerBlock.x - 1) / threadsPerBlock.x);

    curandState* devStates;
    CHECK_CUDA_ERROR(cudaMalloc(&devStates, this->_cols * sizeof(curandState)));

    setup_kernel_float<<<numBlocks, threadsPerBlock, 0, h->stream>>>(devStates, 0, this->size(), this->getLD());

    fillRandomKernel_float<<<numBlocks, threadsPerBlock, 0, h->stream>>>(this->data(), this->_cols, this->getLD(), devStates);
    
    CHECK_CUDA_ERROR(cudaFree(devStates));
    
    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}

template <>
void CuArray1D<double>::fillRandom(Handle* handle) {
    
    Handle* h = handle ? handle : new Handle();

    dim3 threadsPerBlock(256);
    dim3 numBlocks((this->_cols + threadsPerBlock.x - 1) / threadsPerBlock.x);

    curandState* devStates;
    CHECK_CUDA_ERROR(cudaMalloc(&devStates, this->_cols * sizeof(curandState)));

    setup_kernel_double<<<numBlocks, threadsPerBlock, 0, h->stream>>>(devStates, 0, this->size(), this->getLD());

    fillRandomKernel_double<<<numBlocks, threadsPerBlock, 0, h->stream>>>(this->data(), this->size(), this->getLD(), devStates);
    
    // Free the device states
    CHECK_CUDA_ERROR(cudaFree(devStates));

    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}


template class CuArray1D<int>;
template class CuArray1D<float>;
template class CuArray1D<double>;