#include "deviceArrays.h"


template <>
CuArray2D<float> CuArray2D<float>::mult(
    const CuArray2D<float>& other,
    CuArray2D<float>* result,
    CublasHandle* handle,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB
) const {
    if (this->_cols != other._rows) throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    
    CuArray2D<float>* resPtr = result ? result: new CuArray2D<float>(this->_rows, other._cols);
    
    CuArray<float>::mult(other, resPtr, handle, alpha, beta, transposeA, transposeB);

    if (!result) {
        CuArray2D<float> temp = *resPtr;
        delete resPtr;
        return temp;
    }
    
    return *resPtr;
}

template <>
CuArray2D<double> CuArray2D<double>::mult(
    const CuArray2D<double>& other,
    CuArray2D<double>* result,
    CublasHandle* handle,
    double alpha,
    double beta,
    bool transposeA,
    bool transposeB
) const {
    CuArray2D<double>* resPtr = result ? result: new CuArray2D<double>(this->_rows, other._cols);
    
    CuArray<double>::mult(other, resPtr, handle, alpha, beta, transposeA, transposeB);
    
    if (!result) {
        CuArray2D<double> temp = *resPtr;
        delete resPtr;
        return temp;
    }
    
    return *resPtr;
}

template <>
CuArray1D<float> CuArray2D<float>::mult(
    const CuArray1D<float>& other,
    CuArray1D<float>* result,
    CublasHandle* handle,
    float alpha,
    float beta,
    bool transpose
    
) const {

    CuArray1D<float>* resPtr = result ? result: new CuArray1D<float>(other._cols);
    
    CublasHandle* h = handle ? handle : new CublasHandle();

    cublasSgemv(h->handle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, this->_rows, this->_cols, &alpha, this->data(), this->getLD(), other.data(), other.getLD(), &beta, resPtr->data(), resPtr->getLD());

    if (!result) {
        CuArray1D<float> temp = *resPtr;
        delete resPtr;
        return temp;
    }

    if (!handle){
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
         delete h;
    }
    
    return *resPtr;
}

template <>
CuArray1D<double> CuArray2D<double>::mult(
    const CuArray1D<double>& other,
    CuArray1D<double>* result,
    CublasHandle* handle,
    double alpha,
    double beta,
    bool transpose
) const {

    CuArray1D<double>* resPtr = result ? result: new CuArray1D<double>(other._cols);

    CublasHandle* h = handle ? handle : new CublasHandle();
    
    cublasDgemv(
        h->handle,
        transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows, this->_cols,
        &alpha,
        this->data(), this->getLD(),
        other.data(), other.getLD(),
        &beta,
        resPtr->data(), resPtr->getLD()
    );

    if (!result) {
        CuArray1D<double> temp = *resPtr;
        delete resPtr;
        return temp;
    }

    if (!handle){
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
         delete h;
    }
    
    return *resPtr;
}

template <>
CuArray2D<float> CuArray2D<float>::operator*(const CuArray2D<float>& other) const {
    return this->mult(other);
}
template <>
CuArray2D<double> CuArray2D<double>::operator*(const CuArray2D<double>& other) const {
    return this->mult(other);
}
template <>
CuArray1D<float> CuArray2D<float>::operator*(const CuArray1D<float>& other) const {
    return this->mult(other);
}
template <>
CuArray1D<double> CuArray2D<double>::operator*(const CuArray1D<double>& other) const {
    return this->mult(other);
}

template <typename T>
CuArray2D<T>::CuArray2D(size_t rows, size_t cols): CuArray<T>(rows, cols, 0) {
    void* rawPtr = nullptr;
    size_t pitch = 0;
    cudaError_t err = cudaMallocPitch(&rawPtr, &pitch, rows * sizeof(T), cols);
    if (err != cudaSuccess)
        throw std::runtime_error("cudaMallocPitch failed");

    this->_ptr = std::shared_ptr<void>(rawPtr, cudaFreeDeleter);
    this->_ld = pitch / sizeof(T);
}

template <typename T>
CuArray2D<T>::CuArray2D(const CuArray2D<T>& superArray, size_t startRow, size_t startCol, size_t height, size_t width)
    : CuArray<T>(height, width, superArray.getLD()) {
    size_t offset = startCol * superArray.getLD() + startRow;
    this->_ptr = std::shared_ptr<void>(
        superArray._ptr,
        static_cast<char*>(superArray._ptr.get()) + offset * sizeof(T)
    );
}

template <typename T>
size_t CuArray2D<T>::size() const {
    return this->_rows * this->_cols;
}

template <typename T>
size_t CuArray2D<T>::bytes() const {
    return this->_cols * this->_ld * sizeof(T);
}

template <typename T>
void CuArray2D<T>::set(const T* src, cudaStream_t stream) {
    cudaMemcpy2DAsync(
        this->_ptr.get(), this->_ld * sizeof(T),
        src, this->_rows * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyHostToDevice, stream
    );
}

template <typename T>
void CuArray2D<T>::get(T* dst, cudaStream_t stream) const {
    cudaMemcpy2DAsync(
        dst, this->_rows * sizeof(T),
        this->_ptr.get(), this->_ld * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToHost, stream
    );
}

template <typename T>
void CuArray2D<T>::set(const CuArray<T>& src, cudaStream_t stream) {
    cudaMemcpy2DAsync(
        this->_ptr.get(), this->_ld * sizeof(T),
        src.data(), src.getLD() * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToDevice, stream
    );
}

template <typename T>
void CuArray2D<T>::get(CuArray<T>& dst, cudaStream_t cuStream) const {
    cudaMemcpy2DAsync(
        dst.data(), dst.getLD() * sizeof(T),
        this->_ptr.get(), this->_ld * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToDevice, cuStream
    );
}

template <typename T>
void CuArray2D<T>::set(std::istream& input_stream, cudaStream_t cuStream) {

    SetFromFile<T> helper(this->_rows, this->_cols, input_stream);

    while (helper.hasNext()) {
        helper.readNextChunk();
        CuArray2D<T> subArray(
            *this,
            0,
            helper.getColsProcessed(),
            this->_rows,
            helper.getNextChunkColNumber()
        );

        subArray.set(helper.getHostBuffer(), cuStream);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());//TODO: this might be avoidable with multi threading

        helper.updateProgress();
    }
}

template <typename T>
void CuArray2D<T>::get(std::ostream& output_stream, cudaStream_t stream) const {

    GetToFile<T> helper(this->_rows, this->_cols, output_stream);

    while (helper.hasNext()) {
        CuArray2D<T> subArray(
            *this,
            0,
            helper.getColsProcessed(),
            this->_rows,
            helper.getNextChunkColNumber()
        );

        subArray.get(helper.getHostBuffer(), stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());//TODO: this might be avoidable with multi threading

        helper.writeNextChunkToFile();
        helper.updateProgress();
    }
}


template class CuArray2D<int>;
template class CuArray2D<float>;
template class CuArray2D<double>;