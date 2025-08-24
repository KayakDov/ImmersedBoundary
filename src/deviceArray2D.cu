#include "deviceArrays.h"


template <>
CuArray2D<float> CuArray2D<float>::mult(
    const CuArray2D<float>& other,
    CuArray2D<float>* result,
    Handle* handle,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB
) const {
    
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
    Handle* handle,
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
    Handle* handle,
    float alpha,
    float beta,
    bool transpose
    
) const {

    CuArray1D<float>* resPtr = result ? result: new CuArray1D<float>(other._cols);
    
    Handle* h = handle ? handle : new Handle();

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

/**
 * Multiplies a 2D array with a 1D array, returning a new 1D array.
 * @param other The 1D array to multiply with.
 * @param result Optional pointer to an existing 1D array to store the result.
 * @param handle Optional CublasHandle for managing the cuBLAS context.
 * @param alpha Scalar multiplier for the result.   
 *  @param beta Scalar multiplier for the existing values in the result array.
 * @param transpose If true, the 2D array is transposed before multiplication.
 * @return A new 1D array containing the result of the multiplication.
 */
template <>
CuArray1D<double> CuArray2D<double>::mult(
    const CuArray1D<double>& other,
    CuArray1D<double>* result,
    Handle* handle,
    double alpha,
    double beta,
    bool transpose
) const {

    CuArray1D<double>* resPtr = result ? result: new CuArray1D<double>(other._cols);

    Handle* h = handle ? handle : new Handle();
    
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

    StreamSet<T> helper(this->_rows, this->_cols, input_stream);

    while (helper.hasNext()) {
        helper.readChunk();
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

    StreamGet<T> helper(this->_rows, this->_cols, output_stream);

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

        helper.writeChunk();
        helper.updateProgress();
    }
}


template <>
CuArray2D<float> CuArray2D<float>::plus(
    const CuArray2D<float>& x, 
    CuArray2D<float>* result,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB,
    Handle* handle
) {
    if (this->_rows != x._rows || this->_cols != x._cols) {
        throw std::invalid_argument("Matrix dimensions do not match for add.");
    }
    
    // Determine the result pointer, creating a new one if necessary
    CuArray2D<float>* resPtr = result ? result : new CuArray2D<float>(this->_rows, this->_cols);
    
    Handle* h = handle ? handle : new Handle();

    cublasSgeam(
        h->handle, 
        transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
        transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows, this->_cols, 
        &alpha, x.data(), x.getLD(),
        &beta, this->data(), this->getLD(),
        resPtr->data(), resPtr->getLD()
    );

    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
    
    if (!result) {
        CuArray2D<float> temp = *resPtr;
        delete resPtr;
        return temp;
    }
    
    return *resPtr;
}

template <>
CuArray2D<double> CuArray2D<double>::plus(
    const CuArray2D<double>& x, 
    CuArray2D<double>* result,
    double alpha,
    double beta,
    bool transposeA,
    bool transposeB,
    Handle* handle
) {
    if (this->_rows != x._rows || this->_cols != x._cols) {
        throw std::invalid_argument("Matrix dimensions do not match for add.");
    }

    // Determine the result pointer, creating a new one if necessary
    CuArray2D<double>* resPtr = result ? result : new CuArray2D<double>(this->_rows, this->_cols);
    
    Handle* h = handle ? handle : new Handle();

    cublasDgeam(
        h->handle, 
        transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
        transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows, this->_cols, 
        &alpha, x.data(), x.getLD(),
        &beta, this->data(), this->getLD(),
        resPtr->data(), resPtr->getLD()
    );
    
    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
    
    if (!result) {
        CuArray2D<double> temp = *resPtr;
        delete resPtr;
        return temp;
    }
    
    return *resPtr;
}

template <>
CuArray2D<float> CuArray2D<float>::minus(
    const CuArray2D<float>& x,
    CuArray2D<float>* result,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB,
    Handle* handle
) {
    return this->plus(x, result, -alpha, beta, transposeA, transposeB, handle);
}

template <>
CuArray2D<double> CuArray2D<double>::minus(
    const CuArray2D<double>& x,
    CuArray2D<double>* result,
    double alpha,
    double beta,
    bool transposeA,
    bool transposeB,
    Handle* handle
) {
    return this->plus(x, result, -alpha, beta, transposeA, transposeB, handle);
}

template <typename T>
__global__ void scaleKernel(T* matrix, size_t rows, size_t cols, size_t ld, T alpha) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) matrix[row + col * ld] *= alpha;
    
}

template <typename T>
void CuArray2D<T>::_scale_impl(T alpha, Handle* handle) {
    if (this->_rows == 0 || this->_cols == 0) {
        return;
    }
    
    Handle* h = handle;
    if (!h) {
        h = new Handle();
    }

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(
        (this->_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (this->_rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    scaleKernel<<<numBlocks, threadsPerBlock, 0, h->stream>>>(
        this->data(),
        this->_rows,
        this->_cols,
        this->getLD(),
        alpha
    );

    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }
}

template <>
void CuArray2D<float>::mult(float alpha, Handle* handle) {
    this->_scale_impl(alpha, handle);
}

template <>
void CuArray2D<double>::mult(double alpha, Handle* handle) {
    this->_scale_impl(alpha, handle);
}

/**
 * Multiplies a banded 2D float matrix with a 1D vector using cuBLAS gbmv.
 */
template <>
CuArray1D<float> CuArray2D<float>::bandedMult(
    const CuArray1D<float>& x,
    CuArray1D<float>* result,
    Handle* handle,
    float alpha,
    float beta,
    int kl,   // number of sub-diagonals
    int ku,   // number of super-diagonals
    bool transpose
) const {
    if ((transpose && this->_rows != x._cols) ||
        (!transpose && this->_cols != x._cols)) {
        throw std::invalid_argument("Matrix/vector dimensions do not match for banded multiplication.");
    }

    CuArray1D<float>* resPtr = result ? result : new CuArray1D<float>(transpose ? this->_cols : this->_rows);

    Handle* h = handle ? handle : new Handle();

    cublasSgbmv(
        h->handle,
        transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows,
        this->_cols,
        kl,
        ku,
        &alpha,
        this->data(),
        this->getLD(),
        x.data(),
        x.getLD(),
        &beta,
        resPtr->data(),
        resPtr->getLD()
    );

    if (!result) {
        CuArray1D<float> temp = *resPtr;
        delete resPtr;
        return temp;
    }

    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }

    return *resPtr;
}

/**
 * Multiplies a banded 2D double matrix with a 1D vector using cuBLAS gbmv.
 */
template <>
CuArray1D<double> CuArray2D<double>::bandedMult(
    const CuArray1D<double>& x,
    CuArray1D<double>* result,
    Handle* handle,
    double alpha,
    double beta,
    int kl,   // number of sub-diagonals
    int ku,   // number of super-diagonals
    bool transpose
) const {
    if ((transpose && this->_rows != x._cols) ||
        (!transpose && this->_cols != x._cols)) {
        throw std::invalid_argument("Matrix/vector dimensions do not match for banded multiplication.");
    }

    CuArray1D<double>* resPtr = result ? result : new CuArray1D<double>(transpose ? this->_cols : this->_rows);

    Handle* h = handle ? handle : new Handle();

    cublasDgbmv(
        h->handle,
        transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
        this->_rows,
        this->_cols,
        kl,
        ku,
        &alpha,
        this->data(),
        this->getLD(),
        x.data(),
        x.getLD(),
        &beta,
        resPtr->data(),
        resPtr->getLD()
    );

    if (!result) {
        CuArray1D<double> temp = *resPtr;
        delete resPtr;
        return temp;
    }

    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }

    return *resPtr;
}



/**
 * Kernel for sparse diagonal matrix-vector multiplication.
 * 
 * When calling this kernel, <<<numberOfBlocks, threadsPerBlock, sharedMemorySize, stream>>>,  Number of blocks should be the number of rows in the solution vector.  
 * Threads per block should be numDiags/2.  Shared memory size should be sizeof(T) * numDiags, where numDiags is the number of diagonals in the packed matrix.
 * 
 * @param A Packed diagonals of the matrix.  Trailing values are not read.  Each row is a diagonal, and the matrix is stored in column-major order.
 * @param height Number of rows in the greater matrix, not the packed matrix.  This is also the number of columns in the greater matrix and the number of rows in x.
 * @param ld Leading dimension of the packed diagonals.
 * @param diags Indices of the diagonals.  Negative indices indicate sub-diagonals. 
 * Positive indices indicate super-diagonals. 
 * For example, diags = {-1, 0, 1} means the first diagonal is the sub-diagonal, the second is the main diagonal, and the third is the super-diagonal.
 * @param numDiags Number of nonzero diagonals.  This number must be less than or eqaul to 64.
 * @param x Input vector.
 * @param stride Stride for the input vector x.
 * @param result Output vector.
 * @param strideR Stride for the output vector result.
 * @param alpha Scalar multiplier for the matrix-vector product.
 * @param beta Scalar multiplier for the existing values in the result vector.
 */
template <typename T>
__global__ void diagMatVecKernel(
    const T* __restrict__ A, // packed diagonals
    const int height, const int ld,
    
    const int* __restrict__ diags, // which diagonals
    const int numDiags,

    const T* __restrict__ x,      // input vector
    const int strideX,
    
    T* __restrict__ result,       // output vecto
    const int strideR,

    const T alpha,
    const T beta
){
    extern __shared__ __align__(sizeof(T)) unsigned char sharedMem[];
    T *sData = reinterpret_cast<T *>(sharedMem);
    
    int rowX = blockIdx.x;
    int idx = threadIdx.x;
    bool isValid = rowX < height && idx * 2 < numDiags;
    if (isValid){        
        
        int d = diags[2 * idx]; // First diagonal index
        int i = (d >= 0) ? rowX : rowX + d; // Row index for first diagonal
        sData[idx] = 0 <= i && i < height - abs(d) ? A[i*ld + 2 * idx] * x[(rowX + d) * strideX] : 0;
        
        if(2 * idx + 1 < numDiags) {
            d = diags[2 * idx + 1]; // Second diagonal index
            i = (d >= 0) ? rowX : rowX + d; // Row index for second diagonal
            sData[idx] += 0 <= i && i < height - abs(d) ? A[i*ld + 2 * idx + 1] * x[(rowX + d) * strideX] : 0;
        }
    }

    for(int s = 1; s < blockDim.x; s *= 2) {
        __syncthreads(); 
        if(idx % (s * 2) == 0 && isValid && (idx + s) * 2 < numDiags) sData[idx] += sData[idx + s];
    }
    
    if(isValid && idx == 0) result[rowX * strideR] = alpha * sData[0] + beta * result[rowX * strideR]; // Write the result for this row
}

//TODO: delete  this code
    // T sum = 0;

    // // Loop over diagonals
    // for (int k = 0; k < numDiags; k++) {
    //     int d = diags[k];
    //     int i = (d >= 0) ? rowX : rowX + d;
    //     if(0 <= i && i < height - fabs(d)) 
    //         sum += A[i*ld + k] * x[(rowX + d) * stride];
    // }

    // result[rowX] = sum;

// 
/**
 * Multiplies a sparse diagonal matrix (stored in packed diagonal format) with a 1D vector.
 * 
 * @param x Input vector.
 * @param result Optional pointer to an existing 1D array to store the result.
 * @param handle Optional Cuda handle for stream/context management.
 * @param diags Array of diagonal indices (negative=sub-diagonal, 0=main, positive=super-diagonal)
 * @param stride Stride for input vector x
 * @return A new CuArray1D containing the result.
 */
template <typename T>
CuArray1D<T> CuArray2D<T>::diagMult(
    const int* diags,
    const CuArray1D<T>& x,
    CuArray1D<T>* result,
    Handle* handle,
    const T alpha,
    const T beta    
) const {
    if (this->_rows > 64)
        throw std::invalid_argument("height must be <= 32 for this kernel");

    CuArray1D<T>* resPtr = result ? result : new CuArray1D<T>(this->_cols);
    Handle* h = handle ? handle : new Handle();

    CuArray1D<int> d_diags(this->_rows);
    d_diags.set(diags, h->stream);
    
    int sharedMemSize = sizeof(T) * this->_rows;
    
    diagMatVecKernel<<<this->_cols, (this->_cols + 1) / 2, sharedMemSize, h->stream>>>(
        this->data(), this->_cols, this->getLD(),
        d_diags.data(), this->_rows,
        x.data(), x.getLD(),
        resPtr->data(),
        resPtr->getLD(),
        alpha, beta
    );

    CHECK_CUDA_ERROR(cudaGetLastError());

    if (!result) {
        CuArray1D<T> temp = *resPtr;
        if (!handle) CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete resPtr;
        if (!handle) delete h;
        return temp;
    }

    if (!handle) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        delete h;
    }

    return *resPtr;
}


template class CuArray2D<int>;
template class CuArray2D<float>;
template class CuArray2D<double>;