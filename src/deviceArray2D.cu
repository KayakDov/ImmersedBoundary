#include "deviceArrays.h"


template <typename T>
Mat<T> Mat<T>::mult(
    const Mat<T>& other,
    Mat<T>* result,
    Handle* handle,
    T alpha,
    T beta,
    bool transposeA,
    bool transposeB
) const {    
    std::unique_ptr<Mat<T>> temp_res_ptr;
    Mat<T>* resPtr = _get_or_create_target(this->_rows, other._cols, result, temp_res_ptr);        
    
    gpuArray<T>::mult(other, resPtr, handle, alpha, beta, transposeA, transposeB);
    
    return *resPtr;
}


template <typename T>
Vec<T> Mat<T>::mult(
    const Vec<T>& other,
    Vec<T>* result,
    Handle* handle,
    T alpha,
    T beta,
    bool transpose
    
) const {
    
    std::unique_ptr<Vec<T>> temp_res_ptr;
    Vec<T>* resPtr = _get_or_create_target(this->_rows, result, temp_res_ptr);    
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    if constexpr(is_same_v<T, float>)
        cublasSgemv(h->handle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, this->_rows, this->_cols, &alpha, this->data(), this->getLD(), other.data(), other.getLD(), &beta, resPtr->data(), resPtr->getLD());
    else if constexpr(is_same_v<T, double>)
        cublasDgemv(h->handle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, this->_rows, this->_cols, &alpha, this->data(), this->getLD(), other.data(), other.getLD(), &beta, resPtr->data(), resPtr->getLD());
    else static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Vec::add unsupported type.");
        
    return *resPtr;
}


template <typename T>
Mat<T> Mat<T>::operator*(const Mat<T>& other) const {
    return this->mult(other);
}

template <typename T>
Vec<T> Mat<T>::operator*(const Vec<T>& other) const {
    return this->mult(other);
}

template <typename T>
Mat<T>::Mat(size_t rows, size_t cols): gpuArray<T>(rows, cols, 0) {
    void* rawPtr = nullptr;
    size_t pitch = 0;
    cudaError_t err = cudaMallocPitch(&rawPtr, &pitch, rows * sizeof(T), cols);
    if (err != cudaSuccess)
        throw std::runtime_error("cudaMallocPitch failed");

    this->_ptr = std::shared_ptr<void>(rawPtr, cudaFreeDeleter);
    this->_ld = pitch / sizeof(T);
}

template <typename T>
Mat<T>::Mat(const Mat<T>& superArray, size_t startRow, size_t startCol, size_t height, size_t width)
    : gpuArray<T>(height, width, superArray.getLD()) {
    size_t offset = startCol * superArray.getLD() + startRow;
    this->_ptr = std::shared_ptr<void>(
        superArray._ptr,
        static_cast<char*>(superArray._ptr.get()) + offset * sizeof(T)
    );
}

template <typename T>
size_t Mat<T>::size() const {
    return this->_rows * this->_cols;
}

template <typename T>
size_t Mat<T>::bytes() const {
    return this->_cols * this->_ld * sizeof(T);
}

template <typename T>
void Mat<T>::set(const T* src, cudaStream_t stream) {
    cudaMemcpy2DAsync(
        this->_ptr.get(), this->_ld * sizeof(T),
        src, this->_rows * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyHostToDevice, stream
    );
}

template <typename T>
void Mat<T>::get(T* dst, cudaStream_t stream) const {
    cudaMemcpy2DAsync(
        dst, this->_rows * sizeof(T),
        this->_ptr.get(), this->_ld * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToHost, stream
    );
}

template <typename T>
void Mat<T>::set(const GpuArray<T>& src, cudaStream_t stream) {
    cudaMemcpy2DAsync(
        this->_ptr.get(), this->_ld * sizeof(T),
        src.data(), src.getLD() * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToDevice, stream
    );
}

template <typename T>
void Mat<T>::get(GpuArray<T>& dst, cudaStream_t cuStream) const {
    cudaMemcpy2DAsync(
        dst.data(), dst.getLD() * sizeof(T),
        this->_ptr.get(), this->_ld * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToDevice, cuStream
    );
}

/**
 * Note, if set from text will read data as row major and is much slower.  If set from binary data, will read as column major and is fast.
 */
template <typename T>
void Mat<T>::set(std::istream& input_stream, bool isText, bool readColMajor, cudaStream_t cuStream) {

    if(!readColMajor){
        Mat<T> temp(this->_cols, this->_rows);
        temp.set(input_stream, isText, true, cuStream);
        Handle hand(cuStream);
        temp.transpose(*this, &hand);
        return;
    }

    StreamSet<T> helper(this->_rows, this->_cols, input_stream);

    while (helper.hasNext()) {
        helper.readChunk(isText);//This will either be a set of columns or a set of rows.
        Mat<T> subArray(
            *this,
            0,
            helper.getColsProcessed(),
            this->_rows,
            helper.getChunkWidth()
        );

        subArray.set(helper.getBuffer().data(), cuStream);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());//TODO: this might be avoidable with multi threading

        helper.updateProgress();
    }
}

/**
 * Note, if gets to text, will print data as row major and is much slower.  If gets to binary data, will write as column major and is fast.
 */
template <typename T>
void Mat<T>::get(std::ostream& output_stream, bool isText, bool printColMajor, cudaStream_t stream) const {

    StreamGet<T> helper(this->_rows, this->_cols, output_stream);
    
    if(!printColMajor) {
        Mat<T> mat(this->_rows, this->_cols);
        Handle handle(stream);
        this -> transpose(mat, &handle);
        mat.get(output_stream, isText, true, stream);
        return;        
    }
    
    while (helper.hasNext()) {
        Mat<T> subArray(
            *this,
            0,
            helper.getColsProcessed(),
            this->_rows,
            helper.getChunkWidth()
        );

        subArray.get(helper.getBuffer().data(), stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());//TODO: this might be avoidable with multi threading

        helper.writeChunk(isText);
        helper.updateProgress();
    }
}

template <typename T>
Mat<T> Mat<T>::plus(
    const Mat<T>& x, 
    Mat<T>* result,
    T alpha,
    T beta,
    bool transposeA,
    bool transposeB,
    Handle* handle
) {
    if (this->_rows != x._rows || this->_cols != x._cols)
        throw std::invalid_argument("Matrix dimensions do not match for add.");
    
    std::unique_ptr<Mat<T>> temp_res_ptr;
    Mat<T>* resPtr = _get_or_create_target(this->_rows, x._cols, result, temp_res_ptr);    
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    
    if constexpr (std::is_same_v<T, float>)
        cublasSgeam(
            h->handle, 
            transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
            transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
            this->_rows, this->_cols, 
            &alpha, x.data(), x.getLD(),
            &beta, this->data(), this->getLD(),
            resPtr->data(), resPtr->getLD()
        );
    else if constexpr (std:: is_same_v<T, double>)
        cublasDgeam(
            h->handle, 
            transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
            transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
            this->_rows, this->_cols, 
            &alpha, x.data(), x.getLD(),
            &beta, this->data(), this->getLD(),
            resPtr->data(), resPtr->getLD()
        );
    else static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Vec::add unsupported type.");

    return *resPtr;
}


template <typename T>
Mat<T> Mat<T>::minus(
    const Mat<T>& x,
    Mat<T>* result,
    T alpha,
    T beta,
    bool transposeA,
    bool transposeB,
    Handle* handle
) {
    return this->plus(x, result, -alpha, beta, transposeA, transposeB, handle);
}

template <typename T>
__global__ void scaleKernel(T* matrix, size_t rows, size_t cols, size_t ld, const T alpha) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) matrix[row + col * ld] *= alpha;
    
}

template <typename T>
void Mat<T>::_scale_impl(T alpha, Handle* handle) {
    if (this->_rows == 0 || this->_cols == 0) return;
    
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

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
}

template <typename T>
void Mat<T>::mult(T alpha, Handle* handle) {
    this->_scale_impl(alpha, handle);
}

/**
 * Multiplies a banded 2D matrix with a 1D vector using cuBLAS gbmv.
 */
template <typename T>
Vec<T> Mat<T>::bandedMult(
    const Vec<T>& x,
    Vec<T>* result,
    Handle* handle,
    T alpha,
    T beta,
    int kl,   // number of sub-diagonals
    int ku,   // number of super-diagonals
    bool transpose
) const {
    if ((transpose && this->_rows != x._cols) ||
        (!transpose && this->_cols != x._cols)) 
        throw std::invalid_argument("Matrix/vector dimensions do not match for banded multiplication.");    

    std::unique_ptr<Vec<T>> temp_res_ptr;
    Vec<T>* resPtr = _get_or_create_target(this->_rows, result, temp_res_ptr);    
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    
    if constexpr (std::is_same_v<T, float>)
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
    else if constexpr (std::is_same_v<T, double>)
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
    else static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Vec::add unsupported type.");

    return *resPtr;
}


/**
 * Kernel for sparse diagonal matrix-vector multiplication.
 * 
 * When calling this kernel, <<<numberOfBlocks, threadsPerBlock, sharedMemorySize, stream>>>,  
 * Number of blocks should be the number of rows in the solution vector.  
 * Threads per block should be 32.  
 * Shared memory size should be sizeof(T) * 32.
 * 
 * @param A Packed diagonals of the matrix.  Trailing values are not read.  Each row is a diagonal, and the matrix is stored in column-major order.  There may be no more than 32 rows.
 * @param height Number of rows in the greater matrix, not the packed matrix.  This is also the number of columns in the greater matrix and the number of rows in x.
 * @param ld Leading dimension of the packed diagonals.
 * @param diags Indices of the diagonals.  Negative indices indicate sub-diagonals. 
 * Positive indices indicate super-diagonals. 
 * For example, diags = {-1, 0, 1} means the first diagonal is the sub-diagonal, the second is the main diagonal, and the third is the super-diagonal.
 * @param numDiags Number of nonzero diagonals.  This number must be less than or equal to 32.
 * @param x Input vector.
 * @param stride Stride for the input vector x.
 * @param result Output vector.
 * @param strideR Stride for the output vector result.
 * @param alpha Scalar multiplier for the matrix-vector product.
 * @param beta Scalar multiplier for the existing values in the result vector.
 */
template <typename T>
__global__ void diag32MatVecKernel(
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
    int rowX = blockIdx.x;
    int rowA = threadIdx.x;
    bool isValid = rowX < height && rowA < numDiags;
    T val;
    if (isValid){//TODO: this condition can be removed by requiring input matrices have exactly 32 rows, with extra rows having all 0's.  This may give a small speed boost.       
        int d = diags[rowA], colA = rowX;
        if(d < 0) colA += d;
        val = 0 <= colA && colA < height - abs(d) ? A[colA*ld + rowA] * x[(rowX + d) * strideX] : 0;
    } else val = 0;
    
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    
    int indR = rowX * strideR;

    if(isValid && rowA == 0) result[indR] = alpha * val + beta * result[indR];
}

/**
 * Multiplies a sparse diagonal matrix (stored in packed diagonal format) with a 1D vector.
 * 
 * @param x Input vector.
 * @param result Optional pointer to an existing 1D array to store the result.
 * @param handle Optional Cuda handle for stream/context management.
 * @param diags Array of diagonal indices (negative=sub-diagonal, 0=main, positive=super-diagonal)
 * @param stride Stride for input vector x
 * @return A new CuArray1D containing the result.
 * 
 */
template <typename T>
Vec<T> Mat<T>::diagMult(
    const Vec<int>& diags,
    const Vec<T>& x,
    Vec<T>* result,
    Handle* handle,
    const T alpha,
    const T beta    
) const {
    if (this->_rows > 32)
        throw std::invalid_argument("height must be <= 32 for this kernel");

    std::unique_ptr<Vec<T>> temp_res_ptr;
    Vec<T>* resPtr = _get_or_create_target(this->_rows, x._cols, result, temp_res_ptr);
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    
    diag32MatVecKernel<<<this->_cols, 32, 0, h->stream>>>(
        this->data(), this->_cols, this->getLD(),
        diags.data(), this->_rows,
        x.data(), x.getLD(),
        resPtr->data(),
        resPtr->getLD(),
        alpha, beta
    );

    CHECK_CUDA_ERROR(cudaGetLastError());

    return *resPtr;
}



/**
 * Transposes the matrix. This method creates and returns a new CuArray2D
 * object with transposed dimensions.
 * @param result Optional pointer to an existing CuArray2D to store the result.
 * @param handle Optional Cuda handle for stream/context management.
 * @return A new CuArray2D object containing the transposed matrix.
 */
template <typename T>
void Mat<T>::transpose(
    Mat<T>& result,
    Handle* handle
) const {    
    
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgeam(
            h->handle, 
            CUBLAS_OP_T, // Transpose A
            CUBLAS_OP_N, // Don't transpose B (it's not used)
            this->_cols, // Result rows
            this->_rows, // Result columns
            &alpha, 
            this->data(), this->getLD(),
            &beta, nullptr, this->getLD(), // B is not referenced since beta=0
            result.data(), result.getLD()
        );
    } else if constexpr (std::is_same_v<T, double>) {
        double alpha = 1.0;
        double beta = 0.0;
        cublasDgeam(
            h->handle, 
            CUBLAS_OP_T, 
            CUBLAS_OP_N,
            this->_cols,
            this->_rows,
            &alpha, 
            this->data(), this->getLD(),
            &beta, nullptr, this->getLD(),
            result.data(), result.getLD()
        );
    }
}

/**
 * @brief Performs an in-place transpose of the matrix. This method modifies the
 * existing CuArray2D object by creating and using a temporary buffer.
 *
 * @param temp Optional pre-allocated temporary matrix to use for the transpose operation.  It should be the same size as this matrix.
 * If nullptr, a new temporary matrix will be created.
 * @param handle Optional Cuda handle for stream/context management.
 */
template <typename T>
void Mat<T>::transpose(Handle* handle, Mat<T>* temp) {
    if (this->_rows == 0 || this->_cols == 0) return;
    if(this->_rows != this->_cols)
        throw std::runtime_error("In-place transpose is only supported for square matrices. For non-square matrices, use the out-of-place version.");
    
    std::unique_ptr<Mat<T>> temp_res_ptr;
    Mat<T>* temp_ptr = _get_or_create_target(this->_cols, this->_rows, temp, temp_res_ptr);
    
    if (temp_ptr->_rows != this->_cols || temp_ptr->_cols != this->_rows)
        throw std::invalid_argument("Provided temporary matrix has incorrect dimensions for transpose.");
    
    this->transpose(*temp_ptr, handle);

    this->set(*temp_ptr);
}


template class Mat<int>;
template class Mat<float>;
template class Mat<double>;