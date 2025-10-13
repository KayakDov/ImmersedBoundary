#include "../headers/vec.h"
#include "../headers/deviceArraySupport.h"
#include "../headers/singleton.h"

#include <cusolverDn.h>
#include <stdexcept>
#include "../headers/Mat.h"

#include <iostream>

template <typename T>
Mat<T> Mat<T>::mult(
    const Mat<T>& other,
    Mat<T>* result,
    Handle* handle,
    const Singleton<T> *alpha,
    const Singleton<T> *beta,
    bool transposeA,
    bool transposeB
) const {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Mat<T>> temp_res_ptrMat;
    Mat<T>* resPtr = this->_get_or_create_target(this->_rows, other._cols, result, temp_res_ptrMat);
    std::unique_ptr<Singleton<T>> temp_a_ptrSing;
    const Singleton<T>* a = this->_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptrSing);
    std::unique_ptr<Singleton<T>> temp_b_ptrSing2;
    const Singleton<T>* b = this->_get_or_create_target(static_cast<T>(0), *h, beta, temp_b_ptrSing2);
    
    GpuArray<T>::mult(other, resPtr, h, a, b, transposeA, transposeB);
    
    return *resPtr;
}


template <typename T>
Vec<T> Mat<T>::mult(
    const Vec<T>& other,
    Vec<T>* result,
    Handle* handle,
    const Singleton<T> *alpha,
    const Singleton<T> *beta,
    bool transpose

) const {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Vec<T>> temp_res_ptr;
    Vec<T>* resPtr = this->_get_or_create_target(this->_rows, result, temp_res_ptr, h->stream);
    std::unique_ptr<Singleton<T>> temp_a_ptr;
    const Singleton<T>* a = this->_get_or_create_target(1, *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* b = this->_get_or_create_target(0, *h, beta, temp_b_ptr);

    if constexpr(std::is_same_v<T, float>)
        cublasSgemv(h->handle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, this->_rows, this->_cols, a->data(), this->data(), this->_ld, other.data(), other._ld, b->data(), resPtr->data(), resPtr->_ld);
    else if constexpr(std::is_same_v<T, double>)
        cublasDgemv(h->handle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, this->_rows, this->_cols, a->data(), this->data(), this->_ld, other.data(), other._ld, b->data(), resPtr->data(), resPtr->_ld);
    else throw std::invalid_argument("Unsupported type.");
        
    return *resPtr;
}

template <typename T>
Mat<T> Mat<T>::operator*(const Mat<T>& other) const {
    return this->mult(other);
}

template <typename T>
Vec<T> Mat<T>::operator*(const Vec<T>& other) const {
    return this->mult(other, nullptr, nullptr, nullptr, nullptr, false);
}

template<typename T>
Mat<T>::Mat(size_t rows, size_t cols, size_t ld, std::shared_ptr<T> _ptr): GpuArray<T>(rows, cols, ld, _ptr) {
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
void Mat<T>::get(T* dst, const cudaStream_t stream) const {
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
        src.data(), src._ld * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToDevice, stream
    );
}

template <typename T>
void Mat<T>::get(GpuArray<T>& dst, cudaStream_t cuStream) const {
    cudaMemcpy2DAsync(
        dst.data(), dst._ld * sizeof(T),
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

    Handle hand(cuStream);
    if(!readColMajor){
        Mat<T> temp = Mat<T>::create(this->_cols, this->_rows);
        temp.set(input_stream, isText, true, cuStream);

        temp.transpose(*this, &hand);
        return;
    }

    StreamSet<T> helper(this->_rows, this->_cols, input_stream);

    while (helper.hasNext()) {
        helper.readChunk(isText);//This will either be a set of columns or a set of rows.
        Mat<T> subMat = this->subMat(
            0,
            helper.getColsProcessed(),
            this->_rows,
            helper.getChunkWidth()
        );

        subMat.set(helper.getBuffer().data(), cuStream);

        hand.synch();//TODO: this might be avoidable with multi threading

        helper.updateProgress();
    }
}

/**
 * Note, if gets to text, will print data as row major and is much slower.  If gets to binary data, will write as column major and is fast.
 */
template <typename T>
void Mat<T>::get(std::ostream& output_stream, bool isText, bool printColMajor, cudaStream_t stream) const {

    StreamGet<T> helper(this->_rows, this->_cols, output_stream);
    Handle handle(stream);

    if(!printColMajor) {
        Mat<T> mat = Mat<T>::create(this->_cols, this->_rows);
        this -> transpose(mat, &handle);
        mat.get(output_stream, isText, true, stream);
        return;        
    }
    
    while (helper.hasNext()) {
        Mat<T> subMat = this->subMat(
            0,
            helper.getColsProcessed(),
            this->_rows,
            helper.getChunkWidth()
        );

        subMat.get(helper.getBuffer().data(), stream);
        handle.synch();//TODO: this might be avoidable with multi threading

        helper.writeChunk(isText);
        helper.updateProgress();
    }
}

template <typename T>
Mat<T> Mat<T>::plus(
    const Mat<T>& x, 
    Mat<T>* result,
    const Singleton<T>* alpha,
    const Singleton<T>* beta,
    bool transposeA,
    bool transposeB,
    Handle* handle
) {
    if (this->_rows != x._rows || this->_cols != x._cols)
        throw std::invalid_argument("Matrix dimensions do not match for add.");
    
    std::unique_ptr<Mat<T>> temp_res_ptr;
    Mat<T>* resPtr = this->_get_or_create_target(this->_rows, x._cols, result, temp_res_ptr);
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Singleton<T>> temp_a_ptr;
    const Singleton<T>* a = this->_get_or_create_target(1, *h , alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* b = this->_get_or_create_target(0, *h, beta, temp_b_ptr);
    
    if constexpr (std::is_same_v<T, float>)
        cublasSgeam(
            h->handle, 
            transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
            transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
            this->_rows, this->_cols, 
            a->data(), x.data(), x._ld,
            b->data(), this->data(), this->_ld,
            resPtr->data(), resPtr->_ld
        );
    else if constexpr (std:: is_same_v<T, double>)
        cublasDgeam(
            h->handle, 
            transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
            transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
            this->_rows, this->_cols, 
            a->data(), x.data(), x._ld,
            b->data(), this->data(), this->_ld,
            resPtr->data(), resPtr->_ld
        );
    else throw std::invalid_argument("Unsupported type.");

    return *resPtr;
}


template <typename T>
Mat<T> Mat<T>::minus(
    const Mat<T>& x,
    Mat<T>* result,
    const Singleton<T>* alpha,
    const Singleton<T>* beta,
    bool transposeA,
    bool transposeB,
    Handle* handle
) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Singleton<T>> temp_a_ptr;
    const Singleton<T>* a = this->_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* b = this->_get_or_create_target(beta ? -(beta->get()) : static_cast<T>(-1), *h, beta, temp_b_ptr);




    return this->plus(x, result, a, b, transposeA, transposeB, h);
}

template <typename T>
__global__ void scaleKernel(T* matrix, size_t rows, size_t cols, size_t ld, const T* alpha) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) matrix[row + col * ld] *= *alpha;
    
}


template <typename T>
void Mat<T>::mult(const Singleton<T>& alpha, Handle* handle) {
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
        this->_ld,
        alpha.data()
    );
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
        Singleton<T> alpha = Singleton<T>::create(static_cast<T>(1), h->stream);
        Singleton<T> beta = Singleton<T>::create(static_cast<T>(0), h->stream);

        cublasSgeam(
            h->handle, 
            CUBLAS_OP_T, // Transpose A
            CUBLAS_OP_N, // Don't transpose B (it's not used)
            this->_cols, // Result rows
            this->_rows, // Result columns
            alpha.data(), 
            this->data(), this->_ld,
            beta.data(), nullptr, this->_ld, // B is not referenced since beta=0
            result.data(), result._ld
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
            this->data(), this->_ld,
            &beta, nullptr, this->_ld,
            result.data(), result._ld
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
    Mat<T>* temp_ptr = this->_get_or_create_target(this->_cols, this->_rows, temp, temp_res_ptr);
    
    if (temp_ptr->_rows != this->_cols || temp_ptr->_cols != this->_rows)
        throw std::invalid_argument("Provided temporary matrix has incorrect dimensions for transpose.");
    
    this->transpose(*temp_ptr, handle);

    this->set(*temp_ptr, handle->stream);
}

template <typename T>
Mat<T> Mat<T>::create(size_t rows, size_t cols){
    T* rawPtr = nullptr;
    size_t pitch = 0;

    CHECK_CUDA_ERROR(cudaMallocPitch(&rawPtr, &pitch, rows * sizeof(T), cols));//Note: there does not seem to be an asynchronos version of this method.

    return Mat<T>(rows, cols, pitch / sizeof(T), std::shared_ptr<T>(rawPtr, cudaFreeDeleter));;
}

template <typename T>
Mat<T> Mat<T>::subMat(const size_t startRow, const size_t startCol, const size_t height, const size_t width) const{

    return Mat<T>(
        height,
        width,
        this->_ld,
        std::shared_ptr<T>(this->_ptr, const_cast<T*>(this->data() + startCol * this->_ld + startRow))
    );
}


template <typename T>
Vec<T> Mat<T>::col(const size_t index){
    if (index >= this->_cols) throw std::out_of_range("Out of range");
    return Vec<T>(this->_rows, std::shared_ptr<T>(this->_ptr, this->data() + index * this->_ld), 1);
}
template <typename T>
Vec<T> Mat<T>::row(const size_t index){
    if (index > this->_rows) throw std::out_of_range("Out of range");
    return Vec<T>(this->_cols, std::shared_ptr<T>(this->_ptr, this->_ptr.get() + index), this->_ld);
}

// Assuming you have a standard Deleter for cudaFree
struct cudaFreeDeleter {
    void operator()(void* ptr) const {
        if (ptr) cudaFree(ptr);
    }
};


/**
 * This method multiplies each column by a constant so that the selected row has a 1 in it.
 * @tparam T  The type of data.
 * @param A The matrix to be normalized. This matrix is modified in-place.
 * @param normalizeByRow The row that will have a 1 in it after the operation.
 * @param height The height of the matrix.
 * @param width The width of the matrix.
 * @param ld The leading dimension of the matrix (typically the height).
 */
template <typename T>
__global__ void normalizeByRowKernel(
    T* __restrict__ A,
    const size_t normalizeByRow,
    const size_t height, const size_t width, const size_t ld
) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width){
        const size_t colInd = col * ld;
        const T val = A[colInd + normalizeByRow];
        if (val != 0) A[colInd + row] *= ( static_cast<T>(1) / val );
    }
}

template <typename T>
void Mat<T>::normalizeCols(size_t setRowTo1, Handle* handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    constexpr dim3 blockDim(16, 16);

    const dim3 gridDim(
        (this->_cols + blockDim.x - 1) / blockDim.x,
        (this->_rows + blockDim.y - 1) / blockDim.y
    );

    normalizeByRowKernel<T><<<gridDim, blockDim, 0, h->stream>>>(
        this->data(),
        setRowTo1,
        this->_rows,
        this ->_cols,
        this->_ld
    );
}

template class Mat<float>;
template class Mat<double>;
template class Mat<size_t>;
template class Mat<int32_t>;
template class Mat<unsigned char>;

