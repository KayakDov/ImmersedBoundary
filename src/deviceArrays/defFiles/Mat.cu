#include "../headers/Vec.h"
#include "../headers/deviceArraySupport.h"
#include "../headers/Singleton.h"

#include <cusolverDn.h>
#include <stdexcept>
#include "../headers/Mat.h"

#include <iostream>

#include "../headers/GridDim.hpp"

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
    Mat<T>* resPtr = Mat<T>::_get_or_create_target(this->_rows, other._cols, result, temp_res_ptrMat);
    std::unique_ptr<Singleton<T>> temp_a_ptrSing;
    const Singleton<T>* a = Singleton<T>::_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptrSing);
    std::unique_ptr<Singleton<T>> temp_b_ptrSing2;
    const Singleton<T>* b = Singleton<T>::_get_or_create_target(static_cast<T>(0), *h, beta, temp_b_ptrSing2);
    
    GpuArray<T>::mult(other, resPtr, h, a, b, transposeA, transposeB);
    
    return *resPtr;
}

template<typename T>
Mat<T> Mat<T>::mult(const Mat<T> &other, Mat<T> *result, Handle *handle, bool transposeA, bool transposeB) const {
    return mult(other, result, handle, &Singleton<T>::ONE, &Singleton<T>::ZERO, transposeA, transposeB);
}


template <typename T>
void Mat<T>::mult(
    const Vec<T> &other,
    Vec<T> &result,
    Handle *handle,
    const Singleton<T> *alpha,
    const Singleton<T> *beta,
    bool transpose

) const {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Singleton<T>> temp_a_ptr;
    const Singleton<T>* a = Singleton<T>::_get_or_create_target(1, *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* b = Singleton<T>::_get_or_create_target(0, *h, beta, temp_b_ptr);

    if constexpr(std::is_same_v<T, float>)
        cublasSgemv(*h, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, this->_rows, this->_cols, a->toKernel1d(), this->toKernel2d(), this->_ld, other.toKernel1d(), other._ld, b->toKernel1d(), result.toKernel1d(), result._ld);
    else if constexpr(std::is_same_v<T, double>)
        cublasDgemv(*h, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, this->_rows, this->_cols, a->toKernel1d(), this->toKernel2d(), this->_ld, other.toKernel1d(), other._ld, b->toKernel1d(), result.toKernel1d(), result._ld);
    else throw std::invalid_argument("Unsupported type.");
}

template <typename T>
Mat<T> Mat<T>::operator*(const Mat<T>& other) const {
    return this->mult(other, nullptr, nullptr, nullptr, nullptr, false, false);
}

template <typename T>
Vec<T> Mat<T>::operator*(const Vec<T>& other) const {
    Vec<T> result = Vec<T>::create(this->_rows, nullptr);
    this->mult(other, result, nullptr, nullptr, nullptr, false);
    return result;
}



template <typename T>
Mat<T>* Mat<T>::_get_or_create_target(const size_t rows, const size_t cols, Mat<T>* result, std::unique_ptr<Mat<T>>& out_ptr_unique) {
    if (result) return result;
    else {
        out_ptr_unique = std::make_unique<Mat<T>>(Mat<T>::create(rows, cols));

        return out_ptr_unique.get();
    }
}

template<typename T>
KernelPrep Mat<T>::kernelPrep(bool t) const {
    return GpuArray<T>::kernelPrep(t);
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
        this->data(), this->_ld * sizeof(T),
        src.data(), src._ld * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToDevice, stream
    );
}

template <typename T>
void Mat<T>::get(GpuArray<T>& dst, cudaStream_t cuStream) const {
    cudaMemcpy2DAsync(
        dst.data(), dst._ld * sizeof(T),
        this->data(), this->_ld * sizeof(T),
        this->_rows * sizeof(T), this->_cols,
        cudaMemcpyDeviceToDevice, cuStream
    );
}

/**
 * Note, if set from text will read data as row major and is much slower.  If set from binary data, will read as column major and is fast.
 */
template <typename T>
void Mat<T>::set(std::istream& input_stream, bool isText, bool isColMjr, Handle* hand) {

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(hand, temp_hand_ptr);

    if(!isColMjr){
        Mat<T> temp = Mat<T>::create(this->_cols, this->_rows);
        temp.set(input_stream, isText, !isColMjr, h);
        temp.transpose(*this, h);
        return;
    }

    StreamSet<T> helper(this->_rows, this->_cols, input_stream);

    while (helper.hasNext()) {
        helper.readChunk(isText);//This will either be a set of columns or a set of rows.
        Mat<T> subMat = this->Mat<T>::subMat(
            0,
            helper.getColsProcessed(),
            this->_rows,
            helper.getChunkWidth()
        );

        subMat.set(helper.getBuffer().data(), *h);

        h->synch();//TODO: this might be avoidable with multi threading

        helper.updateProgress();
    }
}

/**
 *TODO: rewrite this so that it returns an object that holds the paramaters, and we can call ostream << objectThatHoldsParamaters for smoother code.  Also for istream.
 * Note, if gets to text, will print data as row major and is much slower.  If gets to binary data, will write as column major and is fast.
 */
template <typename T>
std::ostream &Mat<T>::get(std::ostream &output_stream, bool isText, bool printColMajor, Handle& hand) const {

    StreamGet<T> helper(this->_rows, this->_cols, output_stream);

    if(!printColMajor) {
        auto transposed = Mat<T>::create(this->_cols, this->_rows);
        this -> transpose(transposed, &hand);
        transposed.get(output_stream, isText, true, hand);
        return output_stream;
    }
    
    while (helper.hasNext()) {
        Mat<T> subMat = this->subMat(
            0,
            helper.getColsProcessed(),
            this->_rows,
            helper.getChunkWidth()
        );

        subMat.get(helper.getBuffer().data(), hand);
        hand.synch();//TODO: this might be avoidable with multi threading

        helper.writeChunk(isText);
        helper.updateProgress();
    }
    return output_stream;
}

template<typename T>
Singleton<T> Mat<T>::get(size_t row, size_t col) {
    return Singleton<T>(std::shared_ptr<T>(this->_ptr, this->_ptr.get() + col * this->_ld + row));
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
    Mat<T>* resPtr = Mat<T>::_get_or_create_target(this->_rows, x._cols, result, temp_res_ptr);
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);
    std::unique_ptr<Singleton<T>> temp_a_ptr;
    const Singleton<T>* a = Singleton<T>::_get_or_create_target(1, *h , alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* b = Singleton<T>::_get_or_create_target(0, *h, beta, temp_b_ptr);
    
    if constexpr (std::is_same_v<T, float>)
        CHECK_CUBLAS_ERROR(cublasSgeam(
            *h,
            transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
            transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
            this->_rows, this->_cols, 
            a->toKernel1d(), x.toKernel2d(), x._ld,
            b->toKernel1d(), this->toKernel2d(), this->_ld,
            resPtr->toKernel2d(), resPtr->_ld
        ));
    else if constexpr (std:: is_same_v<T, double>)
        CHECK_CUBLAS_ERROR(cublasDgeam(
            *h,
            transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
            transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
            this->_rows, this->_cols, 
            a->toKernel1d(), x.toKernel2d(), x._ld,
            b->toKernel1d(), this->toKernel2d(), this->_ld,
            resPtr->toKernel2d(), resPtr->_ld
        ));
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
    const Singleton<T>* a = Singleton<T>::_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T>> temp_b_ptr;
    const Singleton<T>* b = Singleton<T>::_get_or_create_target(beta ? -(beta->get()) : static_cast<T>(-1), *h, beta, temp_b_ptr);

    return this->plus(x, result, a, b, transposeA, transposeB, h);
}

template <typename T>
__global__ void scaleKernel(DeviceData2d<T>  matrix, const T* alpha) {
    if (GridInd2d ind; ind < matrix) matrix[ind] *= *alpha;
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

    scaleKernel<<<numBlocks, threadsPerBlock, 0, *h>>>(
        this->toKernel2d(),
        alpha.toKernel1d().data
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

        CHECK_CUBLAS_ERROR(cublasSgeam(
            *h,
            CUBLAS_OP_T, // Transpose A
            CUBLAS_OP_N, // Don't transpose B (it's not used)
            this->_cols, // Result rows
            this->_rows, // Result columns
            Singleton<T>::ONE.toKernel1d(),
            this->toKernel2d(), this->_ld,
            Singleton<T>::ZERO.toKernel1d(), nullptr, this->_ld, // B is not referenced since beta=0
            result.toKernel2d(), result._ld
        ));
    } else if constexpr (std::is_same_v<T, double>) {
        CHECK_CUBLAS_ERROR(cublasDgeam(
            *h,
            CUBLAS_OP_T, 
            CUBLAS_OP_N,
            this->_cols,
            this->_rows,
            Singleton<T>::ONE.data(),
            this->data(), this->_ld,
            Singleton<T>::ZERO.data(), nullptr, this->_ld,
            result.data(), result._ld
        ));
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

    this->set(*temp_ptr, *handle);
}

template <typename T>
Mat<T> Mat<T>::create(size_t rows, size_t cols, bool initDescr){
    T* rawPtr = nullptr;
    size_t pitch = 0;

    CHECK_CUDA_ERROR(cudaMallocPitch(&rawPtr, &pitch, rows * sizeof(T), cols));//Note: there does not seem to be an asynchronos version of this method.

    auto result = Mat<T>(rows, cols, pitch / sizeof(T), std::shared_ptr<T>(rawPtr, cudaFreeDeleter));

    if (initDescr) result.initDescr();

    return result;
}

template<typename T>
Mat<T> Mat<T>::create(size_t rows, size_t cols, const size_t ld, T *devicePointer) {
    return Mat<T>(rows, cols, ld, nonOwningGpuPtr(devicePointer));
}

template <typename T>
std::shared_ptr<T> Mat<T>::offset(size_t row, size_t col) {
    return std::shared_ptr<T>(this->_ptr, const_cast<T*>(this->_ptr.get() + col * this->_ld + row));
}

template <typename T>
std::shared_ptr<T> Mat<T>::offset(size_t row, size_t col) const{
    return std::shared_ptr<T>(this->_ptr,this->_ptr.get() + col * this->_ld + row);
}

template <typename T>
Mat<T> Mat<T>::subMat(const size_t startRow, const size_t startCol, const size_t height, const size_t width) const{

    return Mat<T>(
        height,
        width,
        this->_ld,
        offset(startRow, startCol)
    );
}

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
    DeviceData2d<T> A,
    const size_t normalizeByRow
) {

    if (const GridInd2d ind; ind < A)
        if (const T val = A(normalizeByRow, ind.col); val != 0) A[ind] *= ( static_cast<T>(1) / val );

}

template <typename T>
void Mat<T>::normalizeCols(size_t setRowTo1, Handle* handle) {
    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    KernelPrep kp = this->kernelPrep();
    normalizeByRowKernel<T><<<kp.numBlocks, kp.threadsPerBlock, 0, *h>>>(this->toKernel2d(), setRowTo1);
}

template<typename T>
void Mat<T>::batchMult(
    const Mat<T>& a1,
    const size_t strideA, const Mat<T>& b1,
    const size_t strideB, Mat<T>& c1,
    const size_t strideC,
    const bool transposeA, const bool transposeB,
    Handle& hand, const size_t batchCount,
    const Singleton<T>& alpha, const Singleton<T>& beta
) {

    const size_t m = transposeA ? a1._cols : a1._rows,
        n = transposeB ? b1._rows : b1._cols,
        k = transposeA ? a1._rows : a1._cols;

    cublasOperation_t transA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
        transB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    if constexpr (std::is_same_v<T, float>) {
        CHECK_CUBLAS_ERROR(cublasSgemmStridedBatched(hand, transA, transB, m, n, k,
            alpha.toKernel1d(), a1.toKernel2d(), a1._ld, strideA,
            b1.toKernel2d(), b1._ld, strideB, beta.toKernel1d(),
            c1.toKernel2d(), c1._ld, strideC, batchCount));
    }

    else if constexpr (std:: is_same_v<T, double>){
        CHECK_CUBLAS_ERROR(cublasDgemmStridedBatched(hand, transA, transB, m, n, k,
            alpha.toKernel1d(), a1.toKernel2d(), a1._ld, strideA,
            b1.toKernel2d(), b1._ld, strideB, beta.toKernel1d(),
            c1.toKernel2d(), c1._ld, strideC, batchCount));
    }
    else throw std::invalid_argument("Unsupported type.");
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
    const Singleton<T> *a = Singleton<T>::_get_or_create_target(static_cast<T>(1), *h, alpha, temp_a_ptr);
    std::unique_ptr<Singleton<T> > temp_b_ptr;
    const Singleton<T> *b = Singleton<T>::_get_or_create_target(static_cast<T>(0), *h, beta, temp_b_ptr);

    other.mult(*this, result, handle, a, b, !transpose);
}


template<typename T>
GpuArray<T>::operator DeviceData2d<T>() {
    return toKernel2d();
}

template<typename T>
GpuArray<T>::operator DeviceData2d<T>() const {
    return toKernel2d();
}

//TODO: these methods are in the parent class as private and copied here as public.  This is redundant code.  Sort it out.  It's private in parent class so that Vec and Tensor don't accidently use it.  But we need the code here and in dependents of this class.
template <typename T>
DeviceData2d<T> Mat<T>::toKernel2d() { return DeviceData2d<T>(this->_rows, this->_cols, this->_ld, this->_ptr.get()); }
template <typename T>
DeviceData2d<T> Mat<T>::toKernel2d() const { return DeviceData2d<T>(this->_rows, this->_cols, this->_ld, this->_ptr.get()); }

template<typename T>
void Mat<T>::factorLU(Handle *hand, Vec<int32_t> *rowSwaps, Singleton<int32_t> *info, Vec<T> *workSpace) {

    std::unique_ptr<Handle> temp_hand_ptr;
    auto h = Handle::_get_or_create_handle(hand, temp_hand_ptr);


    int32_t lwork = workSpace?workSpace->size():0;

    if (!workSpace) {
        if constexpr (std::is_same_v<T, double>)
            CHECK_CUSOLVER_ERROR(cusolverDnDgetrf_bufferSize(*h, this->_rows, this->_cols, this->toKernel2d(), this->_ld, &lwork));
        else if constexpr (std::is_same_v<T, float>)
            CHECK_CUSOLVER_ERROR(cusolverDnSgetrf_bufferSize(*h, this->_rows, this->_cols, this->toKernel2d(), this->_ld, &lwork));
        else throw std::invalid_argument("Unsupported type.");
    }

    std::unique_ptr<Vec<T>> temp_workSpace_ptr;
    auto ws = Vec<T>::_get_or_create_target( lwork, workSpace, temp_workSpace_ptr, *h);

    std::unique_ptr<Vec<int32_t>> temp_rr;
    auto rr = Vec<int32_t>::_get_or_create_target(this->_rows, rowSwaps, temp_rr, *h);

    std::unique_ptr<Singleton<int32_t>> temp_info;
    auto inf = Singleton<int32_t>::_get_or_create_target(info, temp_info, *h);

    if constexpr (std::is_same_v<T, double>)
        CHECK_CUSOLVER_ERROR(cusolverDnDgetrf(*h, this->_rows, this->_cols, this->data(), this->_ld, ws->data(), rr->data(), inf->data()));
    else if constexpr (std::is_same_v<T, float>)
        CHECK_CUSOLVER_ERROR(cusolverDnSgetrf(*h, this->_rows, this->_cols, this->data(), this->_ld, ws->data(), rr->data(), inf->data()));
}

template<typename T>
Vec<T>::operator Mat<T>() {
    return Mat<T>(this->_rows, this->_cols, this->_ld, this->_ptr);
}
template<typename T>
Vec<T>::operator Mat<T>() const{
    return Mat<T>(this->_rows, this->_cols, this->_ld, this->_ptr);
}

template<typename T>
void Mat<T>::initDescr() const{
    cusparseDnMatDescr_t rawDescr;
    const cudaDataType valueType = sizeof(T) == 8 ? CUDA_R_64F : CUDA_R_32F;;

    CHECK_SPARSE_ERROR(cusparseCreateDnMat(
        &rawDescr,
        this->_rows,         // Number of rows
        this->_cols,         // Number of columns
        this->_ld,           // Leading dimension (can be > rows)
        const_cast<void*>(static_cast<const void*>(this->data())),
        valueType,
        CUSPARSE_ORDER_COL    // Change to CUSPARSE_ORDER_ROW if C-style
    ));

    // Use the same smart pointer pattern as your SimpleArray
    dnMatDescr = DnMatDescrPtr(rawDescr, [](const cusparseDnMatDescr_t p) {
        if (p) cusparseDestroyDnMat(p);
    });
}

template<typename T>
cusparseDnMatDescr_t Mat<T>::getDescr() const {
    if (!dnMatDescr) initDescr();

    return dnMatDescr.get();
}

template class Mat<float>;
template class Mat<double>;
template class Mat<size_t>;
template class Mat<int32_t>;
template class Mat<uint32_t>;
template class Mat<unsigned char>;

template void Vec<float>::mult(const Mat<float>&, Vec<float>&, Handle*, const Singleton<float>*, const Singleton<float>*, bool) const;
template void Vec<double>::mult(const Mat<double>&, Vec<double>&, Handle*, const Singleton<double>*, const Singleton<double>*, bool) const;
template void Vec<size_t>::mult(const Mat<size_t>&, Vec<size_t>&, Handle*, const Singleton<size_t>*, const Singleton<size_t>*, bool) const;
template void Vec<int32_t>::mult(const Mat<int32_t>&, Vec<int32_t>&, Handle*, const Singleton<int32_t>*, const Singleton<int32_t>*, bool) const;
template void Vec<unsigned char>::mult(const Mat<unsigned char>&, Vec<unsigned char>&, Handle*, const Singleton<unsigned char>*, const Singleton<unsigned char>*, bool) const;
template void Vec<uint32_t>::mult(const Mat<uint32_t>&, Vec<uint32_t>&, Handle*, const Singleton<uint32_t>*, const Singleton<uint32_t>*, bool) const;

template Vec<float>::operator Mat<float>();
template Vec<double>::operator Mat<double>();
template Vec<size_t>::operator Mat<size_t>();
template Vec<int32_t>::operator Mat<int32_t>();
template Vec<unsigned char>::operator Mat<unsigned char>();
template Vec<uint32_t>::operator Mat<uint32_t>();

template Vec<float>::operator Mat<float>() const;
template Vec<double>::operator Mat<double>() const;
template Vec<size_t>::operator Mat<size_t>() const;
template Vec<int32_t>::operator Mat<int32_t>() const;
template Vec<unsigned char>::operator Mat<unsigned char>() const;
template Vec<uint32_t>::operator Mat<uint32_t>() const;