#include <cusolver_common.h>
#include <iostream>
#include <sstream>

#include "../headers/SquareMat.h"
#include "../headers/KernelSupport.cuh"
#include "../headers/deviceArraySupport.h"
#include "../headers/Singleton.h"
#include "../headers/Vec.h"
#include "../headers/sparse/BandedMat.h"
#include "../headers/Support/GridDim.hpp"


template<typename T>
SquareMat<T>::SquareMat(const size_t rowsCols, const size_t ld, std::shared_ptr<T> _ptr) :
    Mat<T>(rowsCols, rowsCols, ld, _ptr) {
}

template<typename T>
SquareMat<T> SquareMat<T>::create(size_t rowsCols) {
    Mat<T> mat = Mat<T>::create(rowsCols, rowsCols);
    return SquareMat<T>(rowsCols, mat._ld, mat.ptr());
}

template<typename T>
SquareMat<T> SquareMat<T>::create(size_t rowsCols, size_t ld, T *ptr) {
    return SquareMat<T>(rowsCols, ld, nonOwningGpuPtr(ptr));
}


/**
 * @brief Check cuSOLVER `info` result and throw if an error occurred.
 *
 * @param info_dev  Device-side info object returned from cusolverDnXgeev.
 * @param context   Optional string to add context to the exception message.
 *
 * @throws std::runtime_error if info != 0.
 */
inline void processInfo(const Singleton<int32_t>& info_dev,
                        const std::string& context = "cusolverDnXgeev")
{
    const int info_host = info_dev.get();

    if (info_host == 0) return;

    std::ostringstream msg;
    msg << "cuSOLVER error in " << context << ": info = " << info_host << ". ";

    switch (info_host) {
        case -1: msg << "Parameter 1 (handle) had an illegal value."; break;
        case -2: msg << "Parameter 2 (jobvl/jobvr) had an illegal value."; break;
        case -3: msg << "Parameter 3 (n, matrix size) had an illegal value."; break;
        case -4: msg << "Parameter 4 (A pointer/lda) had an illegal value."; break;
        case -5: msg << "Parameter 5 (W/eigenvalue array) had an illegal value."; break;
        case -6: msg << "Parameter 6 (VL matrix pointer/ldvl) had an illegal value."; break;
        case -7: msg << "Parameter 7 (VR matrix pointer/ldvr) had an illegal value."; break;
        case -8: msg << "Parameter 8 (workspace pointer/size) had an illegal value."; break;
            // You can extend this mapping based on full cusolverDnXgeev docs.
        default:
            if (info_host > 0) {
                msg << "The QR algorithm failed to compute all eigenvalues. "
                    << info_host << " off-diagonal elements of the Hessenberg "
                    << "matrix did not converge to zero.";
            } else {
                msg << "Unknown negative parameter error.";
            }
            break;
    }

    throw std::runtime_error(msg.str());
}


template<typename T>
SquareMat<T> SquareMat<T>::empty() {
    return SquareMat<T>(0, 0, nonOwningGpuPtr<T>(nullptr));
}

template <typename T>
void SquareMat<T>::eigen(
    Vec<T>& eVals,           // Will hold BOTH real (first n) and imaginary (last n) parts
    SquareMat<T>* eVecs,     // Eigenvectors (stored as real/imaginary parts) Set to null if not computed.
    Handle& hand
) const {
    if (this->_rows != this->_cols)
        throw std::invalid_argument("Eigenvalue computation requires a square matrix.");

    auto n = static_cast<int64_t>(this->_rows);

    std::unique_ptr<Vec<T>> temp_reEVal;

    // FIXED: Replaced undeclared '*h' with the passed 'hand' variable
    Vec<T>* eValsPtr = Vec<T>::_get_or_create_target(2*n, &eVals, temp_reEVal, hand);

    size_t workDeviceBytes = 0, workHostBytes = 0;
    cudaDataType_t dataType;

    // Your float/double logic is perfectly suited for the 64-bit generic API
    if constexpr (std::is_same_v<T, float>) dataType = CUDA_R_32F;
    else if constexpr (std::is_same_v<T, double>) dataType = CUDA_R_64F;
    else throw std::invalid_argument("Unsupported type for cusolverDnXgeev.");

    cusolverEigMode_t findVectors = eVecs != nullptr ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

    CHECK_CUSOLVER_ERROR(cusolverDnXgeev_bufferSize(
        hand, nullptr,
        CUSOLVER_EIG_MODE_NOVECTOR, findVectors, n,
        dataType, this->toKernel2d(), this->_ld,
        dataType, eValsPtr->toKernel1d(),
        dataType, nullptr, n,
        dataType, eVecs == nullptr ? nullptr : eVecs->data(), eVecs == nullptr ? n : eVecs->_ld,
        dataType,
        &workDeviceBytes,
        &workHostBytes
    ));

    Vec<uint8_t> workspaceDevice = Vec<uint8_t>::create(workDeviceBytes, hand);
    std::vector<uint8_t> workspaceHost(workHostBytes);
    Singleton<int32_t> info_dev = Singleton<int32_t>::create(hand);

    CHECK_CUSOLVER_ERROR(cusolverDnXgeev(
        hand, nullptr,
        CUSOLVER_EIG_MODE_NOVECTOR, findVectors, n,
        dataType, this->toKernel2d(), this->_ld,
        dataType, eValsPtr->toKernel1d(),
        dataType, nullptr, n,
        dataType, eVecs == nullptr ? nullptr : eVecs->data(), eVecs == nullptr ? n : eVecs->_ld,
        dataType,
        workspaceDevice.toKernel1d(),
        workDeviceBytes,
        workspaceHost.data(),
        workHostBytes,
        info_dev.toKernel1d()
    ));

    // processInfo(info_dev);
}

template<typename T>
SquareMat<T> SquareMat<T>::setToIdentity(cudaStream_t stream) {

    this->fill(0, stream);
    this->diag(0).fill(1, stream);
    return *this;
}


template<typename T>
SquareMat<T> Mat<T>::sqSubMat(size_t startRow, size_t startCol, size_t dim) const {
    return SquareMat<T>(dim, this->_ld, offset(startRow, startCol));
}
template<typename T>
SquareMat<T> Mat<T>::sqSubMatFirstBiggest() const {
    return SquareMat<T>(std::min(this->_rows, this->_cols), this->_ld, offset(0, 0));
}

template<typename T>
void SquareMat<T>::solveLUDecomposed(Mat<T> &b, Vec<int32_t>& rowSwaps, Handle *handle, Singleton<int32_t>* info, bool transpose) {

    std::unique_ptr<Handle> tempHand;
    auto h = Handle::_get_or_create_handle(handle, tempHand);
    std::unique_ptr<Singleton<int32_t>> tempinfo;
    auto inf = Singleton<int32_t>::_get_or_create_target(info, tempinfo, *h);

    const cublasOperation_t transp = transpose ? CUBLAS_OP_T: CUBLAS_OP_N;

    if constexpr(std::is_same_v<T, double>)
        CHECK_CUSOLVER_ERROR(cusolverDnDgetrs(*h, transp, this->_rows, b._cols, this -> data(), this -> _ld, rowSwaps.data(), b.data(), b._ld, inf->toKernel1d()));
    else if constexpr(std::is_same_v<T, float>)
        CHECK_CUSOLVER_ERROR(cusolverDnSgetrs(*h, transp, this->_rows, b._cols, this -> data(), this -> _ld, rowSwaps.data(), b.data(), b._ld, inf->toKernel1d()));
    else throw std::invalid_argument("Unsupported type.");
}

template<typename T>
void SquareMat<T>::solve(Mat<T>& b, Handle *handle, Singleton<int32_t> *info, Vec<T> *workspace, Vec<int32_t> *rowSwaps) {
    std::unique_ptr<Handle> tempHand;
    auto h = Handle::_get_or_create_handle(handle, tempHand);
    std::unique_ptr<Singleton<int32_t>> tempinfo;
    auto inf = Singleton<int32_t>::_get_or_create_target(info, tempinfo, *h);
    std::unique_ptr<Vec<int32_t>> tempRowSwapsPointer;
    auto rs = Vec<int32_t>::_get_or_create_target(this->_rows ,rowSwaps, tempRowSwapsPointer, *h);

    this->factorLU(h, rs, inf, workspace);
    solveLUDecomposed(b, *rs, h, inf, false);
}

template<typename T>
void SquareMat<T>::solve(Vec<T> &b, Handle *handle, Singleton<int32_t> *info, Vec<T> *workspace,
    Vec<int32_t> *rowSwaps) {
    Mat<T> mat = static_cast<Mat<T>>(b);
    solve(mat, handle, info, workspace, rowSwaps);
}

template<typename T>
double SquareMat<T>::determinant(Vec<int32_t>& sizeOfNumRows, Singleton<int32_t>& info, Vec<T>& workSpaceForLUDecomp, Handle& handle) {
    // factorLU(Handle *hand, Vec<int32_t> *rowSwaps, Singleton<int32_t> *info, Vec<T> *workSpace) {
    this->factorLU(&handle, &sizeOfNumRows, &info, &workSpaceForLUDecomp);
    int32_t infoHost = info.get(handle);

    if (infoHost != 0) return 0;

    double det = static_cast<T>(1);

    Vec<T> diagonal = this->diag(0);

    det *= diagonal.productAllElements(
            workSpaceForLUDecomp.subVec(0, static_cast<int32_t>(diagonal.kernelPrep().numBlocks.x)),
            handle
        );

    std::vector<int32_t> pivots(this->_rows, 0);

    sizeOfNumRows.get(pivots.data(), handle);

    int32_t numSwaps = 0;
    for (size_t i = 0; i < this->_rows; ++i) if (pivots[i] != i + 1) ++numSwaps;

    if (numSwaps % 2) det *= -1;

    return det;
}


template<typename T>
double SquareMat<T>::determinant(Handle& hand) { //TODO: This method should call my LU decomp method.
    int lwork = 0;

    if constexpr (std::is_same_v<T, double>)
        CHECK_CUSOLVER_ERROR(
            cusolverDnDgetrf_bufferSize(
                hand,
                this->_rows,
                this->_cols,
                this->toKernel2d(),
                this->_ld,
                &lwork));

    else if constexpr (std::is_same_v<T, float>)
        CHECK_CUSOLVER_ERROR(
            cusolverDnSgetrf_bufferSize(
                hand,
                this->_rows,
                this->_cols,
                this->toKernel2d(),
                this->_ld,
                &lwork));

    else throw std::invalid_argument("Unsupported type.");

    auto rowSwaps = Vec<int32_t>::create(this->_rows, hand);

    auto info = Singleton<int32_t>::create(hand);

    auto workSpace = Vec<T>::create(
        std::max(
            lwork,
            static_cast<int>(KernelPrep(this->_rows).numBlocks.x)
        ),
        hand
    );

    return determinant(rowSwaps, info, workSpace, hand);
}


template<typename T>
bool SquareMat<T>::isSingular(double tolerance, Vec<int32_t>& rowSwaps, Singleton<int32_t>& info, Vec<T>& workSpace, Handle& hand) {
    if (this->_rows == 0) return true;

    // Perform in-place LU decomposition
    this->factorLU(&hand, &rowSwaps, &info, &workSpace);

    int32_t infoHost = info.get(hand);

    // cuSOLVER getrf sets info > 0 if an exact zero pivot is found (U(i,i) == 0).
    if (infoHost > 0) return true;
    if (infoHost < 0) throw std::runtime_error("Illegal parameter passed to LU decomposition.");

    // Extract the main diagonal of the resulting LU packed matrix
    Vec<T> diagonal = this->diag(0);
    std::vector<T> diagHost(this->_rows);
    diagonal.get(diagHost.data(), hand);
    hand.synch();

    double max_diag = 0.0;
    double min_diag = std::numeric_limits<double>::max();

    for (size_t i = 0; i < this->_rows; ++i) {
        double abs_val = std::abs(static_cast<double>(diagHost[i]));
        if (abs_val > max_diag) max_diag = abs_val;
        if (abs_val < min_diag) min_diag = abs_val;
    }

    if (max_diag == 0.0) return true;

    if (tolerance < 0.0) {
        if constexpr (std::is_same_v<T, float>) tolerance = this->_rows * 1.1920929e-07; // n * flt_epsilon
        else tolerance = this->_rows * 2.220446049250313e-16; // n * dbl_epsilon
    }

    // Singularity check: min diagonal is excessively small compared to max diagonal
    return (min_diag <= tolerance * max_diag);
}


template<typename T>
bool SquareMat<T>::isSingular(double tolerance, Handle& hand) const {
    if (this->_rows == 0) return true;

    int lwork = 0;

    if constexpr (std::is_same_v<T, double>) {
        CHECK_CUSOLVER_ERROR(cusolverDnDgetrf_bufferSize(hand, this->_rows, this->_cols, this->toKernel2d(), this->_ld, &lwork));
    } else if constexpr (std::is_same_v<T, float>) {
        CHECK_CUSOLVER_ERROR(cusolverDnSgetrf_bufferSize(hand, this->_rows, this->_cols, this->toKernel2d(), this->_ld, &lwork));
    } else throw std::invalid_argument("Unsupported type.");

    auto rowSwaps = Vec<int32_t>::create(this->_rows, hand);
    auto info = Singleton<int32_t>::create(hand);
    auto workSpace = Vec<T>::create(std::max(lwork, static_cast<int>(KernelPrep(this->_rows).numBlocks.x)), hand);

    // DEEP COPY: factorLU modifies the data in-place, so we must operate on a copy
    auto copyMat = SquareMat<T>::create(this->_rows);
    this->get(copyMat, hand);

    return copyMat.isSingular(tolerance, rowSwaps, info, workSpace, hand);
}



// 1. Define the expansion macro
#define INSTANTIATE_SQUARE_MAT(T) \
template class SquareMat<T>; \
template SquareMat<T> Mat<T>::sqSubMat(size_t, size_t, size_t) const; \
template SquareMat<T> Mat<T>::sqSubMatFirstBiggest() const;

// 2. Use the macro for all your types
INSTANTIATE_SQUARE_MAT(float)
INSTANTIATE_SQUARE_MAT(double)
INSTANTIATE_SQUARE_MAT(int32_t)
INSTANTIATE_SQUARE_MAT(size_t)
INSTANTIATE_SQUARE_MAT(unsigned char)

// 3. Clean up (optional)
#undef INSTANTIATE_SQUARE_MAT