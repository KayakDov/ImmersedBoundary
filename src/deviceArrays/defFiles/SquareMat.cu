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
SquareMat<T> SquareMat<T>::create(size_t rowsCols, Handle& hand) {
    Mat<T> mat = Mat<T>::create(rowsCols, rowsCols, hand);
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
inline void processInfo(const Singleton<int32_t>& info_dev, Handle& stream, const std::string& context = "cusolverDnXgeev")
{
    const int info_host = info_dev.get(stream);

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
void SquareMat<T>::eigen(//TODO: This method may not be correctly handaling real and imaginary componenets.
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
SquareMat<T> SquareMat<T>::setToIdentity(Handle& stream) {

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
template<typename Int>
void SquareMat<T>::solveLUDecomposed(Mat<T> &b, Vec<Int>& rowSwaps, Handle& handle, Singleton<int32_t>& info, bool transpose) {

    const cublasOperation_t transp = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

    // 32-bit Integer Path (Standard typed API)
    if constexpr (std::is_same_v<Int, int32_t>) {
        if constexpr (std::is_same_v<T, double>) {
            CHECK_CUSOLVER_ERROR(cusolverDnDgetrs(
                handle, transp, this->_rows, b._cols,
                this->data(), this->_ld,
                rowSwaps.data(),
                b.data(), b._ld,
                info.toKernel1d()
            ));
        } else if constexpr (std::is_same_v<T, float>) {
            CHECK_CUSOLVER_ERROR(cusolverDnSgetrs(
                handle, transp, this->_rows, b._cols,
                this->data(), this->_ld,
                rowSwaps.data(),
                b.data(), b._ld,
                info.toKernel1d()
            ));
        } else {
            throw std::invalid_argument("Unsupported floating-point type for 32-bit cuSOLVER getrs.");
        }
    }
    else if constexpr (std::is_same_v<Int, int64_t>) {
        cudaDataType_t dataType;
        if constexpr (std::is_same_v<T, float>) dataType = CUDA_R_32F;
        else if constexpr (std::is_same_v<T, double>) dataType = CUDA_R_64F;
        else throw std::invalid_argument("Unsupported floating-point type for 64-bit cuSOLVER getrs.");

        CHECK_CUSOLVER_ERROR(cusolverDnXgetrs(
            handle, nullptr, transp,
            static_cast<int64_t>(this->_rows),
            static_cast<int64_t>(b._cols),
            dataType, this->data(), static_cast<int64_t>(this->_ld),
            rowSwaps.data(),
            dataType, b.data(), static_cast<int64_t>(b._ld),
            info.toKernel1d()
        ));
    }
    // Failsafe for invalid types
    else {
        static_assert(sizeof(Int) == 0, "Int template parameter must be int32_t or int64_t.");
    }
}

template<typename T>
template<typename Int>
void SquareMat<T>::inverse(SquareMat<T> &result, SimpleArray<Int>& rowSwaps, Handle& handle, Singleton<int32_t> info, SimpleArray<T>& buffer, bool transpose) {
    result.setToIdentity(handle);
    solve(result, handle, info, buffer, rowSwaps, transpose);
}

template<typename T>
template<typename Int>
void SquareMat<T>::solve(Mat<T>& b, Handle& handle, Singleton<int32_t>& info, SimpleArray<T>& buffer, SimpleArray<Int>& rowSwaps, bool transpose) {

    this->factorLU(handle, rowSwaps, info, buffer);
    solveLUDecomposed(b, rowSwaps, handle, info, transpose);
}


template<typename T>
double SquareMat<T>::determinant(SimpleArray<int32_t>& sizeOfNumRows, Singleton<int32_t>& info, SimpleArray<T>& workSpaceForLUDecomp, Handle& handle) {
    if constexpr (!std::is_floating_point_v<T>) {
        throw std::runtime_error("Determinant is not defined for non-floating point types.");
        return 0.0; // Unreachable
    }

    this->factorLU(handle, sizeOfNumRows, info, workSpaceForLUDecomp);
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
double SquareMat<T>::determinant(Handle& hand) {
    //TODO: This method should call my LU decomp method.
    if constexpr (!std::is_floating_point_v<T>) {
        throw std::runtime_error("determinent requires a floating-point matrix (float or double).");
    } else {


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

        auto rowSwaps = SimpleArray<int32_t>::create(this->_rows, hand);

        auto info = Singleton<int32_t>::create(hand);

        auto workSpace = SimpleArray<T>::create(
            std::max(
                lwork,
                static_cast<int>(KernelPrep(this->_rows).numBlocks.x)
            ),
            hand
        );

        return determinant(rowSwaps, info, workSpace, hand);
    }
}


template<typename T>
bool SquareMat<T>::isSingular(double tolerance, SimpleArray<int32_t>& rowSwaps, Singleton<int32_t>& info, SimpleArray<T>& workSpace, Handle& hand) {
    if constexpr (!std::is_floating_point_v<T>) {
        throw std::runtime_error("isSingular is not defined for non-floating point types.");
        return true; // Unreachable
    }
    if (this->_rows == 0) return true;

    // Perform in-place LU decomposition
    this->factorLU(hand, rowSwaps, info, workSpace);

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

    auto rowSwaps = SimpleArray<int32_t>::create(this->_rows, hand);
    auto info = Singleton<int32_t>::create(hand);
    auto workSpace = SimpleArray<T>::create(std::max(lwork, static_cast<int>(KernelPrep(this->_rows).numBlocks.x)), hand);

    // DEEP COPY: factorLU modifies the data in-place, so we must operate on a copy
    auto copyMat = SquareMat<T>::create(this->_rows, hand);
    this->get(copyMat, hand);

    return copyMat.isSingular(tolerance, rowSwaps, info, workSpace, hand);
}


template<typename T>
std::pair<size_t, size_t> SquareMat<T>::eigenSPDBufferSize(Handle& hand) const
{
    // Use if constexpr instead of static_assert to avoid breaking integer class instantiations
    if constexpr (!std::is_floating_point_v<T>) {
        throw std::runtime_error("eigenSPDBufferSize requires a floating-point matrix (float or double).");
    } else {
        const cudaDataType_t dataType =
            std::is_same_v<T, double> ? CUDA_R_64F : CUDA_R_32F;

        cusolverDnParams_t params;
        CHECK_CUSOLVER_ERROR(cusolverDnCreateParams(&params));

        size_t deviceBytesNeeded = 0;
        size_t hostBytesNeeded   = 0;

        CHECK_CUSOLVER_ERROR(cusolverDnXsyevd_bufferSize(
            hand,
            params,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            this->_rows,
            dataType, this->data(), static_cast<int64_t>(this->_ld),
            dataType, nullptr,
            dataType,
            &deviceBytesNeeded,
            &hostBytesNeeded
        ));

        CHECK_CUSOLVER_ERROR(cusolverDnDestroyParams(params));

        const size_t deviceElems = (deviceBytesNeeded + sizeof(T) - 1) / sizeof(T);
        const size_t hostElems = (hostBytesNeeded   + sizeof(T) - 1) / sizeof(T);

        return {deviceElems, hostElems};
    }
}

/**
 * @brief Checks the info result from cusolverDnXsyevd and throws if an error occurred.
 *
 * cusolverDnXsyevd sets info as follows:
 *   info = 0  : success.
 *   info < 0  : the (-info)-th parameter had an illegal value.
 *   info > 0  : the algorithm failed to converge; info off-diagonal elements
 *               of an intermediate tridiagonal form did not converge to zero.
 *
 * @param info_dev  Device-side Singleton<int32_t> written by cusolverDnXsyevd.
 * @param context   String appended to the error message for caller identification.
 *
 * @throws std::runtime_error if info != 0.
 */
inline void processInfoSyevd(const Singleton<int32_t>& info_dev, Handle& stream, const std::string& context = "cusolverDnXsyevd")
{
    const int info_host = info_dev.get(stream);

    if (info_host == 0) return;

    std::ostringstream msg;
    msg << "cuSOLVER error in " << context << ": info = " << info_host << ". ";

    if (info_host > 0) {
        msg << "The symmetric eigensolver (Divide & Conquer / QR iteration) failed "
            << "to converge. " << info_host << " off-diagonal elements of the "
            << "intermediate tridiagonal matrix did not converge to zero. "
            << "The matrix may be numerically indefinite or very ill-conditioned.";
    } else {
        const int param = -info_host;
        msg << "Parameter " << param << " had an illegal value. ";
        switch (param) {
            case 1:  msg << "(handle)"; break;
            case 2:  msg << "(params descriptor)"; break;
            case 3:  msg << "(jobz: eigenvector mode)"; break;
            case 4:  msg << "(uplo: fill mode)"; break;
            case 5:  msg << "(n: matrix dimension)"; break;
            case 6:  msg << "(dataTypeA)"; break;
            case 7:  msg << "(A: device matrix pointer)"; break;
            case 8:  msg << "(lda: leading dimension of A)"; break;
            case 9:  msg << "(dataTypeW)"; break;
            case 10: msg << "(W: device eigenvalue array pointer)"; break;
            case 11: msg << "(computeType)"; break;
            case 12: msg << "(bufferOnDevice: device workspace pointer)"; break;
            case 13: msg << "(workspaceInBytesOnDevice)"; break;
            case 14: msg << "(bufferOnHost: host workspace pointer)"; break;
            case 15: msg << "(workspaceInBytesOnHost)"; break;
            case 16: msg << "(info: device result pointer)"; break;
            default: msg << "(unknown parameter)"; break;
        }
    }

    std::cout << "SquareMat.cu::processInforSyved info ran" << std::endl;

    throw std::runtime_error(msg.str());
}

template<typename T>
void SquareMat<T>::eigenSPD(
    Vec<T>&              eVals,
    Handle&              hand,
    SimpleArray<T>&      deviceBuffer,
    T* hostBuffer,
    size_t               hostBufferSize,
    Singleton<int32_t>&  info
)
{
    if constexpr (!std::is_floating_point_v<T>) {
        throw std::runtime_error("eigenSPD requires a floating-point matrix (float or double).");
    } else {
        if (eVals.size() < this->_rows)
            throw std::invalid_argument("eigenSPD: eVals is too small for the matrix.");

        const cudaDataType_t dataType =
            std::is_same_v<T, double> ? CUDA_R_64F : CUDA_R_32F;

        cusolverDnParams_t params;
        CHECK_CUSOLVER_ERROR(cusolverDnCreateParams(&params));

        CHECK_CUSOLVER_ERROR(cusolverDnXsyevd(
            hand,
            params,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            this->_rows,
            dataType, this->data(), this->_ld,
            dataType, eVals.data(),
            dataType,
            deviceBuffer.data(), deviceBuffer.size() * sizeof(T),
            hostBuffer, hostBufferSize * sizeof(T),
            info.data()
        ));

        CHECK_CUSOLVER_ERROR(cusolverDnDestroyParams(params));
        hand.synch(); //It looks like if hostBuffer gets cleaned up to soon after this is done running, then crash.
        // processInfoSyevd(info, hand);
    }
}



template<typename T>
void SquareMat<T>::eigenSPD(
    Vec<T>&       eVals,
    Handle&       hand
){
    auto [deviceElems, hostElems] = eigenSPDBufferSize(hand);

    auto preDeviceBuffer = SimpleArray<T>::create(std::max(deviceElems, size_t{1}) + 1, hand);
    std::vector<T> hostBuffer(std::max(hostElems, size_t{1}));

    auto deviceBuffer = preDeviceBuffer.subArray(0, deviceElems);
    auto info = Singleton<int32_t>::create(hand);

    eigenSPD(eVals, hand, deviceBuffer, hostBuffer.data(), hostElems, info);
}

// ============================================================================
// --- Member function templates: integer pivot variants ---
// ============================================================================

// 32-bit pivots
template void SquareMat<float>::solveLUDecomposed<int32_t>(Mat<float>&, Vec<int32_t>&, Handle&, Singleton<int32_t>&, bool);
template void SquareMat<double>::solveLUDecomposed<int32_t>(Mat<double>&, Vec<int32_t>&, Handle&, Singleton<int32_t>&, bool);

template void SquareMat<float>::solve<int32_t>(Mat<float>&, Handle&, Singleton<int32_t>&, SimpleArray<float>&, SimpleArray<int32_t>&, bool);
template void SquareMat<double>::solve<int32_t>(Mat<double>&, Handle&, Singleton<int32_t>&, SimpleArray<double>&, SimpleArray<int32_t>&, bool);

template void SquareMat<float>::inverse<int32_t>(SquareMat<float>&, SimpleArray<int32_t>&, Handle&, Singleton<int32_t>, SimpleArray<float>&, bool);
template void SquareMat<double>::inverse<int32_t>(SquareMat<double>&, SimpleArray<int32_t>&, Handle&, Singleton<int32_t>, SimpleArray<double>&, bool);

// 64-bit pivots
template void SquareMat<float>::solveLUDecomposed<int64_t>(Mat<float>&, Vec<int64_t>&, Handle&, Singleton<int32_t>&, bool);
template void SquareMat<double>::solveLUDecomposed<int64_t>(Mat<double>&, Vec<int64_t>&, Handle&, Singleton<int32_t>&, bool);

template void SquareMat<float>::solve<int64_t>(Mat<float>&, Handle&, Singleton<int32_t>&, SimpleArray<float>&, SimpleArray<int64_t>&, bool);
template void SquareMat<double>::solve<int64_t>(Mat<double>&, Handle&, Singleton<int32_t>&, SimpleArray<double>&, SimpleArray<int64_t>&, bool);

template void SquareMat<float>::inverse<int64_t>(SquareMat<float>&, SimpleArray<int64_t>&, Handle&, Singleton<int32_t>, SimpleArray<float>&, bool);
template void SquareMat<double>::inverse<int64_t>(SquareMat<double>&, SimpleArray<int64_t>&, Handle&, Singleton<int32_t>, SimpleArray<double>&, bool);

// ============================================================================
// --- Full class + Mat helper instantiations ---
// Covers: SquareMat<T> class body (including eigenSPD/eigenSPDBufferSize via
// if constexpr), Mat<T>::sqSubMat, and Mat<T>::sqSubMatFirstBiggest.
//
// eigenSPD and eigenSPDBufferSize do NOT get separate explicit instantiations
// here because `template class SquareMat<T>` already covers all non-template
// member functions for each T. Listing them separately would produce
// "explicitly instantiated more than once" errors.
//
// Type notes (Linux x86-64):
//   int32_t == int      (covers BandedMat<int>, ImmersedEq<T,int>)
//   int64_t == long     (covers ImmersedEq<T,long>)
//   size_t  == unsigned long  (covers BandedMat<size_t/unsigned long>)
// ============================================================================
#define INSTANTIATE_SQUARE_MAT(T)                                      \
template class SquareMat<T>;                                           \
template SquareMat<T> Mat<T>::sqSubMat(size_t, size_t, size_t) const; \
template SquareMat<T> Mat<T>::sqSubMatFirstBiggest() const;

INSTANTIATE_SQUARE_MAT(float)
INSTANTIATE_SQUARE_MAT(double)
INSTANTIATE_SQUARE_MAT(int32_t)     // == int
INSTANTIATE_SQUARE_MAT(int64_t)     // == long  (needed by ImmersedEq<T,long>)
INSTANTIATE_SQUARE_MAT(size_t)      // == unsigned long
INSTANTIATE_SQUARE_MAT(unsigned char)

#undef INSTANTIATE_SQUARE_MAT