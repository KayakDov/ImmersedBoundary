#include <cusolver_common.h>
#include <iostream>
#include <sstream>

#include "../headers/SquareMat.h"
#include "../headers/KernelSupport.cuh"
#include "../headers/deviceArraySupport.h"
#include "../headers/Singleton.h"
#include "../headers/Vec.h"
#include "../headers/BandedMat.h"
#include "../headers/GridDim.cuh"


template<typename T>
SquareMat<T>::SquareMat(const size_t rowsCols, const size_t ld, std::shared_ptr<T> _ptr) :
    Mat<T>(rowsCols, rowsCols, ld, _ptr) {
}

template<typename T>
SquareMat<T> SquareMat<T>::create(size_t rowsCols) {
    Mat<T> mat = Mat<T>::create(rowsCols, rowsCols);
    return SquareMat<T>(rowsCols, mat._ld, mat.ptr());
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


// Wrapper for cusolverDnXgeev
template <typename T>
void SquareMat<T>::eigen(
    Vec<T>& eVals,     // Real part of eigenvalues
    SquareMat<T>* eVecs,      // Eigenvectors (stored as real/imaginary parts)  Set to null if vectors should not be computed.
    Mat<T>* temp,        // Optional pre-allocated temporary matrix to use for the eigenvalue computation.  It should be the same size as this matrix.
    Handle* handle
) const {
    if (this->_rows != this->_cols)
        throw std::invalid_argument("Eigenvalue computation requires a square matrix.");

    std::unique_ptr<Handle> temp_hand_ptr;
    Handle* h = Handle::_get_or_create_handle(handle, temp_hand_ptr);

    auto n = static_cast<int64_t>(this->_rows);

    std::unique_ptr<Mat<T>> temp_mat_ptr;
    Mat<T>* copy = Mat<T>::_get_or_create_target(n, n, temp, temp_mat_ptr);
    this->get(*copy, h->stream);

    std::unique_ptr<Vec<T>> temp_reEVal;
    Vec<T>* eValsPtr = Vec<T>::_get_or_create_target(2*n, &eVals, temp_reEVal, h->stream);

    size_t workDeviceBytes, workHostBytes;
    cudaDataType_t dataType;
    if constexpr (std::is_same_v<T, float>) dataType = CUDA_R_32F;
    else if constexpr (std::is_same_v<T, double>) dataType = CUDA_R_64F;
    else throw std::invalid_argument("Unsupported type for cusolverDnXgeev.");

    cusolverEigMode_t findVectors = eVecs != nullptr ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

    CHECK_CUSOLVER_ERROR(cusolverDnXgeev_bufferSize(
        h->cusolverHandle, nullptr,
        CUSOLVER_EIG_MODE_NOVECTOR, findVectors, n,
        dataType, copy->toKernel2d(), copy->_ld,
        dataType, eValsPtr->toKernel1d(),
        dataType, nullptr, n,
        dataType, eVecs == nullptr ? nullptr : eVecs->data(), eVecs == nullptr? n : eVecs->_ld,
        dataType,
        &workDeviceBytes,
        &workHostBytes
        ));

    Vec<uint8_t> workspaceDevice = Vec<uint8_t>::create(workDeviceBytes, h->stream);
    std::vector<uint8_t> workspaceHost(workHostBytes);
    Singleton<int32_t> info_dev = Singleton<int32_t>::create(h->stream);

    CHECK_CUSOLVER_ERROR(cusolverDnXgeev(
        h->cusolverHandle, nullptr,
        CUSOLVER_EIG_MODE_NOVECTOR, findVectors, n,
        dataType, copy->toKernel2d(), copy->_ld,
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

template <typename T>
__global__ void setToIdentityKernel(DeviceData2d<T> mat) {

    if (const GridInd2d ind; ind < mat)
        mat[ind] = ind.row == ind.col ? 1 : 0;

}

template<typename T>
SquareMat<T> SquareMat<T>::setToIdentity(cudaStream_t stream) {

    constexpr dim3 blockDim(16, 16);

    const dim3 gridDim(
        (this->_cols + blockDim.x - 1) / blockDim.x,
        (this->_rows + blockDim.y - 1) / blockDim.y
    );

    setToIdentityKernel<T><<<gridDim, blockDim, 0, stream>>>(
        this->toKernel2d()
    );
    return *this;
}

template class SquareMat<float>;
template class SquareMat<double>;
template class SquareMat<size_t>;
template class SquareMat<int32_t>;
template class SquareMat<unsigned char>;