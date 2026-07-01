#include "../headers/Support/Streamable.h"

#include "../headers/Tensor.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/GpuArray.h"
#include "deviceArrays/headers/SimpleArray.h"
#include "deviceArrays/headers/Vec.h"


// ----------------------------------------------------------------------
// GpuArrayReader Constructors
// ----------------------------------------------------------------------

template<typename T>
StreamContext<T>::StreamContext(const cudaStream_t &s, bool text, bool colMjr): stream(s), isText(text), colMajor(colMjr) {
}

template<typename T>
GpuIn<T>::GpuIn(GpuArray<T> &dst, const cudaStream_t &stream, bool isText, bool columnMjr): StreamContext<T>(stream, isText, columnMjr), src(dst){
}

template<typename T>
GpuIn<T>::GpuIn(Tensor<T>& dst, const cudaStream_t &stream, bool isText, bool columnMjr)
    : GpuIn<T>(dst.utilityMatrix, stream, isText, columnMjr) {}


template <typename T>
std::istream& GpuIn<T>::read(std::istream& is) {
    size_t outer_dim = this->colMajor ? this->src._cols : this->src._rows;
    size_t inner_dim = this->colMajor ? this->src._rows : this->src._cols;

    const cudaStream_t current_stream = this->stream;

    for (size_t i = 0; i < outer_dim; ++i) {
        Vec<T> view = this->colMajor ? this->src.col(i) : this->src.row(i);
        std::vector<T> host_buffer(view.size());

        if (this->isText) {
            for (size_t j = 0; j < inner_dim; ++j) {
                if (!(is >> host_buffer[j])) {
                    is.setstate(std::ios::failbit);
                    return is;
                }
            }
        } else {
            is.read(reinterpret_cast<char*>(host_buffer.data()), inner_dim * sizeof(T));
            if (!is) return is;
        }

        try {
            view.set(host_buffer.data(), current_stream);
            cudaStreamSynchronize(current_stream);
        } catch (const std::exception& e) {
            std::cerr << "Error during Host to GPU transfer for streaming: " << e.what() << std::endl;
            throw;
        }
    }
    return is;
}

// ----------------------------------------------------------------------
// GpuArrayWriter Constructors
// ----------------------------------------------------------------------

template<typename T>
GpuOut<T>::GpuOut(const GpuArray<T>& src, const cudaStream_t &stream, bool isText, bool columnMjr): StreamContext<T>(stream, isText, columnMjr), src(src) {
}

template<typename T>
GpuOut<T>::GpuOut(const Tensor<T>& src, const cudaStream_t &stream, bool isText, bool columnMjr)
    : GpuOut<T>(src.utilityMatrix, stream, isText, columnMjr) {}


template <typename T>
std::ostream& GpuOut<T>::write(std::ostream& os) const {
    size_t outer_dim = this->colMajor ? this->src._cols : this->src._rows;
    size_t inner_dim = this->colMajor ? this->src._rows : this->src._cols;

    const cudaStream_t current_stream = this->stream;

    for (size_t i = 0; i < outer_dim; ++i) {
        Vec<T> view = this->colMajor ? this->src.col(i) : this->src.row(i);

        std::vector<T> host_buffer(view.size());

        view.get(host_buffer.data(), current_stream);
        CHECK_CUDA_ERROR(cudaGetLastError());
        cudaStreamSynchronize(current_stream);

        if (this->isText) {
            os << "[";
            for (size_t j = 0; j < inner_dim; ++j) {
                os << host_buffer[j];
                if (j < inner_dim - 1) {
                    os << ", ";
                }
            }
            os << "]\n";
        } else {
            os.write(reinterpret_cast<const char*>(host_buffer.data()), inner_dim * sizeof(T));
        }
    }
    return os;
}

template<typename  dataStruct, typename T>
GpuX3Out<dataStruct, T>::GpuX3Out(const XYZ<dataStruct> &gpu3d, Handle &hand): gpu3d(gpu3d), hand(hand) {
}


// Helper macro for scalar GPU wrappers
#define INSTANTIATE_GPU_IO(T)   \
template class GpuIn<T>;    \
template class GpuOut<T>;

INSTANTIATE_GPU_IO(float)
INSTANTIATE_GPU_IO(double)
INSTANTIATE_GPU_IO(size_t)
INSTANTIATE_GPU_IO(int32_t)
INSTANTIATE_GPU_IO(int64_t)
INSTANTIATE_GPU_IO(unsigned char)
INSTANTIATE_GPU_IO(uint32_t)

#undef INSTANTIATE_GPU_IO
#undef INSTANTIATE_GPU_X3_OUT

template class GpuX3Out<Vec<double>, double>;
template class GpuX3Out<Vec<float>, float>;

template class GpuX3Out<Mat<float>, float>;
template class GpuX3Out<Mat<double>, double>;

template class GpuX3Out<SquareMat<double>, double>;
template class GpuX3Out<SquareMat<float>, float>;

template class GpuX3Out<SimpleArray<double>, double>;
template class GpuX3Out<SimpleArray<float>, float>;