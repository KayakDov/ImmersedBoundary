#include "../headers/Support/Streamable.h"

#include "../headers/Tensor.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>

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
GpuOut<T>::GpuOut(const GpuArray<T> &src, const cudaStream_t &stream, bool isText, bool columnMjr): StreamContext<T>(stream, isText, columnMjr), src(src) {
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
        cudaStreamSynchronize(current_stream);

        if (this->isText) {
            for (size_t j = 0; j < inner_dim; ++j) {
                os << host_buffer[j];
                if (j < inner_dim - 1) {
                    os << ", ";
                }
            }
            os << "\n";
        } else {
            os.write(reinterpret_cast<const char*>(host_buffer.data()), inner_dim * sizeof(T));
        }
    }
    return os;
}



// float
template class GpuIn<float>;
template class GpuOut<float>;

// double
template class GpuIn<double>;
template class GpuOut<double>;

// size_t
template class GpuIn<size_t>;
template class GpuOut<size_t>;

// int32_t
template class GpuIn<int32_t>;
template class GpuOut<int32_t>;

template class GpuIn<int64_t>;
template class GpuOut<int64_t>;

// unsigned char
template class GpuIn<unsigned char>;
template class GpuOut<unsigned char>;

template class GpuIn<uint32_t>;
template class GpuOut<uint32_t>;