#include "deviceArrays/headers/Streamable.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>

template<typename T>
Streamable<T>::Streamable(const cudaStream_t& stream, GpuArray<T>& src, const bool isText, const bool columnMjr)
    : isText(isText), colMajor(columnMjr), stream(stream), src(src) {}

template<typename T>
Streamable<T>::Streamable(const cudaStream_t &hand, Tensor<T> &src, bool isText, bool isColMjr): Streamable(hand, src.utilityMatrix, isText, isColMjr) {
}

/**
 * @brief Writes the contents of the wrapped GpuArray to an output stream.
 *
 * It pulls data row-by-row or column-by-column to the host and streams it in
 * either text or binary format.
 *
 * @tparam T The data type.
 * @param os The output stream.
 * @return A reference to the output stream.
 */
template <typename T>
std::ostream& Streamable<T>::write(std::ostream& os) const {
    size_t outer_dim = colMajor ? src._cols : src._rows;
    size_t inner_dim = colMajor ? src._rows : src._cols;

    for (size_t i = 0; i < outer_dim; ++i) {

        Vec<T> view = colMajor ? src.col(i) : src.row(i);

        std::vector<T> host_buffer(view.size());

        view.get(host_buffer.data(), stream);
        cudaStreamSynchronize(stream);

        if (isText) {
            for (size_t j = 0; j < inner_dim; ++j) os << host_buffer[j] << ", ";
            os << "\n";
        } else os.write(reinterpret_cast<const char*>(host_buffer.data()), inner_dim * sizeof(T));
    }
    return os;
}

/**
 * @brief Reads data from an input stream into the wrapped GpuArray.
 *
 * It reads data into a host array row-by-row or column-by-column and pushes
 * it to the device memory.
 *
 * @tparam T The data type.
 * @param is The input stream.
 * @return A reference to the input stream.
 */
template <typename T>
std::istream& Streamable<T>::read(std::istream& is) {
    size_t outer_dim = colMajor ? src._cols : src._rows;
    size_t inner_dim = colMajor ? src._rows : src._cols;

    for (size_t i = 0; i < outer_dim; ++i) {
        Vec<T> view = colMajor ? src.col(i) : src.row(i);
        std::vector<T> host_buffer(view.size());

        if (isText) {
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
            view.set(host_buffer.data(), stream);
            cudaStreamSynchronize(stream);
        } catch (const std::exception& e) {
            std::cerr << "Error during Host to GPU transfer for streaming: " << e.what() << std::endl;
            throw;
        }
    }
    return is;
}


template class Streamable<float>;
template class Streamable<double>;
template class Streamable<size_t>;
template class Streamable<int32_t>;
template class Streamable<unsigned char>;