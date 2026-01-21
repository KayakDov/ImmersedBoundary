#ifndef BICGSTAB_STREAMABLE_H
#define BICGSTAB_STREAMABLE_H

// Assuming this header (or headers included by it) provides the definitions for
// GpuArray<T>, Vec<T>, Tensor<T>, and cudaStream_t
#include "deviceArrays/headers/SimpleArray.h"
#include <iostream>
#include <type_traits> // For std::is_const_v

/**
 * @brief Base class for GpuArray streaming, holding common context parameters.
 */
template <typename T>
class StreamContext {
public:
    const bool colMajor;
    const bool isText;
    const cudaStream_t& stream;

    /**
     * @brief Constructs the base streaming context parameters.
     * @param s The CUDA stream to use for host-device transfers.
     * @param text True if streaming should be human-readable text, false for binary.
     * @param colMjr True if streaming should be column-major order, false for row-major.
     */
    StreamContext(const cudaStream_t& s, bool text, bool colMjr);
};

/**
 * @brief Wrapper class for reading data FROM an input stream INTO a non-const GpuArray<T> or Tensor<T>.
 *
 * This class supports the global stream extraction operator (operator>>).
 */
template<typename T>
class GpuIn : public StreamContext<T> {

public:
    GpuArray<T>& src;

    GpuIn(GpuArray<T>& dst, const cudaStream_t &stream, bool isText = true, bool columnMjr = false);


    GpuIn(Tensor<T>& dst, const cudaStream_t &stream, bool isText = true, bool columnMjr = false);

    /**
     * @brief The core reading logic implemented in the .cu file.
     * @param is The input stream to read data from.
     * @return A reference to the input stream.
     */
    std::istream& read(std::istream& is);
};

/**
 * @brief Wrapper class for writing data FROM a const GpuArray<T> or Tensor<T> TO an output stream.
 *
 * This class supports the global stream insertion operator (operator<<).
 */
template<typename T>
class GpuOut : public StreamContext<T> {

public:
    const GpuArray<T>& src;
    GpuOut(const GpuArray<T>& src, const cudaStream_t &stream, bool isText = true, bool columnMjr = false);

    GpuOut(const Tensor<T>& src, const cudaStream_t &stream, bool isText = true, bool columnMjr = false);

    /**
     * @brief The core writing logic implemented in the .cu file.
     * @param os The output stream to write data to.
     * @return A reference to the output stream.
     */
    std::ostream& write(std::ostream& os) const;
};


/**
 * @brief Global stream insertion operator to wrap a GpuArray for printing.
 *
 * It delegates to the GpuArrayWriter::write() member function.
 *
 * @tparam T The data type of the GpuArray.
 * @param output_stream The target output stream.
 * @param wrapper The GpuArrayWriter containing the array and formatting options.
 * @return A reference to the output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream& output_stream, const GpuOut<T>& wrapper) {
    return wrapper.write(output_stream);
}

/**
 * @brief Global stream extraction operator to wrap a GpuArray for reading.
 *
 * It delegates to the GpuArrayReader::read() member function.
 *
 * @tparam T The data type of the GpuArray.
 * @param input_stream The source input stream.
 * @param wrapper The GpuArrayReader containing the array and formatting options.
 * @return A reference to the input stream.
 */
template <typename T>
std::istream& operator>>(std::istream& input_stream, GpuIn<T>&& wrapper) {
    return wrapper.read(input_stream);
}

#endif //BICGSTAB_STREAMABLE_H