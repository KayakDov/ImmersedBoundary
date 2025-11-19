#ifndef BICGSTAB_STREAMABLE_H
#define BICGSTAB_STREAMABLE_H
#include "deviceArrays/headers/Vec.h"

template<typename T>
class Streamable {

public:
    const bool colMajor;
    const bool isText;
    const cudaStream_t& stream;
    GpuArray<T>& src;

    std::istream& read(std::istream& is);

    Streamable(const cudaStream_t &hand, GpuArray<T> &src, bool isText = true, bool columnMjr = false);



    Streamable(const cudaStream_t &hand, Tensor<T> &src, bool isText = true, bool columnMjr = false);


    std::ostream& write(std::ostream& os) const;

};


/**
 * @brief Global stream insertion operator to wrap a GpuArray for printing.
 *
 * This operator is the entry point for formatting the output of a GpuArray.
 * Example usage: `std::cout << Streamable<float>(true, false, my_handle, my_array);`
 *
 * @tparam T The data type of the GpuArray.
 * @param output_stream The target output stream.
 * @param wrapper The Streamable wrapper containing the array and formatting options.
 * @return A reference to the output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream& output_stream, const Streamable<T>& wrapper) {
    return wrapper.write(output_stream);
}

/**
 * @brief Global stream extraction operator to wrap a GpuArray for reading.
 *
 * This operator is the entry point for reading data from an input stream into a GpuArray.
 * Example usage: `std::cin >> Streamable<float>(true, false, my_handle, my_array);`
 *
 * @tparam T The data type of the GpuArray.
 * @param input_stream The source input stream.
 * @param wrapper The Streamable wrapper containing the array and formatting options.
 * @return A reference to the input stream.
 */
template <typename T>
std::istream& operator>>(std::istream& input_stream, Streamable<T>& wrapper) {
    return wrapper.read(input_stream);
}

#endif //BICGSTAB_STREAMABLE_H
