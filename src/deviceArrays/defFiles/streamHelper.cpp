#include "../headers/deviceArrays.h"
#include <algorithm>
#include "../headers/deviceArraySupport.h"

template <typename T>
StreamHelper<T>::StreamHelper(size_t rows, size_t cols)
    : _totalCols(cols),
    _colsProcessed(0),
    _maxColsPerChunk(std::clamp(static_cast<size_t>((32ull * 1024ull * 1024ull) / (rows * sizeof(T))), size_t(1), size_t(cols))),
    _hostBuffer(_maxColsPerChunk * rows),    
    _rows(rows) {}

template <typename T>
StreamHelper<T>::~StreamHelper() = default;

template <typename T>
bool StreamHelper<T>::hasNext() const {
    return _colsProcessed < _totalCols;
}

template <typename T>
size_t StreamHelper<T>::getChunkWidth() const {
    return std::min(_maxColsPerChunk, _totalCols - _colsProcessed);
}

template <typename T>
std::vector<T>&  StreamHelper<T>::getBuffer() {
    return _hostBuffer;
}

template <typename T>
void StreamHelper<T>::updateProgress() {
    _colsProcessed += getChunkWidth();
}

template <typename T>
size_t StreamHelper<T>::getColsProcessed() const {
    return _colsProcessed;
}

// --- SetFromFile Definitions ---
template <typename T>
StreamSet<T>::StreamSet(size_t rows, size_t cols, std::istream& input_stream)
    : StreamHelper<T>(rows, cols), _input_stream(input_stream) {}

template <typename T>
void StreamSet<T>::readChunk(bool isRowMajor) {
    
    size_t num_elements = this->getChunkWidth() * this->_rows;
    size_t current_chunk_bytes = num_elements * sizeof(T);
    
    if(isRowMajor) {
        for (size_t i = 0; i < num_elements; ++i) 
            if (!(this->_input_stream >> this->_hostBuffer[i]))
                throw std::runtime_error("Failed to read enough elements. Failed at index " + std::to_string(i));
    } else this->_input_stream.read(reinterpret_cast<char*>(this->_hostBuffer.data()), current_chunk_bytes);

    if (!this->_input_stream) throw std::runtime_error("Stream read error or premature end of stream.");
    
}

// --- GetToFile Definitions ---
template <typename T>
StreamGet<T>::StreamGet(size_t rows, size_t cols, std::ostream& output_stream)
    : StreamHelper<T>(rows, cols), _output_stream(output_stream) {cudaDeviceSynchronize();}

template <typename T>
void StreamGet<T>::writeChunk(bool isText) {
    size_t num_elements = this->getChunkWidth() * this->_rows;
    size_t current_chunk_bytes = num_elements * sizeof(T);

    if (current_chunk_bytes > 0) {
        if(isText){
            for (size_t col = 0; col < this->getChunkWidth(); ++col) {
                for(size_t row = 0; row < this->_rows; ++row)
                    if (!(this->_output_stream << this->_hostBuffer[col * this->_rows + row] << '\t')) 
                        throw std::runtime_error("Failed to write enough elements");
                this->_output_stream << '\n';
            }                
                
        } 
        else this->_output_stream.write(reinterpret_cast<const char*>(this->_hostBuffer.data()), current_chunk_bytes);
        if (!this->_output_stream) throw std::runtime_error("Stream write error.");
    }
}

// CuFileHelper
template class StreamHelper<float>;
template class StreamHelper<double>;
template class StreamHelper<int32_t>;
template class StreamHelper<size_t>;
template class StreamHelper<unsigned char>;

// SetFromFile
template class StreamSet<float>;
template class StreamSet<double>;
template class StreamSet<int32_t>;
template class StreamSet<size_t>;
template class StreamSet<unsigned char>;

// GetToFile
template class StreamGet<float>;
template class StreamGet<double>;
template class StreamGet<int32_t>;
template class StreamGet<size_t>;
template class StreamGet<unsigned char>;
