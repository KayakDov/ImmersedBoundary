
#ifndef BICGSTAB_DEVICEARRAYSUPPORT_H
#define BICGSTAB_DEVICEARRAYSUPPORT_H

#include <vector>

template <typename T>
class StreamHelper {
public:
    const size_t _totalCols;
    const size_t _maxColsPerChunk;
    const size_t _rows;
protected:
    size_t _colsProcessed;
    std::vector<T> _hostBuffer;
public:

    StreamHelper(size_t rows, size_t cols);
    virtual ~StreamHelper();
    [[nodiscard]] bool hasNext() const;
    [[nodiscard]] size_t getChunkWidth() const;
    void updateProgress();
    [[nodiscard]] size_t getColsProcessed() const;
    std::vector<T>& getBuffer();
};

template <typename T>
class StreamSet : public StreamHelper<T> {
private:
    std::istream& _input_stream;
public:
    StreamSet(size_t rows, size_t cols, std::istream& input_stream);
    void readChunk(bool isText);
};

template <typename T>
class StreamGet : public StreamHelper<T> {
private:
    std::ostream& _output_stream;
public:
    StreamGet(size_t rows, size_t cols, std::ostream& output_stream);
    void writeChunk(bool isText);
};


#define CHECK_CUSOLVER_ERROR(func) \
do { \
cusolverStatus_t status = (func); \
if (status != CUSOLVER_STATUS_SUCCESS) { \
throw std::runtime_error( \
"cuSOLVER error: " + std::to_string(status) + \
" at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) \
); \
} \
} while (0)


#endif //BICGSTAB_DEVICEARRAYSUPPORT_H