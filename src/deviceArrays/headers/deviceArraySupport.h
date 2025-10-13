
#ifndef BICGSTAB_DEVICEARRAYSUPPORT_H
#define BICGSTAB_DEVICEARRAYSUPPORT_H

#include <vector>
/**
 * @brief Utility for incremental streaming of large column-major 2D data.
 *
 * StreamHelper and its derived classes allow large matrices or tabular data
 * to be processed in fixed-size column chunks. This avoids loading the entire
 * dataset at once, improving memory efficiency when working with large files.
 *
 * These classes are format-agnostic and support both text (formatted) and
 * binary streaming modes.
 */
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

    /**
    * @brief Construct a StreamHelper for a 2D dataset.
    * @param rows Number of rows in the dataset
    * @param cols Number of columns in the dataset
    *
    * The constructor computes a memory-friendly chunk size based on the dataset dimensions.
    */
    StreamHelper(size_t rows, size_t cols);

    /**
     * @brief Default destructor.
     */
    virtual ~StreamHelper();

    /**
     * @brief Check whether there are remaining columns to process.
     * @return True if more columns remain, false otherwise.
     */
    [[nodiscard]] bool hasNext() const;

    /**
     * @brief Get the number of columns in the current chunk.
     * @return Width of the current chunk in columns.
     */
    [[nodiscard]] size_t getChunkWidth() const;

    /**
     * @brief Advance the internal progress counter by the current chunk width.
     *
     * Must be called after processing a chunk to ensure the next chunk is correctly indexed.
     */
    void updateProgress();

    /**
     * @brief Get the total number of columns processed so far.
     * @return Number of columns already processed.
     */
    [[nodiscard]] size_t getColsProcessed() const;

    /**
     * @brief Access the internal host buffer for the current chunk.
     * @return Reference to the host buffer.
     *
     * This buffer is used for reading/writing one chunk at a time.
     */
    std::vector<T>& getBuffer();
};

/**
 * @brief StreamHelper specialization for reading chunks from an input stream.
 *
 * Supports both text-formatted and binary data. Reads one chunk at a time into
 * the internal host buffer.
 *
 * @tparam T Type of elements being read.
 */
template <typename T>
class StreamSet : public StreamHelper<T> {
private:
    std::istream& _input_stream;
public:
    /**
    * @brief Construct a StreamSet for reading data.
    * @param rows Number of rows in the dataset
    * @param cols Number of columns in the dataset
    * @param input_stream Input stream to read from
    */
    StreamSet(size_t rows, size_t cols, std::istream& input_stream);

    /**
     * @brief Read the next chunk from the stream into the host buffer.
     * @param isRowMajor If true, interprets the stream as text and reads elements sequentially. If false, reads binary.
     *
     * @throws std::runtime_error if reading fails or the stream ends prematurely.
     */
    void readChunk(bool isText);
};

/**
 * @brief StreamHelper specialization for writing chunks to an output stream.
 *
 * Supports both text-formatted and binary data. Writes one chunk at a time from
 * the internal host buffer.
 *
 * @tparam T Type of elements being written.
 */
template <typename T>
class StreamGet : public StreamHelper<T> {
private:
    std::ostream& _output_stream;
public:
    /**
    * @brief Construct a StreamGet for writing data.
    * @param rows Number of rows in the dataset
    * @param cols Number of columns in the dataset
    * @param output_stream Output stream to write to
    *
    * @note Synchronizes the CUDA device at construction; optional for CPU-only usage.
    */
    StreamGet(size_t rows, size_t cols, std::ostream& output_stream);

    /**
    * @brief Write the current chunk from the host buffer to the stream.
    * @param isText If true, writes in text format; otherwise writes binary.
    *
    * @throws std::runtime_error if writing fails.
    */
    void writeChunk(bool isText);
};

/**
 * @brief Macro for checking cuSOLVER status and throwing runtime errors.
 * @param func cuSOLVER function call
 *
 * @throws std::runtime_error if the cuSOLVER call does not return CUSOLVER_STATUS_SUCCESS.
 */
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