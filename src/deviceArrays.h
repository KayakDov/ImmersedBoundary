// --- deviceArrays.h ---
// This file declares the classes and functions used in deviceArrays.cu.
// It is included by main.cu to let the compiler know what exists.
#ifndef DEVICEARRAYS_H
#define DEVICEARRAYS_H

#include <vector>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <string>

// Forward declarations of classes to avoid circular includes
template <typename T> class CuArray;
template <typename T> class CuArray1D;
template <typename T> class CuArray2D;
template <typename T> class CuFileHelper;
template <typename T> class SetFromFile;
template <typename T> class GetToFile;

// Enum for index type
enum class IndexType;

// CUDA device memory deleter function for std::shared_ptr.
inline void cudaFreeDeleter(void* ptr);

// Helper function to check for CUDA errors and exit on failure.
void checkCudaErrors(cudaError_t err, const char* file, int line);
// Macro to wrap CUDA function calls for easy error checking.
#define CHECK_CUDA_ERROR(err) checkCudaErrors(err, __FILE__, __LINE__)

// A helper function to verify if two vectors are identical.
template <typename T>
void verifyVectors(const std::vector<T>& expected, const std::vector<T>& result, const std::string& test_name);

// Functions to run tests
void checkForDevice();
template <typename T>
void runTests();
template <typename T>
void runFileIOTests();
void multiTest();

// --- CuFileHelper, SetFromFile, GetToFile class declarations ---
template <typename T>
class CuFileHelper {
public:
    const size_t _totalCols;
    const size_t _maxColsPerChunk; 
    const size_t _rows;
protected:    
    size_t _colsProcessed;
    std::vector<T> _hostBuffer;
public:
    
    CuFileHelper(size_t rows, size_t cols);
    virtual ~CuFileHelper();
    bool hasNext() const;
    size_t getNextChunkColNumber() const;
    T* getHostBuffer();
    void updateProgress();
    size_t getColsProcessed() const;
};

template <typename T>
class SetFromFile : public CuFileHelper<T> {
private:
    std::istream& _input_stream;
public:
    SetFromFile(size_t rows, size_t cols, std::istream& input_stream);
    void readNextChunk();
};

template <typename T>
class GetToFile : public CuFileHelper<T> {
private:
    std::ostream& _output_stream;
public:
    GetToFile(size_t rows, size_t cols, std::ostream& output_stream);
    void writeNextChunkToFile();
};


// --- CuArray base class and derived classes declarations ---
enum class IndexType {
    Row,
    Column
};

template <typename T>
class CuArray {
public:
    const size_t _rows;
    const size_t _cols;
protected:
    std::shared_ptr<void> _ptr;
    size_t _ld;
    CuArray(size_t rows, size_t cols, size_t ld);
public:
    virtual ~CuArray();
    virtual size_t size() const = 0;
    virtual size_t bytes() const = 0;
    virtual void set(const T* hostData, cudaStream_t stream = 0) = 0;
    virtual void get(T* hostData, cudaStream_t stream = 0) const = 0;
    virtual void set(const CuArray<T>& src, cudaStream_t stream = 0) = 0;
    virtual void get(CuArray<T>& dst, cudaStream_t stream = 0) const = 0;
    virtual void set(std::istream& input_stream, cudaStream_t stream = 0) = 0;
    virtual void get(std::ostream& output_stream, cudaStream_t stream = 0) const = 0;
    T* data();
    const T* data() const;
    size_t getLD() const;
    std::shared_ptr<void> getPtr() const;
};

template <typename T>
class CuArray2D : public CuArray<T> {
public:
    CuArray2D(size_t rows, size_t cols);
    CuArray2D(const CuArray2D<T>& superArray, size_t startRow, size_t startCol, size_t height, size_t width);
    size_t size() const override;
    size_t bytes() const override;
    void set(const T* src, cudaStream_t stream = 0) override;
    void get(T* dst, cudaStream_t stream = 0) const override;
    void set(const CuArray<T>& src, cudaStream_t stream = 0) override;
    void get(CuArray<T>& dst, cudaStream_t stream = 0) const override;
    void set(std::istream& input_stream, cudaStream_t stream = 0) override;
    void get(std::ostream& output_stream, cudaStream_t stream = 0) const override;
};

template <typename T>
class CuArray1D : public CuArray<T> {
public:
    explicit CuArray1D(size_t length);
    CuArray1D(const CuArray1D<T>& superArray, size_t offset, size_t length, size_t stride = 1);
    CuArray1D(const CuArray2D<T>& extractFrom, int index, IndexType indexType);
    size_t size() const override;
    size_t bytes() const override;
    void set(const T* hostData, cudaStream_t stream = 0) override;
    void get(T* hostData, cudaStream_t stream = 0) const override;
    void set(const CuArray<T>& src, cudaStream_t stream = 0) override;
    void get(CuArray<T>& dst, cudaStream_t stream = 0) const override;
    void set(std::istream& input_stream, cudaStream_t stream = 0) override;
    void get(std::ostream& output_stream, cudaStream_t stream = 0) const override;
};

#endif // DEVICEARRAYS_H