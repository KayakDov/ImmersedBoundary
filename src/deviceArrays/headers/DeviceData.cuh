//
// Created by dov on 11/4/25.
//

#ifndef BICGSTAB_DEVICEDATA_CUH
#define BICGSTAB_DEVICEDATA_CUH

#include "GridDim.h"
#include <cstddef> // For size_t

class GridInd2d;
class GridInd3d;
template <typename T> class Vec;
template <typename T> class Tensor;
template <typename T> class GpuArray;

template<typename T>
class DeviceData1d {
    friend Vec<T>;
protected:

    DeviceData1d(size_t cols, size_t ld, T* data);

public:
    const size_t cols, ld;
    /**
     * The raw pointer to the data array, which **must** point to allocated
     * device memory (e.g., from cudaMalloc).
     */
    T* __restrict__ data;

    /**
     * @brief Provides 1D array access to the data using the logical flat index.
     * * This operator translates a logical 1D index (0 to rows*cols - 1)
     * into the physical memory index and returns a reference to the element.
     * @param idx The logical 1D index.
     * @return A writable reference to the element T at that index.
     */
    __device__ T& operator[](size_t idx);

    __host__ __device__ operator T*();
    __host__ __device__ operator const T*() const;
};

/**
 * @brief Represents a matrix/tensor data container designed for safe and efficient
 * passing to CUDA kernels.
 *
 * This struct encapsulates the raw device pointer and essential dimension
 * parameters (rows, columns, leading dimension) required for correct memory
 * access and indexing within GPU kernels. The structure itself is intended
 * to be passed BY VALUE to the kernel.
 *
 * @tparam T The type of data stored (e.g., float, double).
 */
template <typename T>
class DeviceData2d: public DeviceData1d<T>{  //TODO: get 1d and 3d versions to be used.
    friend GpuArray<T>;
protected:

    /**
     * @brief Constructor for initializing the DeviceData structure.
     * @param rows The number of rows.
     * @param cols The number of columns.
     * @param ld The leading dimension (physical row size).
     * @param data The device pointer to the allocated data.
     */
    __host__ __device__ DeviceData2d(size_t rows, size_t cols, size_t ld, T *data);
public:
    /** The number of logical rows in the matrix. */
    const size_t rows;

    /**
     * @brief Computes the 1D flat index from a 2D (row, col) coordinate,
     * accounting for the leading dimension (ld).
     * @param row The row index (0 to rows-1).
     * @param col The column index (0 to cols-1).
     * @return The 1D index into the data array.
     */
    __device__ size_t flat(size_t row, size_t col) const;
    /**
     * @brief Provides 1D array access to the data using the logical flat index.
     * * This operator translates a logical 1D index (0 to rows*cols - 1)
     * into the physical memory index and returns a reference to the element.
     * @param idx The logical 1D index.
     * @return A writable reference to the element T at that index.
     */
    __device__ T& operator[](size_t idx);

    /**
     * Accesses the place in this array pointed to by the index.
     * @param ind
     * @return
     */
    __device__ T& operator[](GridInd2d ind);

    /**
     * @brief Provides 2D matrix access to the data.
     * * This operator translates a 2D index (row, col) into the physical
     * memory index and returns a reference to the element.
     * @param row The row index (0 to rows-1).
     * @param col The column index (0 to cols-1).
     * @return A writable reference to the element T at (row, col).
     */
    __device__ T& operator()(size_t row, size_t col);

    /**
     * @brief Computes the 1D flat index from a logical 1D index (row-major style access),
     * translating it into the correct physically flattened index.
     * @param idx The logical 1D index (0 to rows*cols - 1).
     * @return The physically correct 1D index into the data array.
     */
    __device__ size_t flat(size_t idx) const;

    /**
     * @brief Computes the logical row index from a logical 1D index.
     * @param idx The logical 1D index (0 to rows*cols - 1).
     * @return The row index.
     */
    __device__ size_t row(size_t idx) const;

    /**
     * @brief Computes the logical column index from a logical 1D index.
     * @param idx The logical 1D index (0 to rows*cols - 1).
     * @return The column index.
     */
    __device__ size_t col(size_t idx) const;

};

template <typename T>
class DeviceData3d: public DeviceData2d<T> {
    friend Tensor<T>;
private:
    __device__ __host__ DeviceData3d(size_t rows, size_t cols, size_t layers, size_t ld, T *data);
public:
    const size_t layers;

    /**
     * Accesses the place in this array pointed to by the index.
     * @param ind
     * @return
     */
    __device__ T& operator[](GridInd3d ind);

};


/**
 * Checks if the indices point to a spot in the matrix.
 * @tparam T
 * @param ind
 * @param dim
 * @return
 */
template <typename T>
__device__ bool operator<(const GridInd3d& ind, const DeviceData3d<T>& dim);
/**
 * Are the indices outside the dimensions (inclusive).
 * @param ind
 * @param dim
 * @return
 */
template <typename T>
__device__ bool operator>=(const GridInd3d& ind, const DeviceData3d<T>& dim);
/**
 * Checks if the indices point to a spot in the matrix.
 * @tparam T
 * @param ind
 * @param dim
 * @return
 */
template <typename T>
__device__ bool operator<(const GridInd2d& ind, const DeviceData2d<T>& dim);


/**
 * Are the indices outside the dimensions (inclusive).
 * @param ind
 * @param dim
 * @return
 */
template <typename T>
__device__ bool operator>=(const GridInd2d& ind, const DeviceData2d<T>& dim);


#endif //BICGSTAB_DEVICEDATA_CUH