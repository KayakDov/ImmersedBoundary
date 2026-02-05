#ifndef KERNELPREP_H
#define KERNELPREP_H

#include <cuda_runtime.h>
#include <stdexcept>

/**
 * @brief Utility class to calculate optimal CUDA grid and block dimensions for kernel launches.
 * * Provides methods to set up dim3 structures for 1D, 2D, and 3D data processing.
 */
class KernelPrep {

    static constexpr size_t DEFAULT_BLOCK_SIZE_1D = 256;
    static constexpr size_t DEFAULT_BLOCK_SIZE_2D_SIDE = 16;
    static constexpr size_t DEFAULT_BLOCK_SIZE_3D_SIDE = 8;

    static const cudaDeviceProp& deviceProps() {
        static cudaDeviceProp prop = []{
            int device = 0;
            cudaDeviceProp p{};
            if (cudaGetDeviceProperties(&p, device) != cudaSuccess) {
                throw std::runtime_error("Failed to query CUDA device properties");
            }
            return p;
        }();
        return prop;
    }

public:
    static size_t maxThreadsPerBlock() {
        return deviceProps().maxThreadsPerBlock;
    }

    static size_t maxNumBlocksX() {
        return deviceProps().maxGridSize[0];
    }

    static size_t maxNumBlocksY() {
        return deviceProps().maxGridSize[1];
    }

    static size_t maxNumBlocksZ() {
        return deviceProps().maxGridSize[2];
    }
    // KernelPrep.cuh or a shared constants header
    static constexpr size_t CUDA_MAX_GRID_X = 2147483647u;
    static constexpr size_t CUDA_MAX_GRID_Y = 65535u;
    static constexpr size_t CUDA_MAX_GRID_Z = 64u;
    static constexpr size_t CUDA_MAX_BLOCK_THREADS = 1024u;

    /** @brief The calculated grid dimensions (number of blocks). */
    dim3 numBlocks;

    /** @brief The calculated block dimensions (threads per block). */
    dim3 threadsPerBlock;

    // --- Constructors / Configuration Methods ---

    /**
     * @brief Constructs a KernelPrep object and calculates 1D grid/block dimensions.
     * @param count The total number of elements to process.
     */
    explicit KernelPrep(size_t count);

    /**
     * @brief Constructs a KernelPrep object and calculates 2D grid/block dimensions.
     * @param width The width (X dimension) of the data.
     * @param height The height (Y dimension) of the data.
     * @param transpose Set transpose to true to flip x and y dimensions, so that column index is the y dimension
     * and row index is the x dimension.  This is useful when the y dimension is much longer than the x dimension,
     * on account of gpu limits.
     */
    KernelPrep(size_t width, size_t height, bool transpose = false);

    /**
     * @brief Constructs a KernelPrep object and calculates 3D grid/block dimensions.
     * @param width The width (X dimension) of the data.
     * @param height The height (Y dimension) of the data.
     * @param depth The depth (Z dimension) of the data.
     */
    KernelPrep(size_t width, size_t height, size_t depth);

    /**
     * @brief Utility function to perform ceiling division (a / b).
     * @param numerator The total size (a).
     * @param denominator The block size (b).
     * @return The number of blocks needed (ceil(a / b)).
     */
    static size_t ceiling_division(size_t numerator, size_t denominator) {
        return (numerator + denominator - 1) / denominator;
    }

};

#endif // KERNELPREP_H
