/**
 * @file DeviceMemory.h
 * @brief Defines a utility for querying and reporting CUDA device memory.
 * @ingroup device_arrays
 *
 * @details
 * These classes form the core GPU container layer. Public containers generally own or view CUDA device memory through shared ownership, while explicit factory functions allocate new storage.
 */

#ifndef BICGSTAB_DEVICEMEMORY_H
#define BICGSTAB_DEVICEMEMORY_H

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <ostream>
#include <stddef.h>


/** Number of bytes in one mebibyte, used for memory reporting. */
constexpr double BYTES_PER_MB = 1024.0 * 1024.0;
/** Number of bytes in one gibibyte, used for memory reporting. */
constexpr double BYTES_PER_GB = BYTES_PER_MB * 1024.0;


/**
 * @brief Snapshot of the active CUDA device memory state.
 *
 * The constructor queries the active CUDA device and stores the free and total
 * memory values together with the CUDA status returned by the query.
 */
class DeviceMemory {
private:
    size_t freeMem = 0;
    size_t totalMem = 0;
    cudaError_t lastError = cudaSuccess;

public:
    /**
     * @brief Initializes DeviceMemory and immediately queries the CUDA device memory status.
     */
    DeviceMemory();

    /**
     * @brief Allows printing the memory status directly to an ostream (e.g., std::cout << myMemory).
     * This must be declared as a friend to access private members.
     */
    friend std::ostream& operator << (std::ostream& os, const DeviceMemory& dm);
};

/**
 * @brief Writes a human-readable CUDA memory report to a stream.
 * @param os Destination stream.
 * @param dm DeviceMemory snapshot to print.
 * @return @p os.
 */
std::ostream& operator << (std::ostream& os, const DeviceMemory& dm);

#endif //BICGSTAB_DEVICEMEMORY_H
