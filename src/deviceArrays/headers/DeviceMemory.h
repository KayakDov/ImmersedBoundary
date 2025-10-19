#ifndef BICGSTAB_DEVICEMEMORY_H
#define BICGSTAB_DEVICEMEMORY_H

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <ostream>
#include <stddef.h>
#include <stdexcept>

constexpr double BYTES_PER_MB = 1024.0 * 1024.0;
constexpr double BYTES_PER_GB = BYTES_PER_MB * 1024.0;


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

// Global non-member function declaration for stream insertion
std::ostream& operator << (std::ostream& os, const DeviceMemory& dm);

#endif //BICGSTAB_DEVICEMEMORY_H
