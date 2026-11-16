#include "../headers/DeviceMemory.h"
#include <iostream>
#include <iomanip> // Required for std::fixed and std::setprecision
#include <stdexcept>

// --- DeviceMemory Class Methods ---

DeviceMemory::DeviceMemory() {
    // Query memory info and store the error status in the constructor
    lastError = cudaMemGetInfo(&freeMem, &totalMem);

    // We intentionally do not throw here, allowing the operator<< to report the error status instead.
}


// --- Stream Insertion Operator Overload ---

std::ostream& operator << (std::ostream& os, const DeviceMemory& dm) {
    os << "\n==================================================\n";
    os << "Device Memory Status (Query Result: "
       << (dm.lastError == cudaSuccess ? "Success" : cudaGetErrorString(dm.lastError))
       << ")\n";
    os << "==================================================\n";

    if (dm.lastError == cudaSuccess) {
        size_t used_bytes = dm.totalMem - dm.freeMem;

        // Use a standard width and precision for clean output
        os << std::fixed << std::setprecision(2);

        os << "Total Memory: " << (double)dm.totalMem / BYTES_PER_GB << " GB (" << dm.totalMem << " Bytes)\n";
        os << "Free Memory:  " << (double)dm.freeMem / BYTES_PER_GB << " GB (" << dm.freeMem << " Bytes)\n";
        os << "Used Memory:  " << (double)used_bytes / BYTES_PER_GB << " GB (" << used_bytes << " Bytes)\n";
    } else {
        os << "ERROR: Failed to retrieve memory info. Status Code: " << dm.lastError << "\n";
    }
    os << "==================================================\n";
    return os;
}
