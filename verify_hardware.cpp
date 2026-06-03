#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceId = 0;
    cudaDeviceProp prop;
    
    if (cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
        // Query the clock rate using the modern attribute API required by CUDA 13+
        int clockRatekHz = 0;
        cudaDeviceGetAttribute(&clockRatekHz, cudaDevAttrClockRate, deviceId);
        
        std::cout << "--- GPU Hardware Verification ---" << std::endl;
        std::cout << "Device Name: " << prop.name << std::endl;
        std::cout << "Streaming Multiprocessors (SMs): " << prop.multiProcessorCount << std::endl;
        std::cout << "Max Clock Rate: " << clockRatekHz / 1000.0 << " MHz" << std::endl;
        
        // For Ada Lovelace architecture (Compute Capability 8.9), there are 128 FP32 cores per SM
        int coresPerSM = 128; 
        int totalCores = prop.multiProcessorCount * coresPerSM;
        std::cout << "Calculated CUDA Cores: " << totalCores << std::endl;
        
        // Compute Peak FP32 FLOPS
        double clockHz = clockRatekHz * 1000.0; // convert kHz to Hz
        double peakFP32 = totalCores * 2.0 * clockHz;
        
        // Compute Peak FP64 FLOPS (1/64 execution rate scaling on consumer/workstation Ada)
        double peakFP64 = peakFP32 / 64.0;
        double fp64PerMs = peakFP64 / 1000.0;
        
        std::cout << "\n--- Theoretical Speed Limits ---" << std::endl;
        std::cout << "Peak FP32: " << peakFP32 / 1e12 << " TFLOPS" << std::endl;
        std::cout << "Peak FP64: " << peakFP64 / 1e9 << " GFLOPS" << std::endl;
        std::cout << "Your Excel Denominator (FP64 Ops/ms): " << fp64PerMs << std::endl;
    } else {
        std::cerr << "Failed to query CUDA device properties." << std::endl;
    }
    return 0;
}
