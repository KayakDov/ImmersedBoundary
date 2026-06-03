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
        
        // 1. CUDA Cores (For Ada Lovelace architecture 8.9, there are 128 FP32 cores per SM)
        int coresPerSM = 128; 
        int totalCores = prop.multiProcessorCount * coresPerSM;
        
        // 2. Operations per Clock Cycle
        // A Fused Multiply-Add (FMA) instruction does a multiplication and addition in one step
        int opsPerClock = 2; 
        
        // 3. Clock Frequency in Hz
        double clockHz = clockRatekHz * 1000.0; 
        
        // 4. FP64 Precision Ratio 
        // Consumer/Laptop Ada Lovelace chips restrict FP64 hardware to 1/64th of the FP32 pipeline
        int fp64RatioDenominator = 64; 
        
        // --- Perform the Calculations ---
        double peakFP32 = totalCores * opsPerClock * clockHz;
        double peakFP64 = peakFP32 / fp64RatioDenominator;
        double fp64PerMs = peakFP64 / 1000.0;
        
        // --- Print the Equation Breakdown ---
        std::cout << "\n--- Equation Components ---" << std::endl;
        std::cout << "1. CUDA Cores:                 " << totalCores << std::endl;
        std::cout << "2. Operations per Clock Cycle: " << opsPerClock << " (FMA)" << std::endl;
        std::cout << "3. Clock Frequency (Hz):       " << clockHz << std::endl;
        std::cout << "4. FP64 Precision Ratio:       1/" << fp64RatioDenominator << std::endl;
        
        std::cout << "\n--- Calculation Breakdown ---" << std::endl;
        std::cout << "Peak FP32 = " << totalCores << " * " << opsPerClock << " * " << clockHz << std::endl;
        std::cout << "          = " << peakFP32 << " FLOPS" << std::endl;
        
        std::cout << "Peak FP64 = Peak FP32 * (1 / " << fp64RatioDenominator << ")" << std::endl;
        std::cout << "          = " << peakFP64 << " FLOPS" << std::endl;
        
        std::cout << "\n--- Theoretical Speed Limits ---" << std::endl;
        std::cout << "Peak FP32: " << peakFP32 / 1e12 << " TFLOPS" << std::endl;
        std::cout << "Peak FP64: " << peakFP64 / 1e9 << " GFLOPS" << std::endl;
        std::cout << "Your Excel Denominator (FP64 Ops/ms): " << fp64PerMs << std::endl;
        
    } else {
        std::cerr << "Failed to query CUDA device properties." << std::endl;
    }
    return 0;
}
