#include <iostream>
#include "deviceArrays.h"

int main(int argc, char const *argv[])
{
    // Check if a CUDA device is available.
    checkForDevice();
    
    
    CuArray1D<int> deviceArray(10);
    std::cin >> deviceArray; // Read data from standard input into the device array
    std::cout << "Device array contents: " << deviceArray << std::endl;


    // multiTest();
    
    return 0;
}