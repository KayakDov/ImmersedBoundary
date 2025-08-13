#include <iostream>
#include "deviceArrays.h"

int main(int argc, char const *argv[])
{
    // Check if a CUDA device is available.
    checkForDevice();
    
    // Run the tests you have defined in deviceArrays.cu.
    multiTest();
    
    return 0;
}