#include <iostream>
#include "deviceArrays.h"
#include "testMethods.cu"

int main(int argc, char const *argv[])
{
    // Check if a CUDA device is available.
    checkForDevice();
    
    runTests<double>();

    // multiTest();
    
    return 0;
}