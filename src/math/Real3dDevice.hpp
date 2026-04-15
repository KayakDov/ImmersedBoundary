#ifndef CUDABANDED_REAL3DDEVICE_CUH
#define CUDABANDED_REAL3DDEVICE_CUH
#include <cuda_runtime.h>

template<typename Real>
class Real3dDevice {
    public:
    Real x, y, z;
    __host__ __device__ Real3dDevice(Real x, Real y, Real z) : x(x), y(y), z(z) {}
};

#endif //CUDABANDED_REAL3DDEVICE_CUH
