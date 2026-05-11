//
// Created by usr on 4/23/26.
//

#ifndef CUDABANDED_XYZ_H
#define CUDABANDED_XYZ_H

#include <cuda_runtime.h>
#include <ostream>


/**
 * Holds data for each dimension, x, y, and z.
 * @tparam T
 * TODO: pass this class to kernels.
 */
template<typename T>
struct XYZ {
    T x, y, z;

    /**
     * Constructs a new instance.
     * @param x The value in the x dimension.
     * @param y The value in the y dimension.
     * @param z The value in the z dimension.
     */
    XYZ(const T& x, const T& y, const T& z): x(x), y(y), z(z) {}

    /**
     * 0 gets the x value, 1 gets the y value, and 2 gets the z value.
     * @param i The index of the desired value.
     * @return
     */
    __host__ __device__ T& operator[](size_t i) {
        return i == 0 ? x : (i == 1 ? y : z);
    }

    /**
     * 0 gets the x value, 1 gets the y value, and 2 gets the z value.
     * @param i The index of the desired value.
     * @return
     */
    __host__ __device__ const T& operator[](size_t i) const {
        return i == 0 ? x : (i == 1 ? y : z);
    };

    /**
     * creates an instance where the value in all the dimensions is uniform.
     * @param i The value for each dimension.
     * @return A new instance with uniform values.
     */
    __host__ __device__ static XYZ<T>fill(T i) {
        return XYZ<T>(i, i, i);
    }

    // Overloaded << operator
    friend std::ostream& operator<<(std::ostream& os, const XYZ<T>& obj) {
        os << "(" << obj.x << ", " << obj.y << ", " << obj.z << ")";
        return os;
    }
};


#endif //CUDABANDED_XYZ_H
