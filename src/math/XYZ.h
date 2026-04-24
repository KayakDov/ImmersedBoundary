//
// Created by usr on 4/23/26.
//

#ifndef CUDABANDED_XYZ_H
#define CUDABANDED_XYZ_H

#include <cstddef>

/**
 * Holds data for each dimension, x, y, and z.
 * @tparam T
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
    XYZ(const T& x, const T& y, const T& z);

    /**
     * 0 gets the x value, 1 gets the y value, and 2 gets the z value.
     * @param i The index of the desired value.
     * @return
     */
    T& operator[](size_t i);

    /**
     * 0 gets the x value, 1 gets the y value, and 2 gets the z value.
     * @param i The index of the desired value.
     * @return
     */
    const T& operator[](size_t i) const;

    /**
     * creates an instance where the value in all the dimensions is uniform.
     * @param i The value for each dimension.
     * @return A new instance with uniform values.
     */
    static XYZ<T>fill(T i);
};


#endif //CUDABANDED_XYZ_H
