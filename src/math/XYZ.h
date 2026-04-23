//
// Created by usr on 4/23/26.
//

#ifndef CUDABANDED_XYZ_H
#define CUDABANDED_XYZ_H

#include <cstddef>

template<typename T>
struct XYZ {
    T x, y, z;

    XYZ(const T& x, const T& y, const T& z);

    T& operator[](size_t i);

    const T& operator[](size_t i) const;
};


#endif //CUDABANDED_XYZ_H
