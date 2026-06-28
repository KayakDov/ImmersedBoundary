/**
 * @file Real3d.h
 * @brief Defines a lightweight three-component real-valued vector.
 * @ingroup math_utils
 *
 * @details
 * This header is part of the public interface and is included in the generated Doxygen module overview.
 */

#ifndef CUDABANDED_REAL3D_H
#define CUDABANDED_REAL3D_H

#include <cstddef>

#include "XYZ.cuh"

class Real3d : public XYZ<double>{

    public:

    Real3d(double x, double y, double z);


    static const Real3d ZERO;

    [[nodiscard]] double normInf(const Real3d& other) const;

    Real3d operator+(const Real3d& other) const;
    Real3d operator-(const Real3d& other) const;
    Real3d operator*(double scalar) const;
    double operator*(const Real3d& other) const;
};

class Real2d: public Real3d {
    public:
    Real2d(double x, double y);
};

#include <iostream>

inline std::ostream& operator<<(std::ostream& os, const Real3d& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";

}

inline std::ostream& operator<<(std::ostream& os, const Real2d& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}


#endif //CUDABANDED_REAL3D_H