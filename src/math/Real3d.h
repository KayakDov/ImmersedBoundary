#ifndef CUDABANDED_REAL3D_H
#define CUDABANDED_REAL3D_H

#include <cstddef>

class Real3d {

    public:
    double x, y, z;
    Real3d(double x, double y, double z);

    double& operator[](size_t i);

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