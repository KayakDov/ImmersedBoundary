//
// Created by usr on 12/26/25.
//

#include "Real3d.h"

#include <stdexcept>

Real3d::Real3d(double x, double y, double z) : XYZ<double>(x, y, z) {
}

const Real3d Real3d::ZERO(0, 0, 0);

double Real3d::normInf(const Real3d &other) const {
    return std::max(std::max(std::abs(x), std::abs(y)), std::abs(z));
}

Real3d Real3d::operator+(const Real3d &other) const {
    return {x + other.x, y + other.y, z + other.z};
}

Real3d Real3d::operator-(const Real3d &other) const {
    return {x - other.x, y - other.y, z - other.z};
}

Real3d Real3d::operator*(double scalar) const {
    return {x * scalar, y * scalar, z * scalar};
}

double Real3d::operator*(const Real3d &other) const {
    return x * other.x + y * other.y + z * other.z;
}


Real2d::Real2d(double x, double y): Real3d(x, y, 0) {
}
