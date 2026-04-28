//
// Created by usr on 4/23/26.
//

#include "XYZ.h"
#include "deviceArrays/headers/SquareMat.h"

template<typename T>
XYZ<T>::XYZ(const T &x, const T &y, const T &z)
        : x(x), y(y), z(z) {}

template<typename T>
T & XYZ<T>::operator[](size_t i) {
        return i == 0 ? x : (i == 1 ? y : z);
}

template<typename T>
const T & XYZ<T>::operator[](size_t i) const  {
        return i == 0 ? x : (i == 1 ? y : z);
}

template<typename T>
XYZ<T> XYZ<T>::fill(T i) {
        return XYZ<T>(i, i, i);
}


template class XYZ<Vec<double>>;
template class XYZ<Mat<double>>;
template class XYZ<SquareMat<double>>;
template class XYZ<DeviceData1d<double>>;
template class XYZ<DeviceData2d<double>>;


template class XYZ<Vec<float>>;
template class XYZ<Mat<float>>;
template class XYZ<SquareMat<float>>;
template class XYZ<DeviceData1d<float>>;
template class XYZ<DeviceData2d<float>>;

template class XYZ<double>;
template class XYZ<float>;
template class XYZ<bool>;