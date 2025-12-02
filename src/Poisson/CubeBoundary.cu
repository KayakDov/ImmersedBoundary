//
// CubeBoundary.tpp
// Template definitions for CubeBoundary
//

#include "CubeBoundary.h"

#include <iostream>

#include "deviceArrays/headers/Streamable.h"

template<typename T>
CubeBoundary<T>::CubeBoundary(Mat<T>& frontBack_, Mat<T>& leftRight_, Mat<T>& topBottom_)
    : frontBack(frontBack_), leftRight(leftRight_), topBottom(topBottom_) {}

template<typename T>
size_t CubeBoundary<T>::size() const {
    return frontBack.size() + leftRight.size() + topBottom.size();
}

template<typename T>
size_t CubeBoundary<T>::internalSize() const {
    return (frontBack._rows / 2) * leftRight._cols * frontBack._cols;
}

template<typename T>
void CubeBoundary<T>::freeMem() {
    frontBack.freeMem();
    leftRight.freeMem();
    topBottom.freeMem();
}

template<typename T>
CubeBoundary<T> CubeBoundary<T>::ZeroTo1(const size_t dim, cudaStream_t stream) {
    return CubeBoundary<T>::ZeroTo1(dim, dim, dim, stream);
}

template<typename T>
CubeBoundary<T> CubeBoundary<T>::ZeroTo1(const size_t height, const size_t width, const size_t depth, cudaStream_t stream) {
    constexpr T frontFaceVal = 1;

    auto boundaries = Mat<T>::create(
        height * 4 + depth * 2,
        std::max(depth, width)
    );

    auto frontBack = boundaries.subMat(0, 0, 2 * height, width),
         leftRight = boundaries.subMat(frontBack._rows, 0, 2 * height, depth),
         topBottom = boundaries.subMat(frontBack._rows + leftRight._rows, 0, 2 * depth, width);

    auto front = frontBack.subMat(0, 0, height, width),
         back = frontBack.subMat(height, 0, height, width),
         top = topBottom.subMat(0, 0, depth, width),
         bottom = topBottom.subMat(depth, 0, depth, width);

    front.fill(frontFaceVal, stream);
    back.fill(0, stream);

    for (size_t layerInd = 0; layerInd < depth; ++layerInd) {
        T val = frontFaceVal * (static_cast<T>(layerInd) + static_cast<T>(1)) / (depth + static_cast<T>(1));
        leftRight.col(layerInd).fill(val, stream);
        top.row(layerInd).fill(val, stream);
        bottom.row(layerInd).fill(val, stream);
    }

    // std::cout << "CubeBoundary<T>::zeroTo1 front back\n" << GpuOut<T>(frontBack, stream) << std::endl;
    // std::cout << "CubeBoundary<T>::zeroTo1 left right\n" << GpuOut<T>(leftRight, stream) << std::endl;
    // std::cout << "CubeBoundary<T>::zeroTo1 top bottom\n" << GpuOut<T>(topBottom, stream) << std::endl;
    return CubeBoundary<T>(frontBack, leftRight, topBottom);
}

// Optional explicit instantiations if needed
template class CubeBoundary<float>;
template class CubeBoundary<double>;
