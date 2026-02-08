//
// Created by usr on 2/8/26.
//

#include "Thomas.cuh"

template<typename Real>
size_t numSystems(Mat<Real>& heightX_7XnumSystems) {
    return heightX_7XnumSystems._cols/7;
}

template<typename Real>
size_t endOfLHS(Mat<Real>& heightX_7XnumSystems) {
    return numSystems(heightX_7XnumSystems) * 3;
}

template<typename Real>
Thomas<Real>::Thomas(Mat<Real>& heightX_7XnumSystems):
    heightX_3XnumSystemsPlus4(heightX_7XnumSystems),
    triDiags(heightX_7XnumSystems.subMat(
        0,
        0,
        heightX_7XnumSystems._rows,
        endOfLHS(heightX_7XnumSystems).tensor(numSystems(heightX_7XnumSystems))
    )),
    x(heightX_7XnumSystems.subMat(0, endOfLHS(heightX_7XnumSystems), heightX_7XnumSystems._rows, numSystems(heightX_7XnumSystems))),
    b(heightX_7XnumSystems.subMat(0, endOfLHS(heightX_7XnumSystems) + heightX_7XnumSystems._rows, numSystems(heightX_7XnumSystems), heightX_7XnumSystems._rows, numSystems(heightX_7XnumSystems))),
    cPrime(heightX_7XnumSystems.subMat(0, endOfLHS(heightX_7XnumSystems) + 2 * heightX_7XnumSystems._rows, numSystems(heightX_7XnumSystems), heightX_7XnumSystems._rows, numSystems(heightX_7XnumSystems))),
    dPrime(heightX_7XnumSystems.subMat(0, endOfLHS(heightX_7XnumSystems) + 3 * heightX_7XnumSystems._rows, numSystems(heightX_7XnumSystems), heightX_7XnumSystems._rows, numSystems(heightX_7XnumSystems)))
{}

template<typename Real>
void Thomas<Real>::set(Real *triDiagonals, Real *b, Handle& hand) {
    this->triDiags.set(triDiagonals, hand);
    this->b.set(b, hand);
}
//https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
template<typename Real>
__global__ void solveKernel(DeviceData3d<Real> triDiags, DeviceData2d<Real> x, DeviceData2d<Real> b, DeviceData2d<Real> cPrime, DeviceData2d<Real> dPrime) {

    if (size_t system = idx(); system < triDiags.layers) {
        const Real c0 = triDiags(0, 2, system);
        const Real b0 = triDiags(0, 1, system);
        const Real d0 = b(0,system)/triDiags(0, 1, system);
        cPrime(0, system) = c0/b0;
        dPrime(0, system) = d0/b0;
        for (size_t row = 1; row < triDiags.rows; row++) {
            const Real aRow = triDiags(row, 0, system);
            const Real bRow = triDiags(row, 1, system);
            const Real cRow = triDiags(row, 2, system);
            const Real dRow = b(row,system);

            cPrime(row, system) = cRow/(bRow-aRow*cPrime(row - 1, system));
            dPrime(row, system) = (dRow - aRow * dPrime(row - 1, system)) / (bRow - aRow*cPrime(row - 1, system));
        }
        size_t n = triDiags._rows - 1;
        x(n, system) = dPrime(n, system);
        for (size_t row = n - 1; row >= 0; row--)
            x(row, system) = dPrime(row, system) - cPrime(row, system) * x(row + 1, system);
    }
}


template<typename Real>
void Thomas<Real>::solve(Handle &hand) {
}
