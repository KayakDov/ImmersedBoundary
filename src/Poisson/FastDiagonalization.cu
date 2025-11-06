
#include "deviceArrays/headers/Mat.h"        // 1. BASE CLASS MUST BE INCLUDED FIRST
#include "deviceArrays/headers/SquareMat.h"   // 2. DERIVED CLASS SECOND
#include "deviceArrays/headers/Vec.h"
#include "Poisson.h"                        // 3. This indirect path is now safe

#include <cmath>
#include <iostream>
#include <array>

template <typename T>
__device__ __host__ constexpr T PI = static_cast<T>(3.14159265358979323846);

template <typename T>
__global__ void eiganMatLKernel(DeviceData2d<T> eVecs) {
    if (const GridInd2d ind; ind < eVecs)
        eVecs[ind] = std::sqrt(2/static_cast<T>(eVecs.rows + 1)) * std::sin((ind.col  + 1) * (ind.row + 1) * PI<T> / static_cast<T>(eVecs.rows + 1));
}

template <typename T>
__global__ void eiganValLKernel(DeviceData2d<T> eVals) {
    if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < eVals.cols) {
        T sinExpr = std::sin((idx  + 1) * PI<T> / (2*(eVals.cols + 1)));
        eVals[idx] = -4 * sinExpr * sinExpr;
    }
}

template <typename T>
__global__ void setUTildeKernel(DeviceData3d<T> uTilde, const DeviceData2d<T> eVals, const DeviceData2d<T> fTilde) {

    if (GridInd3d ind; ind < uTilde)
        uTilde[ind] = fTilde[ind]/(eVals(0, ind.col) + eVals(1, ind.row) + eVals(2,ind.layer));
}

template <typename T>
class FastDiagonalization: public Poisson<T>{
private:

    /**
     * Sets the eigen matrix for L[i].
     * @param i The index of the desired matrix, 1,2, or 3.
     * @param stream
     */
    void eigenL(size_t i, cudaStream_t stream) { //TODO: this method could run both kernels on different streams.  Also different streams for each index.  Also, I don't use L, so that should be remvoved.
        constexpr dim3 blockDim(16, 16);

        const dim3 gridDim3d(
            (eVecs[i]._cols + blockDim.x - 1) / blockDim.x,
            (eVecs[i]._rows + blockDim.y - 1) / blockDim.y
        );
        eiganMatLKernel<T><<<gridDim3d, blockDim, 0, stream>>>(eVecs[i].toKernel(), eVecs[i]._rows, eVecs[i]._ld);

        constexpr size_t gridDim1d = 256;

        eiganValLKernel<T><<<(eVecs[i]._cols + gridDim1d - 1) / gridDim1d, gridDim1d, 0, stream>>>(eVals.col(i).toKernel(), eVecs[i]._cols);
    }

    void setUTilde(Tensor<T> f, Tensor<T> u, cudaStream_t stream) {
        constexpr dim3 blockDim(16, 16, 16);

        const dim3 gridDim3d(
            (f._cols + blockDim.x - 1) / blockDim.x,
            (f._rows + blockDim.y - 1) / blockDim.y,
            (f._layers + blockDim.z - 1) / blockDim.z
        );
        setUTildeKernel<<<gridDim3d, blockDim, 0, stream>>>(u.toKernel3d(), eVals.toKernel(), f.toKernel3d());
    }

    std::array<SquareMat<T>, 3> eVecs;//Note: transpose is the inverse for these matrices.
    Mat<T> eVals;//The eigenvalues for L[i] are stored in column i.

    /**
     * The kronecker product of a, b, and c
     * @param a
     * @param b
     * @param c
     * @param temp should be a.rows * b.rows x a.cols * b.cols
     * @param result should be a.rows * b.rows * c.rows x a.cols * b.cols * c.cols
     * @param stream
     */
    void kronecker3(const Mat<T>& a, const Mat<T>& b, const Mat<T>& c, Mat<T>& temp, Mat<T>& result, cudaStream_t stream) {
        temp.fill(0, stream);
        a.multKronecker(b, temp, stream);
        temp.multKronecker(c, result, stream);
    }

    std::array<SquareMat<T>, 3> generateSquareMat() const{
        return {SquareMat<T>::create(this->_cols), SquareMat<T>::create(this->_rows),  SquareMat<T>::create(this->_layers)};
    }

    void multiplyEF(Handle& hand, Tensor<T>& f, bool transposeE) {
        Mat<T>::batchMult(Singleton<T>::ONE,
            eVecs[0], 0,
            f.layerRowCol(0), f.layerSize(),
            Singleton<T>::ZERO, f.layerRowCol(0), f.layerSize(),
            transposeE, true, hand,
            f._layers
            );
        Mat<T>::batchMult(Singleton<T>::ONE,
            eVecs[1], 0,
            f.layerRowCol(0), f.layerSize(),
            Singleton<T>::ZERO, f.layerRowCol(0), f.layerSize(),
            transposeE, false, hand, f._layers
            );
        Mat<T>::batchMult(Singleton<T>::ONE,
            eVecs[2], 0,
            f.layerColDepth(0), f._rows,
            Singleton<T>::ZERO, f.layerColDepth(0), f._rows,
            transposeE, true, hand, f._cols
            );
    }
public:
    /**
     *
     * @param boundary The boundary conditions.
     * @param x The solution will be placed here.  This should have a space for every element in the grid.
     * @param f The RHS of the laplace equation.  Whatever you have written here, we will overwrite it. Thi should have
     * a space for every element in the grid.
     * @param stream
     */
    FastDiagonalization(const CubeBoundary<T>& boundary, Vec<T>& x, Vec<T>& f, Handle hand) ://TODO: provide pre alocated memory
        Poisson<T>(boundary, f, hand.stream),
        eVecs(generateSquareMat()),
        eVals(Mat<T>::create(std::max(this->_rows, std::max(this->_cols, this->_layers))), 3)
    {
        Mat<T> temp = SquareMat<T>::create(std::max(this->_rows,this->_cols));

        for (size_t i = 0; i < 3; ++i) eigenL(i, hand.stream);

        multiplyEF(hand, f.tensor(this->_rows, this->_cols), true);
        setUTilde(f, x, hand.stream);
        multiplyEF(hand, x.tensor(this->_rows, this->_cols), false);
    }
};


/**
 * @brief Main entry point to demonstrate the FastDiagonalizationMethod for a 2x2x2 grid.
 */
int main() {

    Handle handle;

    // auto a = Mat<double>::create(2, 2);
    // auto b = Mat<double>::create(2, 2);
    // auto c = Mat<double>::create(2, 2);
    // a.fill(1, handle.stream);
    // b.fill(2, handle.stream);
    //
    // a.get(std::cout << "\nA = \n", true, false, &handle);
    // b.get(std::cout << "\nB = \n", true, false, &handle);
    //
    // a.mult(b, &c, &handle, &Singleton<double>::ONE, &Singleton<double>::ZERO, true,true);
    //
    // c.get(std::cout << "\nC = \n", true, false, &handle);

    constexpr size_t dim = 2;
    Handle hand;

    auto boundary = CubeBoundary<double>::ZeroTo1(dim, hand.stream);
    auto x = Vec<double>::create(boundary.internalSize(), hand.stream);
    auto f = Vec<double>::create(boundary.internalSize(), hand.stream);
    f.fill(0, hand.stream);

    FastDiagonalization<double> fdm(boundary, x, f, hand);

    x.get(std::cout << "x = \n", true, false, &hand);

    return 0;
}



// void setL(size_t i, cudaStream_t stream) {//TODO: try parallelizing this data
//     L[i].fill(static_cast<T>(0), stream);
//     L[i].diag(0).fill(static_cast<T>(-2), stream);
//     L[i].diag(1).fill(static_cast<T>(1), stream);
//     L[i].diag(-1).fill(static_cast<T>(1), stream);
// }
