
#include "deviceArrays/headers/Mat.h"        // 1. BASE CLASS MUST BE INCLUDED FIRST
#include "deviceArrays/headers/SquareMat.h"   // 2. DERIVED CLASS SECOND
#include "deviceArrays/headers/Vec.h"
#include "Poisson.h"                        // 3. This indirect path is now safe

#include <cmath>
#include <iostream>
#include <array>

template <typename T>
__device__ constexpr T PI = static_cast<T>(3.14159265358979323846);

template <typename T>
__global__ void eiganMatLKernel(DeviceData2d<T> eVecs) {
    if (const GridInd2d ind; ind < eVecs)
        eVecs[ind] = std::sqrt(2/static_cast<T>(eVecs.rows + 1)) * std::sin((ind.col  + 1) * (ind.row + 1) * PI<T> / static_cast<T>(eVecs.rows + 1));
}

template <typename T>
__global__ void eiganValLKernel(DeviceData1d<T> eVals) {
    if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < eVals.cols) {
        T sinExpr = std::sin((idx  + 1) * PI<T> / (2*(eVals.cols + 1)));
        eVals[idx] = -4 * sinExpr * sinExpr;
    }
}

template <typename T>
__global__ void setUTildeKernel(DeviceData3d<T> uTilde, const DeviceData2d<T> eVals, const DeviceData3d<T> fTilde) {

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
    void eigenL(size_t i, cudaStream_t stream) { //TODO: this method could run on different streams.  Also different streams for each index.  Also, I don't use L, so that should be remvoved.

        KernelPrep kpVec = eVecs[i].kernelPrep();
        eiganMatLKernel<T><<<kpVec.gridDim, kpVec.blockDim, 0, stream>>>(eVecs[i].toKernel2d());

        KernelPrep kpVal(eVecs[i]._cols);
        eiganValLKernel<T><<<kpVal.gridDim, kpVal.blockDim, 0, stream>>>(eVals.col(i).toKernel1d());

    }

    void setUTilde(Tensor<T> f, Tensor<T> u, Handle stream) {

        KernelPrep kp = f.kernelPrep();
        setUTildeKernel<<<kp.gridDim, kp.blockDim, 0, stream>>>(u.toKernel3d(), eVals.toKernel2d(), f.toKernel3d());

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

    void multiplyEF(Handle& hand, Tensor<T>& f, bool transposeE) {

        auto c1Front = f.layerRowCol(0);

        Mat<T>::batchMult(Singleton<T>::ONE,
            eVecs[0], 0,
            f.layerRowCol(0), f.layerSize(),
            Singleton<T>::ZERO, c1Front, f.layerSize(),
            transposeE, true, hand,
            f._layers
            );
        Mat<T>::batchMult(Singleton<T>::ONE,
            eVecs[1], 0,
            f.layerRowCol(0), f.layerSize(),
            Singleton<T>::ZERO, c1Front, f.layerSize(),
            transposeE, false, hand, f._layers
            );
        auto c1Side = f.layerColDepth(0);
        Mat<T>::batchMult(Singleton<T>::ONE,
            eVecs[2], 0,
            f.layerColDepth(0), f._rows,
            Singleton<T>::ZERO, c1Side, f._rows,
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
        Poisson<T>(boundary, f, hand),
        eVecs({SquareMat<T>::create(this->dim.cols), SquareMat<T>::create(this->dim.rows), SquareMat<T>::create(this->dim.layers)}),
        eVals(Mat<T>::create(std::max(this->dim.rows, std::max(this->dim.cols, this->dim.layers)),3))
    {

        f.get(std::cout << "f:\n", true, false, &hand);

        Mat<T> temp = SquareMat<T>::create(std::max(this->dim.rows,this->dim.cols));

        for (size_t i = 0; i < 3; ++i) eigenL(i, hand);

        auto fTensor = f.tensor(this->dim.rows, this->dim.cols);
        multiplyEF(hand, fTensor, true);

        auto xTensor = x.tensor(this->dim.rows, this->dim.cols);

        setUTilde(fTensor, xTensor, hand);

        multiplyEF(hand, xTensor, false);


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

    const auto boundary = CubeBoundary<double>::ZeroTo1(dim, hand);

    auto x = Vec<double>::create(boundary.internalSize(), hand);

    auto f = Vec<double>::create(boundary.internalSize(), hand);

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
