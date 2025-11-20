
#include "deviceArrays/headers/Mat.h"        // 1. BASE CLASS MUST BE INCLUDED FIRST
#include "deviceArrays/headers/SquareMat.h"   // 2. DERIVED CLASS SECOND
#include "deviceArrays/headers/Vec.h"
#include "Poisson.h"                        // 3. This indirect path is now safe

#include <cmath>
#include <iostream>
#include <array>

#include "deviceArrays/headers/Streamable.h"

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

    if (GridInd3d ind; ind < uTilde) {
        // printf("col=%lu row=%lu layer=%lu | eValsX=%f eValsY=%f eValsZ=%f | fTilde=%f\n",
        //    ind.col, ind.row, ind.layer,
        //    eVals(ind.col, 0),
        //    eVals(ind.row, 1),
        //    eVals(ind.layer, 2),
        //    fTilde[ind]);
        uTilde[ind] = fTilde[ind]/(eVals(ind.col, 0) + eVals(ind.row, 1) + eVals(ind.layer, 2));
    }
}

template <typename T>
class FastDiagonalization: public Poisson<T> {
private:

    /**
     * Sets the eigen matrix for L[i].
     * @param i The index of the desired matrix, 1,2, or 3.
     * @param stream
     */
    void eigenL(size_t i, cudaStream_t stream) { //TODO: this method could run on different streams.  Also different streams for each index.

        KernelPrep kpVec = eVecs[i].kernelPrep();
        eiganMatLKernel<T><<<kpVec.gridDim, kpVec.blockDim, 0, stream>>>(eVecs[i].toKernel2d());

        size_t numVaals = eVecs[i]._cols;
        KernelPrep kpVal(numVaals);
        eiganValLKernel<T><<<kpVal.gridDim, kpVal.blockDim, 0, stream>>>(eVals.col(i).subVec(0, numVaals, 1).toKernel1d());
    }

    void setUTilde(const Tensor<T>& f, Tensor<T>& u, Handle& hand) {
        KernelPrep kp = f.kernelPrep();
        setUTildeKernel<<<kp.gridDim, kp.blockDim, 0, hand>>>(u.toKernel3d(), eVals.toKernel2d(), f.toKernel3d());
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

    void multE(size_t i, bool transposeEigen, bool transpose, const Mat<T>& a1, Mat<T>& dst1, size_t stride, Handle& hand, size_t batchCount) {
        Mat<T>::batchMult(
            transpose? a1      :eVecs[i], transpose? stride:0,
            transpose? eVecs[i]: a1,      transpose? 0     : stride,
            dst1, stride,
            transpose? false : transposeEigen, transpose? transposeEigen : false,
            hand, batchCount
            );
    }

    void multEX(const Mat<T>& srcFront1, Mat<T>& dstFront1, Handle& hand, bool transposeE) {
        multE(0, transposeE, true, srcFront1, dstFront1, srcFront1._rows, hand, this->dim.layers);
    }

    void multEY(const Mat<T>& srcFront1, Mat<T>& dstFront1, Handle& hand, bool transposeE) {
        multE(1, transposeE, false, srcFront1, dstFront1, srcFront1._rows, hand, this->dim.layers);
    }

    void multEZ(const Mat<T>& srcSide1, Mat<T>& dstFSide1, Handle& hand, bool transposeE) {
        multE(2, transposeE, true, srcSide1, dstFSide1, this->dim.layers * this->dim.rows, hand, this->dim.cols);
    }

    void multiplyEF(Handle& hand, Tensor<T>& src, Tensor<T>& dst, bool transposeE) {

        auto xFront = src.layerRowCol(0), dstFront1 = dst.layerRowCol(0);
        auto yFront = dst.layerRowCol(0), dstFront2 = src.layerRowCol(0);
        auto zSide = src.layerColDepth(0), dstSide = dst.layerColDepth(0);

        if (transposeE) {
            multEX(xFront, dstFront1, hand, transposeE);
            multEY(yFront, dstFront2, hand, transposeE);
            multEZ(zSide, dstSide, hand, transposeE);
        } else {
            multEZ(zSide, dstSide, hand, transposeE);
            multEY(yFront, dstFront2, hand, transposeE);
            multEX(xFront, dstFront1, hand, transposeE);
        }
    }
public:
    /**
     *
     * @param boundary The boundary conditions.
     * @param x The solution will be placed here.  This should have a space for every element in the grid.
     * @param f The RHS of the laplace equation.  Whatever you have written here, we will overwrite it. Thi should have
     * a space for every element in the grid.
     * @param fTilda temporary storage the same size as f and x.
     * @param stream
     */
    FastDiagonalization(const CubeBoundary<T>& boundary, Vec<T>& x, Vec<T>& f, Vec<T>& fTilda, Handle& hand) :
        Poisson<T>(boundary, f, hand),
        eVecs({SquareMat<T>::create(this->dim.cols), SquareMat<T>::create(this->dim.rows), SquareMat<T>::create(this->dim.layers)}),
        eVals(Mat<T>::create(std::max(this->dim.rows, std::max(this->dim.cols, this->dim.layers)),3))
    {
        std::cout << "b = \n" << GpuOut<T>(f, hand) << std::endl;

        for (size_t i = 0; i < 3; ++i) eigenL(i, hand);

        auto fTensor = f.tensor(this->dim.rows, this->dim.layers),
            fTildaTensor = fTilda.tensor(this->dim.rows, this->dim.layers);

        multiplyEF(hand, fTensor, fTildaTensor, true);

        setUTilde(fTildaTensor, fTensor, hand);

        auto xTensor = x.tensor(this->dim.rows, this->dim.layers);

        multiplyEF(hand, fTensor, xTensor, false);
    }
};


/**
 *
 * @brief Main entry point to demonstrate the FastDiagonalizationMethod for a 2x2x2 grid.
 */
int main() {

    Handle hand;

    const size_t height = 2, width = 2, depth = 4;

    const auto boundary = CubeBoundary<double>::ZeroTo1(height, width, depth, hand);

    std::cout << "boundary frontBack = \n" << GpuOut<double>(boundary.frontBack, hand) << std::endl;
    std::cout << "boundary topBottom = \n" << GpuOut<double>(boundary.topBottom, hand) << std::endl;
    std::cout << "boundary leftRight = \n" << GpuOut<double>(boundary.leftRight, hand) << std::endl;

    auto memAloc = Mat<double>::create(boundary.internalSize(), 3);

    auto x = memAloc.col(0);
    auto f = memAloc.col(1);
    auto temp = memAloc.col(2);

    f.fill(0, hand);

    FastDiagonalization<double> fdm(boundary, x, f, temp, hand);

    std::cout << "x = \n" << GpuOut<double>(x.tensor(height, depth), hand) << std::endl;

    return 0;
}



// void setL(size_t i, cudaStream_t stream) {//TODO: try parallelizing this data
//     L[i].fill(static_cast<T>(0), stream);
//     L[i].diag(0).fill(static_cast<T>(-2), stream);
//     L[i].diag(1).fill(static_cast<T>(1), stream);
//     L[i].diag(-1).fill(static_cast<T>(1), stream);
// }





// void multiplyEF(Handle& hand, Tensor<T>& f, Tensor<T>& dst, bool transposeE) {
//
//     auto xFront = f.layerRowCol(0), dstFront = dst.layerRowCol(0);
//
//     f.get(std::cout << "my compute f_tilde 1 = \n", true, false, hand);
//     eVecs[0].get(std::cout << "eVecs x = \n", true, false, hand);
//     multLayer(0, f._rows, xFront, dstFront, transposeE, true, f._layers, hand);//Note: the result is transposed of what we want.
//
//     dst.get(std::cout << "my compute f_tilde 2 = \n", true, false, hand);
//
//     multLayer(1, f._rows, dstFront, xFront, transposeE, true, f._layers, hand);  //Here the layer is transposed to set it back to the untransposed position.
//
//     auto xSide = f.layerColDepth(0), dstSide = dst.layerColDepth(0);
//     multLayer(2, f._ld, xSide, dstSide, transposeE, true, f._cols, hand);//Here the result again is in the transposed position
// }