
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

        // cudaDeviceSynchronize();
        // Handle hand;
        // eVecs[i].get(std::cout << "eVecs[" << i << "] = \n", true, false, hand);
        // eVals.col(i).get(std::cout << "eVals[" << i << "] = \n", true, false, hand);


    }

    void setUTilde(Tensor<T> f, Tensor<T> u, Handle& hand) {

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

    /**
     * Multiplies an eigenmatrix by a layer (possibly perpendicular) from the grid.
     * @param index 0 for the x dimension , 1 for the y dimesnion, and 2 for the z dimension.
     * @param stride The stride between layers.
     * @param layer1 The first layer to be multiplied.
     * @param dst1 Be sure that dst1 has the same ld and stride as layer1
     * @param transposeE Should the eigenmatrix be transposed.
     * @param transposeLayer Should the layer be transposed.
     * @param batchCount How many layers are there.
     * @param hand The handle.
     */
    void multLayer(size_t index, const size_t stride, Mat<T>& layer1, Mat<T>& dst1, const bool transposeE, const bool transposeLayer, const size_t batchCount, Handle& hand) {
        Mat<T>::batchMult(
            Singleton<T>::ONE,
            eVecs[index], 0,
            layer1, stride,
            Singleton<T>::ZERO, dst1, stride,
            transposeE, transposeLayer, hand,
            batchCount
            );

    }

    void multiplyEF(Handle& hand, Tensor<T>& f, Tensor<T>& dst, bool transposeE) {

        auto xFront = f.layerRowCol(0), dstFront = dst.layerRowCol(0);

        multLayer(0, f._rows, xFront, dstFront, transposeE, true, f._layers, hand);
        multLayer(1, f._rows, dstFront, xFront, transposeE, false, f._layers, hand);

        auto xSide = f.layerColDepth(0), dstSide = dst.layerColDepth(0);
        multLayer(2, f._ld, xSide, dstSide, transposeE, true, f._cols, hand);
    }
public:
    /**
     *
     * @param boundary The boundary conditions.
     * @param x The solution will be placed here.  This should have a space for every element in the grid.
     * @param f The RHS of the laplace equation.  Whatever you have written here, we will overwrite it. Thi should have
     * a space for every element in the grid.
     * @param temp temporary storage the same size as f and x.
     * @param stream
     */
    FastDiagonalization(const CubeBoundary<T>& boundary, Vec<T>& x, Vec<T>& f, Vec<T>& temp, Handle& hand) ://TODO: provide pre alocated memory
        Poisson<T>(boundary, f, hand),
        eVecs({SquareMat<T>::create(this->dim.cols), SquareMat<T>::create(this->dim.rows), SquareMat<T>::create(this->dim.layers)}),
        eVals(Mat<T>::create(std::max(this->dim.rows, std::max(this->dim.cols, this->dim.layers)),3))
    {//TODO:verify E_x kroniker E_y kroniker E_z kroniker f = the f_tilet I built

        for (size_t i = 0; i < 3; ++i) eigenL(i, hand);

        auto fTensor = f.tensor(this->dim.rows, this->dim.cols), tempTensor = temp.tensor(this->dim.rows, this->dim.cols);

        f.get(std::cout << "my compute f_tilde = \n", true, false, hand);
        multiplyEF(hand, fTensor, tempTensor, true);
        multiplyEF(hand, tempTensor, fTensor, false);//TODO;delete me
        f.get(std::cout << "my compute f_tilde = \n", true, false, hand);




        //
        // //----------------------TODO: It seems like the results of the following on an f that hasn't seen multiplyEF should give the same results as multiply EF.  I need to figure out why they are different.
        // //                      TODO: maybe it's because my batch multiply reads and writes to the same space?
        // // auto tempMat1 = Mat<T>::create(eVecs[0]._rows * eVecs[1]._rows, eVecs[0]._cols* eVecs[1]._cols);
        // // auto tempMat2 = Mat<T>::create(tempMat1._rows * eVecs[2]._rows, tempMat1._cols* eVecs[2]._cols);
        // // eVecs[0].multKronecker(eVecs[1], tempMat1, hand);
        // // tempMat1.multKronecker(eVecs[2], tempMat2, hand);
        // // tempMat2.get(std::cout << "tempMat2 = \n", true, false, hand);
        // // tempMat2.mult(f, f, &hand, &Singleton<T>::ONE, &Singleton<T>::ZERO, false);
        // // f.get(std::cout << "my compute f_tilde = \n", true, false, hand);
        // //------------------------------------------------------------------------------------------------------------------
        //
        //
        // auto xTensor = x.tensor(this->dim.rows, this->dim.cols);
        //
        // setUTilde(tempTensor, xTensor, hand);
        //
        // multiplyEF(hand, tempTensor, xTensor, false);
    }
};


/**
 *
 * @brief Main entry point to demonstrate the FastDiagonalizationMethod for a 2x2x2 grid.
 */
int main() {

    constexpr size_t dim = 2;
    Handle hand;

    const auto boundary = CubeBoundary<double>::ZeroTo1(dim, hand);

    auto memAloc = Mat<double>::create(boundary.internalSize(), 3);

    auto x = memAloc.col(0);
    auto f = memAloc.col(1);
    auto temp = memAloc.col(2);

    f.fill(0, hand.stream);

    FastDiagonalization<double> fdm(boundary, x, f, temp, hand);

    x.get(std::cout << "x = \n", true, false, hand);

    return 0;
}



// void setL(size_t i, cudaStream_t stream) {//TODO: try parallelizing this data
//     L[i].fill(static_cast<T>(0), stream);
//     L[i].diag(0).fill(static_cast<T>(-2), stream);
//     L[i].diag(1).fill(static_cast<T>(1), stream);
//     L[i].diag(-1).fill(static_cast<T>(1), stream);
// }
