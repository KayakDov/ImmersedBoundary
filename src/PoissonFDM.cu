
 #ifndef BICGSTAB_POISSONFDM_CUH
#define BICGSTAB_POISSONFDM_CUH
#include "deviceArrays/deviceArrays.h"
#include "algorithms.cu"


/**
 * @brief Map a linear index to coordinates on the front or back faces of the 3D grid.
 *
 * This function determines the (layer, row, col) coordinates corresponding
 * to the front (layer = 0) or back (layer = depth - 1) boundary faces.
 *
 * @param[out] layer The computed depth index (0 or depth - 1).
 * @param[out] row The computed row index in the grid.
 * @param[out] col The computed column index in the grid.
 * @param[in] height The interior grid height (without boundary).
 * @param[in] width The interior grid width (without boundary).
 * @param[in] depth The interior grid depth (without boundary).
 * @param[in] idx The linear thread index, offset for front/back faces.
 *
 */
__device__ void setIndicesFrontBackFaces(size_t& layer, size_t& row, size_t& col, const size_t height, const size_t width, const size_t depth, size_t idx) {
    layer = idx < height * width ? 0 : depth - 1;
    row = idx % height;
    col = (idx / height) % width;
}

/**
 * @brief Map a linear index to coordinates on the left or right faces of the 3D grid.
 *
 * This function determines the (layer, row, col) coordinates corresponding
 * to the left (col = 0) or right (col = width - 1) boundary faces.
 *
 * The function excludes overlap with front/back faces by checking if
 * the layer is at the boundary (0 or depth - 1). In those cases, it returns false.
 *
 * @param[out] layer The computed depth index.
 * @param[out] row The computed row index.
 * @param[out] col The computed column index (0 or width - 1).
 * @param[in] height The interior grid height (without boundary).
 * @param[in] width The interior grid width (without boundary).
 * @param[in] depth The interior grid depth (without boundary).
 * @param[in] idx The linear thread index, offset for left/right faces.
 * @return true if the index corresponds to a valid left/right face location, false otherwise.
 */
__device__ bool setIndicesLeftRightFaces(size_t& layer, size_t& row, size_t& col, const size_t height, const size_t width, const size_t depth, size_t idx) {
    idx -= 2 * height * width;
    layer = (depth - 1 - (idx / height) % depth) ;
    if (layer == depth - 1 || layer == 0) return false;
    row = idx % height;
    col = idx < height * depth ? 0 : width - 1;
    return true;
}
/**
 * @brief Map a linear index to coordinates on the top or bottom faces of the 3D grid.
 *
 * This function determines the (layer, row, col) coordinates corresponding
 * to the top (row = 0) or bottom (row = height - 1) boundary faces.
 *
 * The function excludes overlap with front/back and left/right faces by checking
 * if the layer is at the boundary (0 or depth - 1) or if the column is at the
 * left/right boundary (0 or width - 1). In those cases, it returns false.
 *
 * @param[out] layer The computed depth index.
 * @param[out] row The computed row index (0 or height - 1).
 * @param[out] col The computed column index.
 * @param[in] height The interior grid height (without boundary).
 * @param[in] width The interior grid width (without boundary).
 * @param[in] depth The interior grid depth (without boundary).
 * @param[in] idx The linear thread index, offset for top/bottom faces.
 * @return true if the index corresponds to a valid top/bottom face location, false otherwise.
 */
__device__ bool setIndicesTopBottomFaces(size_t& layer, size_t& row, size_t& col, const size_t height, const size_t width, const size_t depth, size_t idx) {
    idx -= 2 * height * width + 2 * height * depth;
    layer = depth - 1 - (idx % depth);
    col = (idx / depth) % width;
    if (layer == depth - 1 || layer == 0 || col == width - 1 || col == 0) return false;
    row = idx < depth * width ? 0 : height - 1;
    return true;
}
/**
 * @brief CUDA kernel to apply boundary conditions to the right-hand side of a 3D Poisson problem.
 *
 * This kernel updates the RHS vector `b` with contributions from the six boundary faces
 * (front, back, left, right, top, bottom) of the 3D domain. Each thread corresponds to one
 * boundary element and adds its boundary contribution to the correct position in `b`.
 *
 * @tparam T Floating-point type (float or double).
 * @param[out] b The right-hand side vector of the linear system. Stored in column-major order.
 *               Initially holds the RHS for the interior, and is incremented by boundary contributions.
 * @param[in] frontBack Concatenated arrays holding the boundary values for the front (layer = 0)
 *                      and back (layer = depth - 1) faces.
 * @param[in] fbLd Leading dimension of the front/back matrices.
 * @param[in] fbBlockSize Number of elements in combined faces, frontFace + backFace.
 * @param[in] leftRight Concatenated arrays holding the boundary values for the left (col = 0)
 *                      and right (col = width - 1) faces.
 * @param[in] lrLd Leading dimension of the left/right matrices.
 * @param[in] lrBlockSize Number of elements in one face (left or right).
 * @param[in] topBottom Concatenated arrays holding the boundary values for the top (row = 0)
 *                      and bottom (row = height - 1) faces.
 * @param[in] tbLd Leading dimension of the top/bottom matrices.
 * @param[in] tbBlockSize Number of elements in one face (top or bottom).
 * @param[in] gHeight Interior grid height (number of unknowns along Y).
 * @param[in] gWidth Interior grid width (number of unknowns along X).
 * @param[in] gDepth Interior grid depth (number of unknowns along Z).
 */
template <typename T>
__global__ void setRHSKernel3D(T* __restrict__ b,
                               const T* __restrict__ frontBack, const size_t fbLd, const size_t fbBlockSize,
                               const T* __restrict__ leftRight, const size_t lrLd, const size_t lrBlockSize,
                               const T* __restrict__ topBottom, const size_t tbLd, const size_t tbBlockSize,
                               const size_t gHeight, const size_t gWidth, const size_t gDepth) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fbBlockSize + lrBlockSize + tbBlockSize) return;

    size_t layer, row, col;

    if (idx < fbBlockSize) setIndicesFrontBackFaces(layer, row, col, gHeight, gWidth, gDepth, idx);
    else if (idx < fbBlockSize + lrBlockSize){
        if (!setIndicesLeftRightFaces(layer, row, col, gHeight, gWidth, gDepth, idx)) return;
    }
    else if (!setIndicesTopBottomFaces(layer, row, col, gHeight, gWidth, gDepth, idx)) return;

    b[row + (col + layer * gWidth) * gHeight] -=
          (row == 0            ? topBottom[col*tbLd + layer]                               : 0)
        + (row == gHeight - 1  ? topBottom[col*tbLd + layer + gWidth*tbLd]               : 0)
        + (col == 0            ? leftRight[(gDepth - 1 - layer)*lrLd + row]                : 0)
        + (col == gWidth - 1   ? leftRight[(gDepth -1 - layer)*lrLd + row + gDepth*lrLd] : 0)
        + (layer == 0          ? frontBack[col*fbLd + row]                                 : 0)
        + (layer == gDepth - 1 ? frontBack[col*fbLd + row + gWidth * tbLd]                 : 0);
}

 /**
  * We assume A comes in with the primary diagonal set to 4 or 6 and all other diagonals set to 1.
  * @tparam T double or float
  * @param A Matrix A is stored in a dense (banded) format, where each row corresponds to one diagonal (offset given by indices).
  * @param ldA
  * @param primaryDiagonalRow The row that the primary diagonal index is on.
  * @param gHeight The height of the grid.
  * @param gWidth The width of the grid.
  * @param gDepth The depth of the grid.
  * @param indices The index of each corresponding row.
  * @param numNonZeroDiags The number of rows in A.
  * @param widthA The width of matrix A
  * TODO: exploit that A sparse may be simetrical across multiple axises,a_{i, j} = a_{j, i} and a_{hieght - i, width - j} = a_{j, i}
  * TODO: Should the value of the diagonal be closer to 0 if some of the neighbors are off grid (as opposed to artaficially set to 0 which is already covered)
  */
 template <typename T>
__global__ void setAKernel(T* __restrict__ A, const size_t ldA, const size_t primaryDiagonalRow,
    const size_t gHeight, const size_t gWidth, const size_t gDepth,
    const int32_t* indices, const size_t numNonZeroDiags, const size_t widthA
    ) {
    const size_t gCol = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t gRow = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t gLayer = blockIdx.z * blockDim.z + threadIdx.z;

    if (gRow >= gHeight || gCol >= gWidth || gLayer >= gDepth) return;

    const size_t sparseARow = (gLayer * gWidth + gCol) * gHeight + gRow;
    const size_t sparseARowXLdA =  sparseARow * ldA;
    const size_t primaryDiagonalInd = sparseARowXLdA + primaryDiagonalRow;
    A[primaryDiagonalInd] = static_cast<int32_t>(1) - static_cast<int32_t>(numNonZeroDiags);

    for (size_t rowA = 0; rowA < numNonZeroDiags; ++rowA) {
        const int32_t d = indices[rowA];

        size_t writeInd = sparseARowXLdA + rowA + (d < 0 ? static_cast<int32_t>(ldA) * d : 0);

        if (primaryDiagonalRow != rowA && sparseARow + abs(d) < widthA && (d >= 0 || sparseARow >= -d)){

            const bool bottom = d == 1                                       && gRow == gHeight - 1,
                       top    = d == -1                                      && gRow == 0,
                       left   = d == -static_cast<int32_t>(gHeight)          && gCol == 0,
                       right  = d == static_cast<int32_t>(gHeight)           && gCol == gWidth - 1,
                       front  = d == -static_cast<int32_t>(gWidth * gHeight) && gLayer == 0,
                       back   = d == static_cast<int32_t>(gWidth * gHeight)  && gLayer == gDepth - 1;

            if (d > 0 && (bottom || right || back) || d < 0 && (top || left || front)){

                A[writeInd] = static_cast<T>(0);
                A[primaryDiagonalInd] += static_cast<T>(1);
            } else A[writeInd] = static_cast<T>(1);
        } else if (primaryDiagonalRow != rowA) A[writeInd] = static_cast<T>(1); //else if (rowA != 0) printf("Rejected from 1st loop: gRow=%llu \tgCol=%llu \tgLayer=%llu \trowA=%llu \td=%d writeInd=%llu \tA height = %llu \tA width = %llu\n", (unsigned long long)gRow,(unsigned long long)gCol,(unsigned long long)gLayer,(unsigned long long)rowA,d, (unsigned long long)(numNonZeroDiags), (unsigned long long)widthA);
    }
}

template <typename T>
class PoissonFDM {
private:
    const Mat<T> _frontBack, _leftRight, _topBottom;
    Vec<T> _b;
    const size_t _rows, _cols, _layers;

    void setB3d(cudaStream_t stream) {
        const size_t totalThreadsNeeded = (_frontBack.size() + _leftRight.size() + _topBottom.size());

        constexpr size_t threadsPerBlock = 256;
        const size_t gridDim = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

        setRHSKernel3D<<<gridDim, threadsPerBlock, 0, stream>>>(
            _b.data(),
            _frontBack.data(), _frontBack.getLD(), _frontBack.size(),
            _leftRight.data(), _leftRight.getLD(), _leftRight.size(),
            _topBottom.data(), _topBottom.getLD(), _topBottom.size(),
            _rows, _cols, _layers);

        CHECK_CUDA_ERROR(cudaGetLastError());

        std::cout << "PoissonFDM setB3d \n" << _b << std::endl;
    }

    static inline dim3 makeGridDim(size_t x, size_t y, size_t z, dim3 block) {
        return dim3( (unsigned)((x + block.x - 1) / block.x),
                     (unsigned)((y + block.y - 1) / block.y),
                     (unsigned)((z + block.z - 1) / block.z) );
    }

    /**
     * @brief Launch kernel that assembles A in banded/dense storage.
     *
     * @param indices device pointer to the int32_t offsets array (length numNonZeroDiags).
     * @param handle contains the stream to run on.
     */
    Mat<T> setA3d(int32_t indices[], size_t numInds, Handle& handle) {

        Mat<T> A = Mat<T>::create(numInds, _b.size());

        dim3 block(8, 8, 8);
        dim3 grid = makeGridDim( _cols, _rows, _layers, block);

        Vec<int32_t> inds = Vec<int32_t>::create(numInds, handle.stream);
        inds.set(indices, handle.stream);

        setAKernel<T><<<grid, block, 0, handle.stream>>>(
            A.data(), A._ld, size_t(0),
            _rows, _cols, _layers,
            inds.data(), numInds, A._cols
        );
        CHECK_CUDA_ERROR(cudaGetLastError());

        return A;
    }


    void solve2d(Vec<T>& x, Handle hand) {

    }
    void solve3d(Vec<T>& x, Handle& handle) {

        const size_t numNonZeroDiags = 7;
        Vec<int32_t> indices = Vec<int32_t>::create(numNonZeroDiags, handle.stream);
        int32_t indicesCpu[numNonZeroDiags] = {static_cast<int32_t>(0), static_cast<int32_t>(1), static_cast<int32_t>(-1),
            static_cast<int32_t>(_rows), static_cast<int32_t>(-_rows), static_cast<int32_t>(_rows * _cols),
            static_cast<int32_t>(-_rows * _cols)};


        Mat<T> A = setA3d(indicesCpu, numNonZeroDiags, handle);
        setB3d(handle.stream);

        std::cout << "Poisison::solve3d A = " << std::endl << A << std::endl;

        BiCGSTAB<T> solver(_b);
        solver.solveUnpreconditionedBiCGSTAB(A, indices, &x);

    }

public:
    PoissonFDM(const Mat<T>& frontBack, const Mat<T>& leftRight, const Mat<T>& topBottom, const Vec<T> b):
        _frontBack(frontBack),
        _leftRight(leftRight),
        _topBottom(topBottom),
        _b(b),
        _rows(frontBack._rows),
        _cols(frontBack._cols/2),
        _layers(topBottom._rows) {

    }

    void solve(Vec<T>& x, Handle& handle) {
        if (_layers == 1) solve2d(x, handle);
        else solve3d(x, handle);

    }

};

 int main(int argc, char *argv[]) {
     Handle hand;
     Mat<float> frontBack = Mat<float>::create(2, 4), leftRight = Mat<float>::create(2, 4), topBottom = Mat<float>::create(2, 4);
     Vec<float> b = Vec<float>::create(8, hand.stream);
     // float bCpu[] = {0, 0, 0, 0, 0, 0, 0, 0},
     //    frontBackCpu[] = {-1, 0, 0, 1, 2, 3, 3, 4},
     //    leftRightCpu[] = {1, 0, 0, -1, 4, 3, 3, 2},
     //    topBottomCpu[] = {2, 1, 1, 0, -1, -2, -2, -3};

     float bCpu[] = {0, 0, 0, 0, 0, 0, 0, 0},
             frontBackCpu[] = {-1, -1, -1, -1,
                                                1, 1, 1, 1},
             leftRightCpu[] = {-2, -2, -2, -2,
                                                2, 2, 2, 2},
             topBottomCpu[] = {-3, -3, -3, -3,
                                                3, 3, 3, 3};


     b.set(bCpu, hand.stream);
     frontBack.set(frontBackCpu, hand.stream);
     leftRight.set(leftRightCpu, hand.stream);
     topBottom.set(topBottomCpu, hand.stream);

     PoissonFDM<float> solver(frontBack, leftRight, topBottom, b);
     Vec<float> x = Vec<float>::create(8, hand.stream);
     solver.solve(x, hand);
     std::cout << x << std::endl;

     return 0;
 }


#endif //BICGSTAB_POISSONFDM_CUH
