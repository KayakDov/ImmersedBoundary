
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
          (row == 0            ? topBottom[col*tbLd + (gDepth - 1 - layer)]                : 0)
        + (row == gHeight - 1  ? topBottom[col*tbLd + (gDepth - 1 - layer) + gWidth*tbLd]  : 0)
        + (col == 0            ? leftRight[(gDepth - 1 - layer)*lrLd + row]                : 0)
        + (col == gWidth - 1   ? leftRight[(gDepth -1 - layer)*lrLd + row + gDepth*lrLd]   : 0)
        + (layer == 0          ? frontBack[col*fbLd + row]                                 : 0)
        + (layer == gDepth - 1 ? frontBack[col*fbLd + row + gWidth * tbLd]                 : 0);
}


template <typename T>
class Set0 {
private:
    const size_t* mapDiagIndextoARow;
    T* __restrict__ A;
    const size_t ldA, idGrid, widthA, colXLda;
public:
    __device__ Set0(const size_t* mapDiagIndextoARow, T* __restrict__ A, const size_t ldA, const size_t idGrid, const size_t widthA) :
        mapDiagIndextoARow(mapDiagIndextoARow), A(A), ldA(ldA), idGrid(idGrid), widthA(widthA), colXLda(idGrid * ldA) {}

    __device__ void operator()(const int32_t rowDiagonalIndex) {
        // ++A[colXLda + mapDiagIndextoARow[0]];

        const size_t col = ((static_cast<int32_t>(idGrid) + min(rowDiagonalIndex , 0)) % static_cast<int32_t>(widthA) + static_cast<int32_t>(widthA)) % widthA;
        const size_t indexA = col * ldA + mapDiagIndextoARow[rowDiagonalIndex];

        if (col < widthA - abs(rowDiagonalIndex)) A[indexA] = static_cast<T>(0);
        else A[indexA] = NAN;
    }
};



template <typename T>
__global__ void setAKernel3d(T* __restrict__ A, const size_t ldA, const size_t widthA,
    const size_t gHeight, const size_t gWidth, const size_t gDepth,
    const size_t* mapDiagIndextoARow
    ) {
    const size_t gCol = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t gRow = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t gLayer = blockIdx.z * blockDim.z + threadIdx.z;

    if (gRow >= gHeight || gCol >= gWidth || gLayer >= gDepth) return;

    const size_t idGrid = (gLayer * gWidth + gCol) * gHeight + gRow;

    A[idGrid*ldA + mapDiagIndextoARow[0]] = -6;
    Set0<T> set0(mapDiagIndextoARow, A, ldA, idGrid, widthA);

    if (gRow == 0) set0(-1);
    else if (gRow == gHeight - 1) set0(1);
    if (gCol == 0) set0(-static_cast<int32_t>(gHeight));
    else if (gCol == gWidth - 1) set0(gHeight);
    if (gLayer == 0) set0(-static_cast<int32_t>(gWidth * gHeight));
    else if (gLayer == gDepth - 1) set0(gWidth * gHeight);

}

template <typename T>
class PoissonFDM {
private:
    const Mat<T> _frontBack, _leftRight, _topBottom;
    Vec<T> _b;
    const size_t _rows, _cols, _layers;
public:
    [[nodiscard]] size_t gridSize() const {
        return _rows * _cols * _layers;
    }
private:
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
    Mat<T> setA3d(Vec<size_t> mapDiagIndexToARow, size_t numInds, Handle& handle) {

        Mat<T> A = Mat<T>::create(numInds, _b.size());
        A.subMat(1, 0, A._rows - 1, A._cols).fill(1, handle.stream);

        dim3 block(8, 8, 8);
        dim3 grid = makeGridDim( _cols, _rows, _layers, block);

        setAKernel3d<T><<<grid, block, 0, handle.stream>>>(
            A.data(), A._ld, A._cols,
            _rows, _cols, _layers,
            mapDiagIndexToARow.data() + gridSize()
        );
        CHECK_CUDA_ERROR(cudaGetLastError());

        return A;
    }


    void solve2d(Vec<T>& x, Handle hand) {

    }
    void solve3d(Vec<T>& x, Handle& handle) {

        const size_t numNonZeroDiags = 7;
        Vec<int32_t> mapARowToDiagnalInd = Vec<int32_t>::create(numNonZeroDiags, handle.stream);
        int32_t mapRowToDiagCpu[numNonZeroDiags] = {static_cast<int32_t>(0), static_cast<int32_t>(1), static_cast<int32_t>(-1),
            static_cast<int32_t>(_rows), -static_cast<int32_t>(_rows), static_cast<int32_t>(_rows * _cols),
            -static_cast<int32_t>(_rows * _cols)};
        mapARowToDiagnalInd.set(mapRowToDiagCpu, handle.stream);

        size_t mapDiagToRowCpu[2*gridSize()];
        for (size_t i = 0; i < numNonZeroDiags; ++i) mapDiagToRowCpu[mapRowToDiagCpu[i] + gridSize()] = i;
        Vec<size_t> mapDiagToRow = Vec<size_t>::create(2*gridSize(), handle.stream);\
        mapDiagToRow.set(mapDiagToRowCpu, handle.stream);

        Mat<T> AMat = setA3d(mapDiagToRow, numNonZeroDiags, handle);
        BandedMat<T> A(AMat, mapARowToDiagnalInd);
        setB3d(handle.stream);

        BiCGSTAB<T> solver(_b);

        // std::cout << "PoissonFDM.cu::solve3d" << std::endl;
        // std::cout << "Solving 3D Poisson problem with " << _rows << " rows, " << _cols << " cols, " << _layers << " layers" << std::endl;
        // std::cout << "topBottom:\n" << _topBottom << std::endl;
        // std::cout << "leftRight:\n" << _leftRight << std::endl;
        // std::cout << "frontBack:\n" << _frontBack << std::endl;
        // std::cout << "indices = " << std::endl << mapARowToDiagnalInd << std::endl;
        // std::cout << "A = " << std::endl << A << std::endl;
        // std::cout << "b = " << _b << std::endl;

        solver.solveUnpreconditionedBiCGSTAB(A, &x);

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

    constexpr  size_t dimLength = 80;
    constexpr size_t height = dimLength, width = dimLength, depth = dimLength, size = height * width * depth;
    constexpr double frontFaceVal = 1;

    Mat<double> frontBack = Mat<double>::create(height, 2 * width),
        leftRight = Mat<double>::create(height, 2 * depth),
        topBottom = Mat<double>::create(depth, 2 * width);

    Vec<double> b = Vec<double>::create(size, hand.stream);
    b.fill(0, hand.stream);
    Vec<double> x = Vec<double>::create(size, hand.stream);

    frontBack.subMat(0, 0, height, width).fill(frontFaceVal, hand.stream);
    frontBack.subMat(0, width, height, width).fill(0, hand.stream);
    for (size_t layerInd = 0; layerInd < depth; ++layerInd) {
        double val = (static_cast<double>(layerInd) + 1.0)/(depth + 1.0);
        leftRight.col(layerInd).fill(val, hand.stream);
        leftRight.col(layerInd + depth).fill(val, hand.stream);
        topBottom.row(layerInd).subVec(0, width).fill(val, hand.stream);
        topBottom.row(layerInd).subVec(width, 2 * width).fill(val, hand.stream);
    }

    PoissonFDM<double> solver(frontBack, leftRight, topBottom, b);
    solver.solve(x, hand);
    // std::cout << x << std::endl;

    return 0;
}

#endif //BICGSTAB_POISSONFDM_CUH
