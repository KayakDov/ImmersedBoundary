
 #ifndef BICGSTAB_POISSONFDM_CUH
#define BICGSTAB_POISSONFDM_CUH
#include "deviceArrays/headers/bandedMat.h"
#include "BiCGSTAB.cu"
#include "deviceArrays/headers/DeviceMemory.h"


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
__global__ void setRHSKernel3D(T* __restrict__ b, const size_t bStride,
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

    b[bStride * (row + (col + layer * gWidth) * gHeight)] -=
          (row == 0            ? topBottom[col*tbLd + (gDepth - 1 - layer)]                : 0)
        + (row == gHeight - 1  ? topBottom[col*tbLd + (gDepth - 1 - layer) + gDepth]  : 0)
        + (col == 0            ? leftRight[(gDepth - 1 - layer)*lrLd + row]                : 0)
        + (col == gWidth - 1   ? leftRight[(gDepth -1 - layer)*lrLd + row + gHeight]   : 0)
        + (layer == 0          ? frontBack[col*fbLd + row]                                 : 0)
        + (layer == gDepth - 1 ? frontBack[col*fbLd + row + gHeight]                 : 0);
}

/**
 * @brief Device-side functor to set off-diagonal entries of the system matrix A to 0 or NAN.
 *
 * This is used inside the setAKernel3d kernel to handle the six neighbors for each interior
 * grid point. It ensures that entries corresponding to boundary connections are either
 * correctly set to 0 (non-existent internal connection) or marked as NAN (which typically
 * signals an element outside the valid band storage).
 *
 * @tparam T Floating-point type (float or double).
 */
template <typename T>
class Set0 {
private:
    T* __restrict__ A;
    const size_t ldA, idGrid, heightA;
public:
    /**
     * @brief Constructs the Set0 functor.
     *
     * @param[in] mapDiagIndextoARow Device pointer mapping diagonal index (offset) to the row index in the banded matrix A.
     * @param[in,out] A Pointer to the banded matrix data on the device.
     * @param[in] ldA The leading dimension of A.
     * @param[in] idGrid The linear index of the current grid point (row in A).
     * @param[in] widthA The width (number of columns) of A, equal to gridSize.
     */
    __device__ Set0(T* __restrict__ A, const size_t ldA, const size_t idGrid, const size_t widthA) :
        A(A), ldA(ldA), idGrid(idGrid), heightA(widthA) {}

    /**
     * @brief Sets the corresponding off-diagonal entry to 0 or NAN based on boundary condition logic.
     *
     * This operator is called to check if a specific off-diagonal entry (corresponding to a neighbor)
     * should be set to 0 (internal point) or NAN (outside band storage).
     *
     * @param[in] diagIndex The diagonal index (offset) corresponding to the neighbor being checked.
     */
    __device__ void operator()(const int32_t diagIndex, const size_t colInd) {

        const size_t rowInd = ((static_cast<int32_t>(idGrid) + min(diagIndex , 0)) % static_cast<int32_t>(heightA) + static_cast<int32_t>(heightA)) % heightA;
        const size_t indexA = colInd * ldA + rowInd;

        if (rowInd < heightA - abs(diagIndex)) A[indexA] = static_cast<T>(0);
        else A[indexA] = NAN;
    }
};

struct AdjacencyIndexing {
    const size_t row;
    const int32_t diag;
    AdjacencyIndexing(const size_t row, const int32_t diag): row(row), diag(diag) {}
};

/**
 * @brief CUDA kernel to set up the system matrix A for the 3D Poisson FDM problem.
 *
 * Each thread handles one unknown point $(gRow, gCol, gLayer)$ in the interior grid,
 * setting the main diagonal entry ($A_{i,i} = -6$) and using the Set0 functor to handle
 * the 6 off-diagonal entries (neighbors) and enforce boundary conditions by setting
 * unused band elements to NAN.
 *
 * @tparam T Floating-point type (float or double).
 * @param[out] A Pointer to the banded (or dense) matrix storage on the device.
 * @param[in] ldA The leading dimension of the matrix A.
 * @param[in] heightA The width (number of columns) of A, equal to gridSize.
 * @param[in] gHeight Interior grid height.
 * @param[in] gWidth Interior grid width.
 * @param[in] gDepth Interior grid depth.
 * @param[in] mapDiagIndextoARow A device array mapping the diagonal index offset to the row index within the banded storage format of A.
 */
template <typename T>
__global__ void setAKernel(T* __restrict__ A, const size_t ldA, const size_t heightA,
    const size_t gHeight, const size_t gWidth, const size_t gDepth,
    const size_t hereCol,
    const size_t upCol,
    const size_t downCol,
    const size_t leftCol,
    const size_t rightCol,
    const size_t frontCol,
    const size_t backCol
    ) {
    const size_t gCol = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t gRow = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t gLayer = blockIdx.z * blockDim.z + threadIdx.z;

    if (gRow >= gHeight || gCol >= gWidth || gLayer >= gDepth) return;

    const size_t idGrid = (gLayer * gWidth + gCol) * gHeight + gRow;

    A[hereCol * ldA + idGrid] = -6;
    Set0<T> set0(A, ldA, idGrid, heightA);

    if (gRow == 0) set0(-1, upCol);
    else if (gRow == gHeight - 1) set0(1, downCol);
    if (gCol == 0) set0(-static_cast<int32_t>(gHeight), leftCol);
    else if (gCol == gWidth - 1) set0(gHeight, rightCol);
    if (gLayer == 0) set0(-static_cast<int32_t>(gWidth * gHeight), frontCol);
    else if (gLayer == gDepth - 1) set0(gWidth * gHeight, backCol);

}

constexpr  size_t numDiagonals = 7;
/**
 * @brief Solves the 3D Poisson equation $\nabla^2 u = f$ using the Finite Difference Method (FDM)
 * and the BiCGSTAB iterative solver on the resulting linear system $A\mathbf{x} = \mathbf{b}$.
 *
 * This class handles the construction of the FDM linear system for a 3D grid,
 * including setting up the system matrix $A$ (as a banded matrix), calculating
 * the right-hand side vector $\mathbf{b}$ (incorporating boundary conditions),
 * and leveraging the BiCGSTAB solver for the solution. The class assumes a uniform
 * grid spacing (which is absorbed into the matrix $A$ coefficients).
 *
 * @tparam T Floating-point type used for the computation (typically float or double).
 */
template <typename T>
class PoissonFDM {
private:    
    Vec<T> _b;
    const size_t _rows, _cols, _layers;

public:
    const AdjacencyIndexing here, up, down, left, right, back, front;
    /**
     * @brief Calculates the total number of unknowns (interior points) in the 3D grid.
     * @return The total grid size: $\text{rows} \times \text{cols} \times \text{layers}$.
     */
    [[nodiscard]] size_t gridSize() const {
        return _rows * _cols * _layers;
    }
private:
    const BandedMat<T> A;
    
    void setB(const Mat<T>& frontBack, const Mat<T>& leftRight, const Mat<T>& topBottom, cudaStream_t stream) {
        const size_t totalThreadsNeeded = (frontBack.size() + leftRight.size() + topBottom.size());

        constexpr size_t threadsPerBlock = 256;
        const size_t gridDim = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

        setRHSKernel3D<<<gridDim, threadsPerBlock, 0, stream>>>(
            _b.data(), _b._ld,
            frontBack.data(), frontBack._ld, frontBack.size(),
            leftRight.data(), leftRight._ld, leftRight.size(),
            topBottom.data(), topBottom._ld, topBottom.size(),
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
     * @param numInds The number of indices.
     * @param mindices device pointer to the int32_t offsets array (length numNonZeroDiags).
     * @param handle contains the stream to run on.
     * @param preAlocatedForA Provide prealocated memory here to be written to, numDiagonals x _b.size().
     */
    BandedMat<T> setA(cudaStream_t& stream, Mat<T>& preAlocatedForA, Vec<int32_t>& prealocatedForIndices) {

        preAlocatedForA.subMat(0, 1, preAlocatedForA._rows, preAlocatedForA._cols - 1).fill(1, stream);

        dim3 block(8, 8, 8);
        dim3 grid = makeGridDim( _cols, _rows, _layers, block);

        setAKernel<T><<<grid, block, 0, stream>>>(
            preAlocatedForA.data(), preAlocatedForA._ld, preAlocatedForA._rows,
            _rows, _cols, _layers,
            here.row, up.row, down.row, left.row, right.row, front.row, back.row
        );

        CHECK_CUDA_ERROR(cudaGetLastError());

        loadMapRowToDiag(prealocatedForIndices, stream);

        return BandedMat<T> (preAlocatedForA, prealocatedForIndices);
    }

    void loadMapRowToDiag(Vec<int32_t> diags, const cudaStream_t stream) const {
        int32_t diagsCpu[numDiagonals];
        diagsCpu[up.row] = up.diag;
        diagsCpu[down.row] = down.diag;
        diagsCpu[left.row] = left.diag;
        diagsCpu[right.row] = right.diag;
        diagsCpu[back.row] = back.diag;
        diagsCpu[front.row] = front.diag;
        diagsCpu[here.row] = here.diag;
        diags.set(diagsCpu, stream);
    }
public:
    /**
    * @brief Constructs the PoissonFDM solver object.
    *
    * Initializes the boundary condition matrices and the dimensions of the interior grid.
    * It assumes the RHS vector $\mathbf{b}$ is pre-loaded with the source term $f$.
    *
    * @param[in] frontBack A Mat containing boundary values for the front (layer=0) and back (layer=depth-1) faces.
    * @param[in] leftRight A Mat containing boundary values for the left (col=0) and right (col=width-1) faces.
    * @param[in] topBottom A Mat containing boundary values for the top (row=0) and bottom (row=height-1) faces.
    * @param[in] b The initial right-hand side vector, pre-loaded with the source term $f$.
    * This vector is modified by the solver to include boundary contributions.
    */
    PoissonFDM(const Mat<T>& frontBack, const Mat<T>& leftRight, const Mat<T>& topBottom, const Vec<T> b, Mat<T>& preAlocatedForBandedA, Vec<int32_t>& prealocatedForIndices, cudaStream_t stream):
        _b(b),
        _rows(frontBack._rows/2),
        _cols(frontBack._cols),
        _layers(topBottom._rows/2),
        here(0, 0),
        up(1, 1),
        down(2, -1),
        right(3, _rows),
        left(4, -_rows),
        back(5, _rows*_cols),
        front(6, -_rows*_cols),
        A(setA(stream, preAlocatedForBandedA, prealocatedForIndices))
    {
        // A.Mat<T>::get(std::cout, true, false, stream);
        setB(frontBack, leftRight, topBottom, stream);
    }
    /**
     * @brief Solves the Poisson equation for the grid.
     *
     * Automatically dispatches to the 2D or 3D solver based on whether the number of layers is 1.
     *
     * @param[out] x Pre-allocated memory that the solution will be written to.
     * @param A Pre-allocated memory that will be used to compute the solution.  It should be numDiagonals rows and _b.size() columns.
     * @param[in] hand The CUDA handle (stream/context) to manage the computation.
     */
    void solve(Vec<T>& x, Mat<T> prealocatedForBiCGSTAB) {

        BiCGSTAB<T> solver(_b, &prealocatedForBiCGSTAB);

        solver.solveUnpreconditionedBiCGSTAB(A, x);
    }
};

/**
 * @brief Calls releaseResources() on an arbitrary number of Mat<T> views.
 *
 * This function is useful for explicitly nullifying the pointers in non-owning
 * Mat<T> objects (views) to prevent their destructors from attempting a
 * double-free, thereby freeing up the memory for the next operation.
 *
 * @tparam T The element type of the matrices.
 * @tparam MatTs A variadic pack of Mat<T> types.
 * @param matrices A reference pack of Mat<T> objects (views).
 */
template<typename T>
void freeMat(const std::initializer_list<std::reference_wrapper<Mat<T>>> destructMe) {
    for (auto& ref : destructMe)
        ref.get().freeMem();
}

 /**
  * Creates and solved an example Poisson class on a cube with the given side length.
  * @param dimLength The length of an edge of the grid.  //up to 325 works on Dov's computer.  After that the size of
  * the initally allocated memory exceeds the available memory on the gpu.
  */
 void testPoisson(const size_t dimLength, cudaStream_t& stream) {


    const size_t height = dimLength, width = dimLength, depth = dimLength, size = height * width * depth;
    constexpr double frontFaceVal = 1;

    auto boundaries = Mat<double>::create(
        height * 4 + depth * 2,
        std::max(depth, width)
        );

    auto frontBack = boundaries.subMat(0, 0, 2*height, width),
        leftRight = boundaries.subMat(frontBack._rows, 0, 2 * height, depth),
        topBottom = boundaries.subMat(frontBack._rows + leftRight._rows, 0, 2 * depth, width);

    auto front = frontBack.subMat(0, 0, height, width),
        back = frontBack.subMat(height, 0, height, width),
        top = topBottom.subMat(0, 0, depth, width),
        bottom = topBottom.subMat(depth, 0, depth, width);

    front.fill(frontFaceVal, stream);
    back.fill(0, stream);

    for (size_t layerInd = 0; layerInd < depth; ++layerInd) {
        double val = frontFaceVal * (static_cast<double>(layerInd) + 1.0)/(depth + 1.0);
        leftRight.col(layerInd).fill(val, stream);
        top.row(layerInd).fill(val, stream);
        bottom.row(layerInd).fill(val, stream);
    }

    auto longVecs = Mat<double>::create(size, 2+ numDiagonals + 7);
    auto b = longVecs.col(0);
    b.fill(0, stream);
    auto x = longVecs.col(1);
    auto A = longVecs.subMat(0, 2, size, numDiagonals);
    auto prealocatedForBiCGSTAB = longVecs.subMat(0, 2 + numDiagonals, size, 7);

    auto diagonalInds = Vec<int32_t>::create(numDiagonals);

    PoissonFDM<double> solver(frontBack, leftRight, topBottom, b, A, diagonalInds, stream);

    freeMat<double>({boundaries, frontBack, leftRight, topBottom, front, back, top, bottom});

    solver.solve(x, prealocatedForBiCGSTAB);

    // x.get(std::cout, true, false, stream);
}

/**
 * @brief Main execution function for the Poisson FDM solver example.
 *
 * Sets up a simple 3D grid problem with constant boundary conditions,
 * initializes the RHS and solution vectors, and calls the solver.
 *
 * @param[in] argc Argument count (unused).
 * @param[in] argv Argument vector (unused).
 * @return 0 on successful execution.
 *///TODO:Similarly for any memory allocated by bicgstab.
int main(int argc, char *argv[]) {

    Handle hand;

    std::cout << "dimension size, number of iterations, total time" << std::endl;
    for (size_t i = 2; i < 350; ++i) {

    // size_t i = 2;
        std::cout << i << ", ";
        testPoisson(i, hand.stream);
        cudaDeviceSynchronize();

    }

    return 0;
}

#endif //BICGSTAB_POISSONFDM_CUH
