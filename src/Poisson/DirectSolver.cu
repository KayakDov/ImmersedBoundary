
 #ifndef BICGSTAB_POISSONFDM_CUH
#define BICGSTAB_POISSONFDM_CUH
#include "Poisson.h"
#include "../deviceArrays/headers/BandedMat.h"
#include "../BiCGSTAB/BiCGSTAB.cu"
#include "../deviceArrays/headers/DeviceMemory.h"
#include "deviceArrays/headers/Streamable.h"


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
    DeviceData2d<T>& a;
    const size_t idGrid;
public:
    /**
     * @brief Constructs the Set0 functor.
     *
     * @param[in,out] a Pointer to the banded matrix data on the device.
     * @param[in] idGrid The flat index of the current grid point (row in A).
     */
    __device__ Set0(DeviceData2d<T>& a, const size_t idGrid) :
        a(a), idGrid(idGrid) {}

    /**
     * @brief Sets the corresponding off-diagonal entry to 0 or NAN based on boundary condition logic.
     *
     * This operator is called to check if a specific off-diagonal entry (corresponding to a neighbor)
     * should be set to 0 (internal point) or NAN (outside band storage).
     *
     * @param[in] diagIndex The index of the diagonal corresponding to the neighbor being checked.
     */
    __device__ void operator()(const int32_t diagIndex, const size_t colInd) {

        const size_t rowInd = modPos(static_cast<int32_t>(idGrid) + min(diagIndex , 0), static_cast<int32_t>(a.rows));
        if (rowInd < a.rows - abs(diagIndex)) a(rowInd, colInd) = static_cast<T>(0);
        else a(rowInd, colInd) = NAN;
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
__global__ void setAKernel(DeviceData2d<T> a,
    const GridDim g,
    const size_t hereCol,
    const size_t upCol,
    const size_t downCol,
    const size_t leftCol,
    const size_t rightCol,
    const size_t frontCol,
    const size_t backCol
    ) {
    const GridInd3d ind;

    if (ind >= g) return;

    const size_t idGrid = g[ind];

    a(idGrid, hereCol) = -6;
    Set0<T> set0(a, idGrid);

    if (ind.row == 0) set0(-1, upCol);
    else if (ind.row == g.rows - 1) set0(1, downCol);
    if (ind.col == 0) set0(-static_cast<int32_t>(g.rows*g.layers), leftCol);
    else if (ind.col == g.cols - 1) set0(g.rows*g.layers , rightCol);
    if (ind.layer == 0) set0(-static_cast<int32_t>(g.rows), frontCol);
    else if (ind.layer == g.layers - 1) set0(g.rows, backCol);
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
class DirectSolver : public Poisson<T> {

public:
    const AdjacencyIndexing here, up, down, left, right, back, front;

private:
    const BandedMat<T> A;

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

        KernelPrep kp(this->dim.cols, this->dim.rows, this->dim.layers);
        setAKernel<T><<<kp.gridDim, kp.blockDim, 0, stream>>>(
            preAlocatedForA.toKernel2d(), this->dim,
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
    * @param boundary The boundary conditions.
    * @param[in] b The initial right-hand side vector, pre-loaded with the source term $f$.
    * This vector is modified by the solver to include boundary contributions.
    * @param prealocatedForIndices
    */
    DirectSolver(const CubeBoundary<T>& boundary, Vec<T>& b, Mat<T>& preAlocatedForBandedA, Vec<int32_t>& prealocatedForIndices, cudaStream_t stream):
        Poisson<T>(boundary, b, stream),
        here(0, 0),
        up(1, 1),
        down(2, -1),
        left(3, -this->dim.rows * this->dim.layers),
        right(4, this->dim.rows * this->dim.layers),
        front(5, -this->dim.rows),
        back(6, this->dim.rows),
        A(setA(stream, preAlocatedForBandedA, prealocatedForIndices))
    {
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

        BiCGSTAB<T> solver(this->_b, &prealocatedForBiCGSTAB);

        solver.solveUnpreconditionedBiCGSTAB(A, x);
    }
};


 /**
  * Creates and solved an example Poisson class on a cube with the given side length.
  * @param dimLength The length of an edge of the grid.  //up to 325 works on Dov's computer.  After that the size of
  * the initally allocated memory exceeds the available memory on the gpu.
  */
 void testPoisson(const size_t height, size_t width, size_t depth, Handle& hand) {

    auto boundary = CubeBoundary<double>::ZeroTo1(height, width, depth, hand);

    auto longVecs = Mat<double>::create(boundary.internalSize(), 2 + numDiagonals + 7);
    auto b = longVecs.col(0);
    b.fill(0, hand);
    auto x = longVecs.col(1);
    auto A = longVecs.subMat(0, 2, boundary.internalSize(), numDiagonals);
    auto prealocatedForBiCGSTAB = longVecs.subMat(0, 2 + numDiagonals, boundary.internalSize(), 7);

    auto diagonalInds = Vec<int32_t>::create(numDiagonals);

    DirectSolver<double> solver(boundary, b, A, diagonalInds, hand);

    boundary.freeMem();

    solver.solve(x, prealocatedForBiCGSTAB);

     std::cout << "x = \n" << GpuOut<double>(x, hand) << std::endl;

}


void testBiCGSTAB() {
     Handle hand;

     size_t denseMatDim = 4;

     auto indices = Vec<int32_t>::create(2);
     indices.get(0).set(1, hand);
     indices.get(1).set(-1, hand);

     auto bm = BandedMat<double>::create(denseMatDim, 2,indices);
     bm.col(0).fill(3, hand);
     bm.col(1).fill(-2, hand);

     auto b = Vec<double>::create(denseMatDim, hand);
     b.fill(1, hand);

     auto x = Vec<double>::create(denseMatDim, hand);

     BiCGSTAB<double> bs(b);

     bs.solveUnpreconditionedBiCGSTAB(bm, x);

     std::cout << "x = \n" << GpuOut<double>(x, hand) << std::endl;
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

     testPoisson(2, 2, 2, hand);

    // std::cout << "dimension size, number of iterations, total time" << std::endl;
    // for (size_t i = 2; i < 350; ++i) {
        // std::cout << i << ", ";
        // testPoisson(i, hand);
        // cudaDeviceSynchronize();

    // }



    return 0;
}



#endif //BICGSTAB_POISSONFDM_CUH
