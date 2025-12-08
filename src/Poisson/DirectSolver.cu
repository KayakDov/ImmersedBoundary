
 #ifndef BICGSTAB_POISSONFDM_CUH
#define BICGSTAB_POISSONFDM_CUH
#include "Poisson.h"
#include "../deviceArrays/headers/BandedMat.h"
#include "../BiCGSTAB/BiCGSTAB.cu"
#include "../deviceArrays/headers/DeviceMemory.h"
#include "deviceArrays/headers/Streamable.h"


struct AdjacencyInd {
    /**
     * The column in the banded matrix.
     */
    const size_t col;
    /**
     * The index of the diagonal that is held by that column.
     */
    const int32_t diag;
    __device__ __host__ AdjacencyInd(const size_t row, const int32_t diag) : col(row), diag(diag) {
    }
};

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
     * @param[in] aInd The index of the diagonal corresponding to the neighbor being checked.
     */
    __device__ void operator()(const AdjacencyInd aInd) {

        const size_t rowInd = modPos(static_cast<int32_t>(idGrid) + min(aInd.diag, 0), static_cast<int32_t>(a.rows));
        if (rowInd < a.rows - abs(aInd.diag)) a(rowInd, aInd.col) = static_cast<T>(0);
        else a(rowInd, aInd.col) = NAN;
    }
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
 *
 */
template <typename T>
__global__ void setAKernel(DeviceData2d<T> a,
    const GridDim g,
    const AdjacencyInd here, const AdjacencyInd up, const AdjacencyInd down, const AdjacencyInd left,
    const AdjacencyInd right, const AdjacencyInd front, const AdjacencyInd back
    ) {
    const GridInd3d ind;

    if (ind >= g) return;

    const size_t idGrid = g[ind];

    a(idGrid, here.col) = -6;
    Set0<T> set0(a, idGrid);

    if (ind.row == 0) set0(up);
    else if (ind.row == g.rows - 1) set0(down);

    if (ind.col == 0) set0(left);
    else if (ind.col == g.cols - 1) set0(right);

    if (ind.layer == 0) set0(front);
    else if (ind.layer == g.layers - 1) set0(back);
}

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
    const AdjacencyInd here, up, down, left, right, back, front;

private:
    const BandedMat<T> A;

    /**
     * @brief Launch kernel that assembles A in banded/dense storage.
     *
     * @param numInds The number of indices.
     * @param mindices device pointer to the int32_t offsets array (length numNonZeroDiags).
     * @param handle contains the stream to run on.
     * @param preAlocatedForA Provide prealocated memory here to be written to, numDiagonals x _b.size().
     */
    BandedMat<T> setA(cudaStream_t& stream, Mat<T>& preAlocatedForA, Vec<int32_t>& preAlocatedForIndices) {

        preAlocatedForA.subMat(0, 1, preAlocatedForA._rows, preAlocatedForA._cols - 1).fill(1, stream);

        const KernelPrep kp = this->dim.kernelPrep();
        setAKernel<T><<<kp.numBlocks, kp.threadsPerBlock, 0, stream>>>(
            preAlocatedForA.toKernel2d(), this->dim,
            here, up, down, left, right, front, back
        );

        CHECK_CUDA_ERROR(cudaGetLastError());

        loadMapRowToDiag(preAlocatedForIndices, stream);

        return BandedMat<T> (preAlocatedForA, preAlocatedForIndices);
    }

    void loadMapRowToDiag(Vec<int32_t> diags, const cudaStream_t stream) const {
        int32_t diagsCpu[diags.size()];
        diagsCpu[up.col] = up.diag;
        diagsCpu[down.col] = down.diag;
        diagsCpu[left.col] = left.diag;
        diagsCpu[right.col] = right.diag;
        diagsCpu[back.col] = back.diag;
        diagsCpu[front.col] = front.diag;
        diagsCpu[here.col] = here.diag;
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
    * @param[in] f The initial right-hand side vector, pre-loaded with the source term $f$.  This will be overwritten.
    * This vector is modified by the solver to include boundary contributions.
    * @param prealocatedForIndices
    */
    DirectSolver(const CubeBoundary<T>& boundary, Vec<T>& f, Mat<T>& preAlocatedForBandedA, Vec<int32_t>& prealocatedForIndices, cudaStream_t stream):
        Poisson<T>(boundary, f, stream),
        here(0, 0),
        up(1, -1),
        down(2, 1),
        left(3, -this->dim.rows * this->dim.layers),
        right(4, this->dim.rows * this->dim.layers),
        front(5, -this->dim.rows),
        back(6, this->dim.rows),
        A(setA(stream, preAlocatedForBandedA, prealocatedForIndices))
    {}

    /**
     * @brief Solves the Poisson equation for the grid.
     *
     * Automatically dispatches to the 2D or 3D solver based on whether the number of layers is 1.
     *
     * @param[out] x Pre-allocated memory that the solution will be written to.
     * @param A Pre-allocated memory that will be used to compute the solution.  It should be numDiagonals rows and _b.size() columns.
     * @param[in] hand The CUDA handle (stream/context) to manage the computation.
     */
    void solve(Mat<T> prealocatedForBiCGSTAB) {

        cudaDeviceSynchronize();
        BiCGSTAB<T>::solve(A, this->_b, &prealocatedForBiCGSTAB);
    }
};

#endif //BICGSTAB_POISSONFDM_CUH
