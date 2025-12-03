
#include <iostream>

#include "deviceArrays/headers/handle.h"
#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/Streamable.h"
#include "Poisson/CubeBoundary.h"
#include "Poisson/DirectSolver.cu"
constexpr  size_t numDiagonals = 7;



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

    // std::cout << "RunDirectSolver testPoisson b: " << b.size() << std::endl << GpuOut<double>(b, hand) << std::endl;


    auto x = longVecs.col(1);
    auto A = longVecs.subMat(0, 2, boundary.internalSize(), numDiagonals);
    auto prealocatedForBiCGSTAB = longVecs.subMat(0, 2 + numDiagonals, boundary.internalSize(), 7);

    auto diagonalInds = Vec<int32_t>::create(numDiagonals);

    DirectSolver<double> solver(boundary, b, A, diagonalInds, hand);

    boundary.freeMem();

    solver.solve(x, prealocatedForBiCGSTAB);

    // std::cout << "x = \n" << GpuOut<double>(x.tensor(height, depth), hand) << std::endl;

}

/**
 * benchmarks  the BiCGSTAV algorithm.
 * @param dim The size of a dimension
 * @param hand
 */
void testPoisson(size_t dim, Handle& hand) {
    testPoisson(dim, dim, dim, hand);
}


// --- Helper function to create CUDA stream from Handle ---
cudaStream_t convert_handle(void* hand_ptr) {
    // For simplicity, we assume the void* passed from Fortran is the cudaStream_t itself.
    return static_cast<cudaStream_t>(hand_ptr);
}

// ======================================================================
// TEMPLATE IMPLEMENTATIONS
// ======================================================================

/**
 * @brief Template implementation for creating a DirectSolver object.
 */
template <typename T>
void C_DirectSolver_create_impl(
    void** solver_handle_out,
    const size_t height, const size_t width, const size_t depth,
    T* b_data, const size_t b_size,
    T* bandedA_data, const size_t bandedA_rows, const size_t bandedA_cols,
    int32_t* diag_indices_data, const size_t diag_indices_size,
    void* hand_ptr)
{
    cudaStream_t stream = convert_handle(hand_ptr);

    // 1. Create CubeBoundary (needed for constructor)
    CubeBoundary<T> boundary(height, width, depth, stream);

    // 2. Create device array wrappers
    Vec<T> b_vec(b_data, b_size, stream);
    Mat<T> bandedA_mat(bandedA_data, bandedA_rows, bandedA_cols, stream);
    Vec<int32_t> diag_indices_vec(diag_indices_data, diag_indices_size, stream);

    // 3. Create the DirectSolver object
    DirectSolver<T>* solver = new DirectSolver<T>(
        boundary,
        b_vec,
        bandedA_mat,
        diag_indices_vec,
        stream
    );

    *solver_handle_out = solver;
    boundary.freeMem();
}

/**
 * @brief Template implementation for calling the DirectSolver::solve method.
 */
template <typename T>
void C_DirectSolver_solve_impl(
    void* solver_handle,
    T* x_data, const size_t x_size,
    T* scratch_data, const size_t scratch_rows, const size_t scratch_cols,
    void* hand_ptr)
{
    DirectSolver<T>* solver = static_cast<DirectSolver<T>*>(solver_handle);
    cudaStream_t stream = convert_handle(hand_ptr); // Used for creating internal Mat/Vec

    // Reconstruct device array wrappers for inputs/outputs
    Vec<T> x_vec(x_data, x_size, stream);
    Mat<T> scratch_mat(scratch_data, scratch_rows, scratch_cols, stream);

    // Call the C++ solve method
    solver->solve(x_vec, scratch_mat);
}

/**
 * @brief Template implementation to free the memory of the DirectSolver object.
 */
template <typename T>
void C_DirectSolver_free_impl(void* solver_handle) {
    DirectSolver<T>* solver = static_cast<DirectSolver<T>*>(solver_handle);
    if (solver) {
        delete solver;
    }
}


// ======================================================================
// C-LINKAGE WRAPPERS (DOUBLE AND SINGLE PRECISION)
// Using extern "C" block for cleaner syntax.
// ======================================================================

extern "C" {

    // ------------------- DOUBLE PRECISION -------------------

    void C_DirectSolver_create_double(
        void** solver_handle_out,
        const size_t height, const size_t width, const size_t depth,
        double* b_data, const size_t b_size,
        double* bandedA_data, const size_t bandedA_rows, const size_t bandedA_cols,
        int32_t* diag_indices_data, const size_t diag_indices_size,
        void* hand_ptr)
    {
        C_DirectSolver_create_impl<double>(
            solver_handle_out, height, width, depth,
            b_data, b_size, bandedA_data, bandedA_rows, bandedA_cols,
            diag_indices_data, diag_indices_size, hand_ptr
        );
    }

    void C_DirectSolver_solve_double(
        void* solver_handle,
        double* x_data, const size_t x_size,
        double* scratch_data, const size_t scratch_rows, const size_t scratch_cols,
        void* hand_ptr)
    {
        C_DirectSolver_solve_impl<double>(
            solver_handle, x_data, x_size, scratch_data, scratch_rows, scratch_cols, hand_ptr
        );
    }

    void C_DirectSolver_free_double(void* solver_handle) {
        C_DirectSolver_free_impl<double>(solver_handle);
    }


    // ------------------- SINGLE PRECISION -------------------

    void C_DirectSolver_create_float(
        void** solver_handle_out,
        const size_t height, const size_t width, const size_t depth,
        float* b_data, const size_t b_size,
        float* bandedA_data, const size_t bandedA_rows, const size_t bandedA_cols,
        int32_t* diag_indices_data, const size_t diag_indices_size,
        void* hand_ptr)
    {
        C_DirectSolver_create_impl<float>(
            solver_handle_out, height, width, depth,
            b_data, b_size, bandedA_data, bandedA_rows, bandedA_cols,
            diag_indices_data, diag_indices_size, hand_ptr
        );
    }

    void C_DirectSolver_solve_float(
        void* solver_handle,
        float* x_data, const size_t x_size,
        float* scratch_data, const size_t scratch_rows, const size_t scratch_cols,
        void* hand_ptr)
    {
        C_DirectSolver_solve_impl<float>(
            solver_handle, x_data, x_size, scratch_data, scratch_rows, scratch_cols, hand_ptr
        );
    }

    void C_DirectSolver_free_float(void* solver_handle) {
        C_DirectSolver_free_impl<float>(solver_handle);
    }
}