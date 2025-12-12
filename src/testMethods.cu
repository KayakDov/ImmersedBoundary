
#include <chrono>
#include "Poisson/CubeBoundary.h"
#include "Poisson/EigenDecompSolver.h"
#include "Poisson/DirectSolver.cuh"
// #include "FortranBindings.hpp"
#include "wrapFortranBindings.h"
using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

template<typename T>
void benchMarkEigenDecompSolver(size_t height, size_t width, size_t depth, std::array<Handle, 3>& hand3) {
    const auto boundary = CubeBoundary<double>::ZeroTo1(height, width, depth, hand3[0]);

    auto memAlocFX = Mat<double>::create(boundary.internalSize(), 2);


    size_t maxDim = std::max(width, std::max(height, depth));

    Mat<T> eigenStorage = Mat<T>::create(maxDim, width + height + depth + 3);
    SquareMat<T> eX = eigenStorage.sqSubMat(0, 0, width),
            eY = eigenStorage.sqSubMat(0, width, height),
            eZ = eigenStorage.sqSubMat(0, width + height, depth);

    Mat<T> vals = eigenStorage.subMat(0, width + height + depth, maxDim, 3);

    auto x = memAlocFX.col(0);
    auto f = memAlocFX.col(1);

    f.fill(0, hand3[0]);

    cudaDeviceSynchronize();

    TimePoint start = std::chrono::steady_clock::now();
    EigenDecompSolver<double> fdm(boundary, x, f, eX, eY, eZ, vals, hand3);
    cudaDeviceSynchronize();
    TimePoint end = std::chrono::steady_clock::now();


    double iterationTime = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();

    std::cout << iterationTime << ", ";

    // std::cout << "x = \n" << GpuOut<double>(x.tensor(height, depth), hand3[0]) << std::endl;
}

template<typename T>
void benchMarkEigenDecompSolver(size_t dim, std::array<Handle, 3>& hand3) {
    benchMarkEigenDecompSolver<T>(dim, dim, dim, hand3);
}

/**
 * @brief Main entry point to demonstrate the FastDiagonalizationMethod
 *        for a 2x2x2 grid.
 */
int main() {
    std::array<Handle, 3> hand3;

    constexpr size_t maxDimensions = 5;

    std::cout << "dim, time" << std::endl;

    for (size_t dim = 1; dim < maxDimensions; dim++) {
        std::cout << dim << ", ";
        benchMarkEigenDecompSolver<double>(dim, hand3);
        cudaDeviceSynchronize();

        testPoisson(dim, hand3[0]);
        std::cout << std::endl;
    }

    const size_t n = 3;
    const size_t numInds = 3;
    Handle hand;
    auto inds = Vec<int32_t>::create(numInds, hand);
    auto banded = BandedMat<float>::create(n, numInds, inds);
    auto workSpace = Mat<float>::create(n, 7);
    auto b = Vec<float>::create(n);

    std::vector<int32_t> indsHost = {1, -1, 0};
    inds.set(indsHost.data(), hand);

    std::vector<float> bandedCol0 = {1, 3, 0};
    std::vector<float> bandedCol1 = {2, 5, 0};
    std::vector<float> bandedCol2 = {1, 2, 3};
    banded.col(0).set(bandedCol0.data(), hand);
    banded.col(1).set(bandedCol1.data(), hand);
    banded.col(2).set(bandedCol2.data(), hand);

    std::vector<float> bHost = {4, 12, 9};
    b.set(bHost.data(), hand);


    BiCGSTAB<float>::solve(banded, b, &workSpace);

    std::cout << "x = " << GpuOut<float>(b, hand) << std::endl;

    return 0;
}



// Helper function to check for CUDA errors
// void checkCudaError(cudaError_t err, const std::string& msg) {
//     if (err != cudaSuccess) {
//         std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }

// int main() {
//     // Parameters matching the Fortran program
//     const size_t n = 3;
//     const size_t numDiags = 3;
//
//     // Leading dimensions
//     const size_t aLd = n;
//     const size_t bStride = 1;
//     const size_t indsStride = 1;
//     const size_t workLd = n;
//     const size_t workspaceCols = 7; // size X 7 matrix
//
//     const size_t maxIter = 1000;
//     const  double tol = 1e-6;
//
//     // --- Host data initialization ---
//     std::vector<double> A_h = {
//         1.0, 3.0, 1.0, // Column 1 (diag 1)
//         2.0, 5.0, 2.0, // Column 2 (diag -1)
//         1.0, 2.0, 3.0  // Column 3 (diag 0)
//     };
//     std::vector<int32_t> inds_h = {1, -1, 0};
//
//     std::vector<double> b_h = {4.0, 12.0, 9.0};
//
//     double* A_d = nullptr;
//     int32_t* inds_d = nullptr;
//     double* b_d = nullptr;
//     double* workspace_d = nullptr;
//
//     // --- Allocate device memory ---
//     checkCudaError(cudaMalloc(&A_d, n * numDiags * sizeof(double)), "A_d allocation");
//     checkCudaError(cudaMalloc(&inds_d, numDiags * sizeof(int32_t)), "inds_d allocation");
//     checkCudaError(cudaMalloc(&b_d, n * sizeof(double)), "b_d allocation");
//     checkCudaError(cudaMalloc(&workspace_d, n * workspaceCols * sizeof(double)), "workspace_d allocation");
//
//     // --- Copy host data to device ---
//     checkCudaError(cudaMemcpy(A_d, A_h.data(), n * numDiags * sizeof(  double), cudaMemcpyHostToDevice), "A_d copy");
//     checkCudaError(cudaMemcpy(inds_d, inds_h.data(), numDiags * sizeof(int32_t), cudaMemcpyHostToDevice), "inds_d copy");
//     checkCudaError(cudaMemcpy(b_d, b_h.data(), n * sizeof(  double), cudaMemcpyHostToDevice), "b_d copy");
//
//     std::cout << "--- Starting BiCGSTAB Solver (n=" << n << ", maxIter=" << maxIter << ") ---" << std::endl;
//
//     FOR_solveBiCGSTAB_double(
//         reinterpret_cast<size_t>(A_d),
//         aLd,
//         reinterpret_cast<size_t>(inds_d),
//         indsStride,
//         numDiags,
//         reinterpret_cast<size_t>(b_d),
//         bStride,
//         n,
//         reinterpret_cast<size_t>(workspace_d),
//         workLd,
//         maxIter,
//         tol
//     );
//
//     std::cout << "--- Solver Finished ---" << std::endl;
//
//     // --- Retrieve solution (x is now in b_d) ---
//     std::vector<double> x_h(n);
//     checkCudaError(cudaMemcpy(x_h.data(), b_d, n * sizeof(double), cudaMemcpyDeviceToHost), "x_h copy");
//
//     std::cout << "Solution x (from device):" << std::endl;
//     for (size_t i = 0; i < n; ++i)  std::cout << "  x[" << i << "] = " << x_h[i] << std::endl;
//
//     checkCudaError(cudaFree(A_d), "A_d free");
//     checkCudaError(cudaFree(inds_d), "inds_d free");
//     checkCudaError(cudaFree(b_d), "b_d free");
//     checkCudaError(cudaFree(workspace_d), "workspace_d free");
//
//     return 0;
// }