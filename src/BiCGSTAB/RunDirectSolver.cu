
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

    std::cout << "x = \n" << GpuOut<double>(x.tensor(height, depth), hand) << std::endl;

}

/**
 * benchmarks  the BiCGSTAV algorithm.
 * @param dim The size of a dimension
 * @param hand
 */
void testPoisson(size_t dim, Handle& hand) {
    testPoisson(dim, dim, dim, hand);
}

