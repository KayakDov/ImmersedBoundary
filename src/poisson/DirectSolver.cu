#include "DirectSolver.cuh"
#include "DirectSolver.cuh"

#include "ToeplitzLaplacian.cuh"
//TODO: Combine this class  with RunDirectSolver class, and then make into an hpp or split into .cuh and .cu.



template<typename T>
DirectSolver<T>::DirectSolver(const CubeBoundary<T> &boundary, Vec<T> &f, Mat<T> &preAlocatedForBandedA,
                              Vec<int32_t> &prealocatedForIndices,
                              cudaStream_t stream) : PoissonRHS<T>(boundary, f, stream),
                                                     A(ToeplitzLaplacian<T>(this->dim).setA(stream, preAlocatedForBandedA, prealocatedForIndices)) {
}

template<typename T>
void DirectSolver<T>::solve(Mat<T> prealocatedForBiCGSTAB) {
    cudaDeviceSynchronize();
    BiCGSTAB<T>::solve(A, this->_b, &prealocatedForBiCGSTAB);
}

/**
 * Creates and solved an example Poisson class on a cube with the given side length.
 * @param dimLength The length of an edge of the grid.  //up to 325 works on Dov's computer.  After that the size of
 * the initally allocated memory exceeds the available memory on the gpu.
 */
void testPoisson(const size_t height, size_t width, size_t depth, Handle &hand) {
    auto boundary = CubeBoundary<double>::ZeroTo1(height, width, depth, hand);

    auto longVecs = Mat<double>::create(boundary.internalSize(), 1 + numDiagonals3d + 7);
    auto b = longVecs.col(0);
    b.fill(0, hand);

    // std::cout << "RunDirectSolver testPoisson b: " << b.size() << std::endl << GpuOut<double>(b, hand) << std::endl;

    auto A = longVecs.subMat(0, 1, boundary.internalSize(), numDiagonals3d);
    auto prealocatedForBiCGSTAB = longVecs.subMat(0, 1 + numDiagonals3d, boundary.internalSize(), 7);

    auto diagonalInds = Vec<int32_t>::create(numDiagonals3d);

    DirectSolver<double> solver(boundary, b, A, diagonalInds, hand);

    boundary.freeMem();

    solver.solve(prealocatedForBiCGSTAB);

    // std::cout << "x = \n" << GpuOut<double>(x.tensor(height, depth), hand) << std::endl;
}

/**
 * benchmarks  the BiCGSTAV algorithm.
 * @param dim The size of a dimension
 * @param hand
 */
void testPoisson(size_t dim, Handle &hand) {
    testPoisson(dim, dim, dim, hand);
}

template class DirectSolver<float>;
template class DirectSolver<double>;
