/**
 * @file Eigen.cuh
 * @brief Declares eigenbasis helpers used by Poisson solvers.
 * @ingroup poisson
 *
 * @details
 * The Poisson module assembles structured operators, boundary metadata, and solver-facing data structures for grid-based elliptic solves.
 */

#ifndef CUDABANDED_EIGEN_CUH
#define CUDABANDED_EIGEN_CUH

#include "BoundaryConfig.cuh"
#include "deviceArrays/headers/SquareMat.h"
#include "kronecker/KroneckerTriplet.h"
#include "math/XYZ.cuh"
#include "solvers/Event.h"

template<typename T>
    class Eigen {
    Eigen(const XYZ<Vec<T>> &vals, const KroneckerTriplet<T> &vecs, const KroneckerTriplet<T> &vecsInv);

    template<typename BoundaryConfigT>
    static void generateEigen(const BoundaryConfigT& boundary, Handle *hands3, Event *events, std::shared_ptr<Mat<T>> (&preAllocatedForL_iX3)[3]);



public:
    /**
     * The eigen values, aka the spectrum.
     */
    const XYZ<Vec<T>> vals;
    /**
     * The eigen vectors.
     */
    const KroneckerTriplet<T> vecs;
    const KroneckerTriplet<T> vecsInv;

    /**
     * Generates the eigenvector matrices.
     * @param boundary The boundary conditions.
     * @param hands3 A handle for each dimension.
     * @param events2 an event for each dimension - 1
     * @param hands3
     * @return The Laplacian's Eigen vector matrices.
     */
    template<typename BoundaryConfigT>
    static Eigen make(const BoundaryConfigT &boundary, Handle *hands3, Event *events2);

    [[nodiscard]] GridDim dim() const;


    /**
     * @brief Dispatches the appropriate Eigen-decomposition kernel to generate the spectral basis.
     * * Reads the condition types (Dirichlet/Neumann) of the start and end boundaries,
     * computes the necessary normalization coefficients, and executes the corresponding
     * analytical eigen-kernel.

     * @param stream The CUDA stream to execute the kernel on.
     * @param eVecs  The pre-allocated 2D device array to store the eigenvectors.
     * The dimension $N$ is automatically deduced from `eVecs.cols`.
     * @param eVecsInv
     * @param eVals Places the eigen values here.
     * @param seg the axis segment these eigen values are on.
     */

    static void generateEigen(Handle &hand, SquareMat<T> &eVecs, SquareMat<T> &eVecsInv, Vec<T> &eVals, const UniformSegment<T> &seg) ;

    static void generateEigen(Handle &hand, SquareMat<T> &eVecs, SquareMat<T> &eVecsInv, Vec<T> &eVals, const VariableSegment<T> &axisSegment);

    /**
     * Assigns the eigenvectors and eigen values.
     * @tparam axisSegmentT Should either be a VariableSegment or a UniformSegment.
     * @param hand
     * @param eigins the first n columns will have eigen vectors assigned to them, and the last column will
     * have all the eigenvalues assigned to it.
     * @param axisSegment Boundary conditions and spacing.
     */
    template<typename axisSegmentT>
    static void generateEigen(Handle& hand, Mat<T> eigins, const axisSegmentT& axisSegment) ;

};



#endif //CUDABANDED_EIGEN_CUH
