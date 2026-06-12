//
// Created by usr on 6/11/26.
//

#ifndef CUDABANDED_EIGEN_CUH
#define CUDABANDED_EIGEN_CUH

#include "BoundaryConfig.cuh"
#include "deviceArrays/headers/SquareMat.h"
#include "kronecker/KroneckerTriplet.h"
#include "math/XYZ.cuh"
#include "solvers/Event.h"

template<typename T>
    class Eigen {
    Eigen(const XYZ<Vec<T>>& vals, const XYZ<SquareMat<T>>& vecs);

    static void generateEigen(const BoundaryConfig<T>& boundary, Handle *hands3, Event *events, std::shared_ptr<Mat<T>> (&preAllocatedForL_iX3)[3]);
public:
    /**
     * The eigen values, aka the spectrum.
     */
    const XYZ<Vec<T>> vals;
    /**
     * The eigen vectors.
     */
    const KroneckerTriplet<T> vecs;

    /**
     * Generates the eigenvector matrices.
     * @param boundary The boundary conditions.
     * @param hands3 A handle for each dimension.
     * @param events an event for each dimension - 1
     * @param hands3
     * @return The Laplacian's Eigen vector matrices.
     */
    static Eigen make(const BoundaryConfig<T> &boundary, Handle *hands3, Event *events);

    [[nodiscard]] GridDim dim() const;


    /**
     * @brief Dispatches the appropriate Eigen-decomposition kernel to generate the spectral basis.
     * * Reads the condition types (Dirichlet/Neumann) of the start and end boundaries,
     * computes the necessary normalization coefficients, and executes the corresponding
     * analytical eigen-kernel.

     * @param stream The CUDA stream to execute the kernel on.
     * @param eVecs  The pre-allocated 2D device array to store the eigenvectors.
     * The dimension $N$ is automatically deduced from `eVecs.cols`.
     * @param eVals Places the eigen values here.
     * @param axisSegment the axis segment these eigen values are on.
     */

    static void generateEigen(Handle& hand, SquareMat<T> eVecs, Vec<T> eVals, const UniformSegment<T>& axisSegment) ;

    static void generateEigen(Handle& hand, SquareMat<T> eVecs, Vec<T> eVals, const VariableSegment<T>& axisSegment);

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
