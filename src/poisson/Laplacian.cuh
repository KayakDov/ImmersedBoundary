
#ifndef CUDABANDED_POISSONLHS_H
#define CUDABANDED_POISSONLHS_H
#include <array>
#include <vector>

#include "../deviceArrays/headers/sparse/BandedMat.h"
#include "deviceArrays/headers/SquareMat.h"
#include "math/XYZ.h"
#include "poisson/BoundaryCondition.cuh"
#include "solvers/Event.h"


namespace poisson {
    constexpr size_t numDiagonals3d = 7;
    constexpr size_t numDiagonals2d = 5;

    /**
     * The one dimensional Laplacians for a 2 or 3d grid.  Note, if a pair of Laplacians have the same dimension size and
     * boundary conditions, then the pointers may point to the same memory.
     * @tparam T
     */
    template<typename T>
    class Laplacian1d {
        SimpleArray<int32_t> inds;
        const BoundaryConfig<T> boundary;

    public:
        const XYZ<Mat<T>> rawBanded;
        /**
         *
         * @param boundary The boundary for the 1d laplacians.
         * @param hand The context.
         */
        Laplacian1d(const BoundaryConfig<T> &boundary, Handle &hand);
        /**
         * Selects one of the laplacian 1d operators.
         * @param dim 0 for the x dimension, 1 for the y dimesnion, and 2 for the z dimension.
         * @return a 1d operator.
         */
        BandedMat<T> banded(size_t dim);

        /**
         * The square matrix of the given dimension.
         * @param dim 0 for x, 1 for y, 2 for z.
         * @param hand
         * @return A square 1d laplacian matrix.
         */
        SquareMat<T> dense(size_t dim, Handle &hand);
    };

    template<typename T>
    class Eigen {
        Eigen(const XYZ<Vec<T>>& vals, const XYZ<SquareMat<T>>& vecs);
    public:
        /**
         * The eigen values.
         */
        const XYZ<Vec<T>> vals;
        /**
         * The eigen vectors.
         */
        const XYZ<SquareMat<T>> vecs;

        /**
         * Generates the eigenvector matrices.
         * @param boundary The boundary conditions.
         * @param hands A handle for each dimension.
         * @param events an event for each dimension - 1
         * @param hands
         * @return The Laplacian's Eigen vector matrices.
         */
        static Eigen make(const BoundaryConfig<T> &boundary, Handle *hands, Event *events);

        GridDim dim() const;

    };


    /**
     * @brief Builds the RHS vector for the Laplacian/Poisson system (core implementation).
     *
     * This function launches the CUDA kernel that fills an already-allocated RHS vector.
     * It contains the full boundary-condition logic and is the single source of truth
     * for RHS construction.
     *
     * @tparam T Floating-point type.
     * @param boundary Boundary configuration (Neumann/Dirichlet/mixed conditions).
     * @param correctionGoesHere Preallocated device array to store the RHS values.
     * @param stream CUDA stream used for asynchronous execution.
     */
    template<typename T>
    void boundaryCorrection(const BoundaryConfig<T>& boundary, SimpleArray<T>& correctionGoesHere, cudaStream_t stream);


    /**
     * @brief Constructs and returns a RHS vector for the Laplacian/Poisson system.
     *
     * Allocates device memory internally and computes the RHS using the core implementation.
     *
     * @tparam T Floating-point type.
     * @param boundary Boundary configuration (Neumann/Dirichlet/mixed conditions).
     * @param stream CUDA stream used for asynchronous execution.
     * @return Device array containing the computed RHS vector.
     */
    template<typename T>
    SimpleArray<T> boundaryCorrection(const BoundaryConfig<T>& boundary, cudaStream_t stream);


    /**
     * @brief Constructs the sparse/banded Laplacian operator matrix.
     *
     * This is the core implementation that builds both:
     * - the banded matrix representation of the Laplacian
     * - the diagonal index structure
     *
     * @tparam T Floating-point type.
     * @param boundary Boundary configuration defining Neumann/Dirichlet structure.
     * @param stream CUDA stream used for asynchronous execution.
     * @return Fully constructed banded Laplacian operator.
     */
    template<typename T>
    BandedMat<T> laplacian(const BoundaryConfig<T>& boundary,
                                cudaStream_t stream);

}
#endif //CUDABANDED_POISSONLHS_H
