/**
 * @file Poisson.cuh
 * @brief Declares structured-grid Poisson solver interfaces.
 * @ingroup poisson
 *
 * @details
 * The Poisson module assembles structured operators, boundary metadata, and solver-facing data structures for grid-based elliptic solves.
 */


#ifndef CUDABANDED_POISSONLHS_H
#define CUDABANDED_POISSONLHS_H

#include "BoundaryConfig.cuh"
#include "deviceArrays/headers/sparse/BandedMat.h"
#include "deviceArrays/headers/SquareMat.h"
#include "solvers/Event.h"


namespace poisson {
    constexpr size_t numDiagonals3d = 7;
    constexpr size_t numDiagonals2d = 5;

    /**
     * @brief Builds the RHS vector for the Laplacian/Poisson system (core implementation).
     *
     * This function launches the CUDA kernel that fills an already-allocated RHS vector.
     * It contains the full boundary-condition logic and is the single source of truth
     * for RHS construction.
     *
     * @tparam T Floating-point type.
     * @param rhsCorrectionGoesHere Preallocated device array to store the RHS values.
     * @param stream CUDA stream used for asynchronous execution.
     */
    template<typename T, typename BoundaryConfigT>
    void boundaryCorrection(const BoundaryConfigT& boundary, SimpleArray<T> rhsCorrectionGoesHere, cudaStream_t stream);



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
    template<typename T, typename BoundaryConfigT>
    SimpleArray<T> boundaryCorrection(const BoundaryConfigT& boundary, cudaStream_t stream) ;


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
    template<typename T, typename BoundaryConfigT>
    BandedMat<T> laplacian(const BoundaryConfigT& boundary, cudaStream_t stream);

    /**
     * @brief Constructs the sparse/banded Laplacian operator matrix.
     *
     * This is the core implementation that builds both:
     * - the banded matrix representation of the Laplacian
     * - the diagonal index structure
     *
     * @tparam T Floating-point type.
     * @param boundary Boundary configuration defining Neumann/Dirichlet structure.
     * @param gridSizeXnumDiags The laplacian will be put in this space.
     * @param stream CUDA stream used for asynchronous execution.
     * @return Fully constructed banded Laplacian operator.
     */
    template<typename T, typename BoundaryConfigT>
    BandedMat<T> laplacian(const BoundaryConfigT& boundary, Mat<T>& gridSizeXnumDiags, Vec<int32_t>& numDiags, cudaStream_t stream);


}
#endif //CUDABANDED_POISSONLHS_H
