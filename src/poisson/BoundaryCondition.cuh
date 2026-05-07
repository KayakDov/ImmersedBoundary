/**
 * @file BoundaryCondition.h
 * @brief Defines boundary condition handling for discrete Laplacian systems on staggered grids.
 *
 * This module provides an abstraction for incorporating Dirichlet and Neumann
 * boundary conditions into a linear system of the form:
 *
 *     L u = b
 *
 * where L is a discrete Laplacian operator and b is the right-hand side vector.
 *
 * The implementation assumes a staggered (cell-centered) grid, where domain
 * boundaries lie halfway between interior grid points. Boundary conditions are
 * enforced by modifying the system matrix L and RHS vector b using finite
 * difference stencils.
 */

#pragma once
#include <functional>

#include "deviceArrays/headers/DeviceData.cuh"
#include "solvers/Event.h"
#include <limits>


/**
 * @class BoundaryCondition
 * @brief Abstract base class for boundary condition enforcement.
 *
 * Provides a unified interface for modifying the system matrix and RHS vector
 * to account for boundary conditions. Derived classes implement specific
 * boundary types (e.g., Dirichlet, Neumann).
 *
 * @tparam Real Floating-point type (e.g., float, double)
 */
template<typename Real>
class BoundaryCondition {
public:
    /// Boundary value (Dirichlet value or Neumann gradient)
    const Real value;

    /// Precomputed 1 / (delta^2)
    const Real inverseDeltaSquared;

    /// Precomputed 1 / delta
    const Real inverseDelta;

    const bool isNeumann;
    const bool isStaggered;
    __host__ __device__ bool isDirichlet() const { return !isNeumann;}
    __host__ __device__ bool isNodeCentered() const { return !isStaggered;}

    /**
     * @brief Construct a boundary condition.
     *
     * @param value Boundary value:
     *              - Dirichlet: prescribed field value
     *              - Neumann: prescribed normal derivative
     * @param delta Grid spacing in the relevant direction
     * @param isNeumann Neumann or Dirichlet
     * @param isStaggered Staggered or node centered.
     */
    __device__ __host__ BoundaryCondition(Real value, Real delta, bool isNeumann, bool isStaggered) : value(value), inverseDeltaSquared(1/(delta*delta)), inverseDelta(1/delta), isNeumann(isNeumann), isStaggered(isStaggered) {};

    /**
     * An empty conditon, similar to a null pointer.
     */
    __device__ __host__ BoundaryCondition() : value(0), inverseDeltaSquared(0), inverseDelta(0), isNeumann(false), isStaggered(false) {};

    /**
     * @brief Apply boundary condition contribution to system.
     *
     * Modifies the matrix L for a grid point adjacent to the boundary.
     *
     */
    __device__ void setL(Real& mainDiagVal, Real& offDiagVal) const {
        if (isNeumann)
            mainDiagVal -= this->inverseDeltaSquared;
        else {
            if (isStaggered) mainDiagVal -= 3*this->inverseDeltaSquared;
            else mainDiagVal -= 2 * this->inverseDeltaSquared;
        }
        offDiagVal = this->inverseDeltaSquared;
    }
    /**
    * @brief Equality operator implemented as a hidden friend.
    * Compares all precomputed factors and the condition type.
    */
    __host__ __device__ friend bool operator==(const BoundaryCondition& lhs, const BoundaryCondition& rhs) {
        return (lhs.isNeumann == rhs.isNeumann) &&
               (lhs.isStaggered == rhs.isStaggered) &&
               (lhs.value == rhs.value) &&
               (lhs.inverseDelta == rhs.inverseDelta);
    }

    /**
     * @brief Inequality operator.
     */
    __host__ __device__ friend bool operator!=(const BoundaryCondition& lhs, const BoundaryCondition& rhs) {
        return !(lhs == rhs);
    }


    /**
     * @brief Apply boundary condition contribution to RHS only.
     *
     * Adds the boundary-condition-induced contribution to the right-hand side
     * vector, assuming the operator (L) is assembled independently.
     *
     * @param gridIndFlattened Index of the current grid point
     * @param rhs Right-hand side vector (modified in place)
     */
    __device__ void setBoundaryRHS(Real& rhsVal) const {
        Real contribution = 0;

        if (isNeumann) contribution = -this->value * this->inverseDelta;
        else {
            if (isStaggered) contribution = -2*this->value * this->inverseDeltaSquared;
            else contribution = -this->value * this->inverseDeltaSquared;
        }


        atomicAdd(&rhsVal, contribution);
    }
    /**
     * @brief Apply boundary condition contribution to system.
     *
     * Modifies the matrix L and RHS vector b for a grid point adjacent to the boundary.
     */
    __device__ void setLAndRHS(Real& mainDiagVal, Real& offDiagVal, Real& rhsVal) const {
        setL(mainDiagVal, offDiagVal);
        setBoundaryRHS(rhsVal);
    }

    /**
     * It's possible to create an undefined boundary condition.  This is to allow for flexabuility between 2 and 3 dimesnional creations.
     * @return true if this us undefined, false otherwise.
     */
    __device__ __host__ bool isUndefined() const{
        return inverseDelta == 0;
    }
};

/**
 * @class BoundaryPair
 * @brief Encapsulates the boundary conditions for both ends of a 1D grid segment.
 *
 * This class stores the physical and numerical constraints for the start and end
 * of a dimension (e.g., Left/Right, Top/Bottom). By storing these by value,
 * the pair maintains its own valid copy of the conditions, preventing dangling
 * references during kernel execution.
 * * @tparam T The floating-point type (float or double).
 */
template<typename T>
class BoundaryPair {
public:
    /** @brief The condition applied at the start (lower index) of the segment. */
    const BoundaryCondition<T> start;

    /** @brief The condition applied at the end (higher index) of the segment. */
    const BoundaryCondition<T> end;

    /** @brief The length of the dimension.*/
    const size_t dimLength;

    /**
    * @brief Construct by providing raw parameters for both boundaries.
     * This constructor initializes the internal BoundaryCondition objects directly.
     *
     * @param beginIsNeumann Is the beggin condition Neuman.  Set to false for Dirichlet.
     * @param endIsNeumann Is the end condition Neumann.  Set to false for Dirichlet.
     * @param beginVal The value at the beginning condition.
     * @param endVal The value at the end condition.
     * @param isStaggered True if the grid is staggered, false if it's node centered.
     * @param delta The distance between grid points.
     * @param dimLength The numver of grid points in this dimension.
     */
    __device__ __host__ BoundaryPair(bool beginIsNeumann, bool endIsNeumann, T beginVal, T endVal, bool isStaggered, double delta, size_t dimLength) :
        start(beginVal, delta, beginIsNeumann, isStaggered),
        end((endIsNeumann ? -endVal : endVal), delta, endIsNeumann, isStaggered),
        dimLength(dimLength)
    {}

    /**
     * @brief Equality operator for condition pairs.
     * Required for deduplication in spectral or multi-dimensional setups.
     */
    __host__ __device__ friend bool operator==(const BoundaryPair& lhs,
                                               const BoundaryPair& rhs) {
        return (lhs.start == rhs.start) && (lhs.end == rhs.end) && (lhs.dimLength == rhs.dimLength);
    }

    __host__ __device__ friend bool operator!=(const BoundaryPair& lhs,
                                               const BoundaryPair& rhs) {
        return !(lhs == rhs);
    }


    /**
     * returns the start or the end.
     * @param isEnd true if you want the end boundary condition, false if you want the start.
     * @return the desired boundary condition.
     */
    __host__ __device__ const BoundaryCondition<T>& operator[](bool isEnd) const {
        return isEnd ? end : start;
    }

    /**
     * @brief Apply boundary condition contribution to system.
     *
     * Modifies the matrix L for a grid point adjacent to the boundary.
     * @return true if a modification is made, false otherwise.
     */
    __device__ bool setL(T& mainDiagVal, T& leftDiagonal, T& rightDiagonal, const size_t indexInLine) const {
        if (indexInLine == 0) start.setL(mainDiagVal, rightDiagonal);
        else if (indexInLine == dimLength - 1) end.setL(mainDiagVal, leftDiagonal);
        else return false;
        return true;
    }

    /**
     * @brief Apply boundary condition contribution to RHS only.
     *
     * Adds the boundary-condition-induced contribution to the right-hand side
     * vector, assuming the operator (L) is assembled independently.
     *
     * @param rhs Right-hand side vector (modisfied in place)
     * @param indexInLine  The index of this point in this dimension.
     * @return true if a modification is made, false otherwise.
     */
    __device__ bool setBoundaryRHS1d(DeviceData1d<T>& rhs, const size_t indexInLine) const {
        if (indexInLine == 0) start.setBoundaryRHS(rhs[0]);
        else if (indexInLine == dimLength - 1) end.setBoundaryRHS(rhs[dimLength - 1]);
        else return false;
        return true;
    }
    /**
     * @brief Apply boundary condition contribution to system.
     *
     * Modifies the matrix L and RHS vector b for a grid point adjacent to the boundary.
     * @return true if a modification is made, false otherwise.
     * @return true if a modification is made, false otherwise.
     */
    __device__ bool setLAndRHS(T& mainDiagVal, T& startDiagVal, T& endDiagonalVal, DeviceData1d<T> rhs, const size_t indexInLine) const {
        setL(mainDiagVal, startDiagVal, endDiagonalVal, indexInLine);
        return setBoundaryRHS1d(rhs, indexInLine);
    }

    /**
     * @brief Dispatches the appropriate Eigen-decomposition kernel to generate the spectral basis.
     * * Reads the condition types (Dirichlet/Neumann) of the start and end boundaries,
     * computes the necessary normalization coefficients, and executes the corresponding
     * analytical eigen-kernel.
     * @param stream The CUDA stream to execute the kernel on.
     * @param eVecs  The pre-allocated 2D device array to store the eigenvectors.
     * The dimension $N$ is automatically deduced from `eVecs.cols`.
     * @param eVals Places the eigen values here.
     */
    void generateEigen(cudaStream_t stream, SquareMat<T> eVecs, Vec<T> eVals) const;


    /**
     * Builds the eigenvecotrs and eigenvalues.
     * @param stream
     * @param lengthXLengthPlus1 The eigenvecotrs are places in the beggining, and the last column gets the eigenvalues.
     */
    void generateEigen(cudaStream_t stream, Mat<T>& lengthXLengthPlus1) const {
        generateEigen(stream, lengthXLengthPlus1.sqSubMatFirstBiggest(), lengthXLengthPlus1.lastCol());
    }

    __host__ __device__ bool isUndefined() {
        return dimLength == static_cast<size_t>(-1);
    }

    /**
     * Are both conditions Neumann resulting in singular 1d matrix?
     * @return true if both conditions are neumann.
     */
    __host__ bool bothNeumann() const {
        return start.isNeumann && end.isNeumann;
    }
};


