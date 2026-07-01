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
 * @ingroup poisson
 */

#pragma once

#include "deviceArrays/headers/SimpleArray.h"

template<typename Real>
class BoundaryCondition {
public:
    /// Boundary value (Dirichlet value or Neumann gradient)
    const Real value;
    const bool isNeumann;

    BoundaryCondition(Real value, bool isNeumann): value(value), isNeumann(isNeumann) {};

    __host__ __device__ bool isDirichlet() const { return !isNeumann;}
    /**
    * @brief Equality operator implemented as a hidden friend.
    * Compares all precomputed factors and the condition type.
    */
    __host__ __device__ friend bool operator==(const BoundaryCondition& lhs, const BoundaryCondition& rhs) {
        return (lhs.isNeumann == rhs.isNeumann) &&
               (lhs.value == rhs.value) ;
    }

    /**
     * @brief Inequality operator.
     */
    __host__ __device__ friend bool operator!=(const BoundaryCondition& lhs, const BoundaryCondition& rhs) {
        return !(lhs == rhs);
    }


};

template<typename Real, bool isEnd>
class VariableBoundary: public BoundaryCondition<Real> {
public:
    const DeviceData1d<Real> deltaPM;

    /**
     *
     * @param isNeumann
     * @param val
     * @param deltaPM If isEnd is false (we're at te beginning), then the first value should be the distance to the beggining boundary and the
     * second should be the distance between the first two elements.  If isEnd is true, then the first value should
     * be the distance between the last two elements, and the second mvalue should be the distance between the last
     * element and the clsoing boundary.
     */
    VariableBoundary(bool isNeumann, Real val, const SimpleArray<Real>& deltaPM) :
        BoundaryCondition<Real>(val, isNeumann),
        deltaPM(deltaPM)
    {}

    __device__ void setL(Real& mainDiagVal, Real& offDiagVal) const {
        Real dp = deltaPM[1], dm = deltaPM[0];
        Real distTerm = 2/(dp + dm);

        if (this->isNeumann) {
            if constexpr (isEnd) mainDiagVal -= (offDiagVal = distTerm/dm);
            else mainDiagVal -= (offDiagVal = distTerm/dp);
        } else {
            if constexpr (isEnd) offDiagVal = distTerm / dm;
            else offDiagVal = distTerm / dp;
            mainDiagVal -= 2/(dp * dm);
        }
    }

    __device__ void setBoundaryRHS(Real& rhsVal) const {
        Real contribution = 0;

        Real dp = deltaPM[1], dm = deltaPM[0];
        Real distTerm = 2/(dp + dm);

        if (this->isNeumann) {
            if constexpr (isEnd) contribution = this->value * distTerm;
            else contribution = -this->value * distTerm;
        } else {
            if constexpr (isEnd) contribution = -distTerm * this->value/dp;
            else contribution = -distTerm * this->value/dm;
        }

        atomicAdd(&rhsVal, contribution);
    }

    __device__ __host__ bool isUndefined() const {
        return deltaPM.cols != 2;
    }

};

/**
 * @class UniformBoundary
 * @brief Abstract base class for boundary condition enforcement.
 *
 * Provides a unified interface for modifying the system matrix and RHS vector
 * to account for boundary conditions. Derived classes implement specific
 * boundary types (e.g., Dirichlet, Neumann).
 *
 * @tparam Real Floating-point type (e.g., float, double)
 */
template<typename Real>
class UniformBoundary : public BoundaryCondition<Real>{
public:

    /// Precomputed 1 / (delta^2)
    const Real inverseDeltaSquared;

    /// Precomputed 1 / delta
    const Real inverseDelta;

    const bool isStaggered;
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
    __device__ __host__ UniformBoundary(Real value, Real delta, bool isNeumann, bool isStaggered) :
        BoundaryCondition<Real>(value, isNeumann),
        inverseDeltaSquared(1/(delta*delta)),
        inverseDelta(1/delta),
        isStaggered(isStaggered) {};

    /**
     * An empty conditon, similar to a null pointer.
     */
    __device__ __host__ UniformBoundary() :
        BoundaryCondition<Real>(0, false),
        inverseDeltaSquared(0),
        inverseDelta(0),
        isStaggered(false) {};

    /**
     * @brief Apply boundary condition contribution to system.
     *
     * Modifies the matrix L for a grid point adjacent to the boundary.
     *
     */
    __device__ void setL(Real& mainDiagVal, Real& offDiagVal) const {
        if (this-> isNeumann)
            mainDiagVal -= this->inverseDeltaSquared;
        else {
            if (isStaggered) mainDiagVal -= 3*this->inverseDeltaSquared;
            else mainDiagVal -= 2 * this->inverseDeltaSquared;
        }
        offDiagVal = this->inverseDeltaSquared;
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

        if (this->isNeumann) contribution = -this->value * this->inverseDelta;
        else {
            if (isStaggered) contribution = -2*this->value * this->inverseDeltaSquared;
            else contribution = -this->value * this->inverseDeltaSquared;
        }
        atomicAdd(&rhsVal, contribution);
    }

    __device__ __host__ bool isUndefined() const{
        return inverseDelta == 0;
    }

};

template<typename Real>
class BoundaryConditionHost {
public:
    Real value;
    bool isNeumann;

    BoundaryConditionHost(Real value, bool isNeumann)
        : value(value), isNeumann(isNeumann) {}

    // Convenience converter to the standard trivially copyable device layout
    BoundaryCondition<Real> forDevice() const {
        return BoundaryCondition<Real>(value, isNeumann);
    }
};

