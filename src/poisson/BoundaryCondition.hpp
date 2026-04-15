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
#include "deviceArrays/headers/DeviceData.cuh"

enum class ConditionType{DirichletStaggered, NeumannStaggered, DirichletNodeCentered, NeumannNodeCentered};

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

    const ConditionType condition;

    /**
     * @brief Construct a boundary condition.
     *
     * @param value Boundary value:
     *              - Dirichlet: prescribed field value
     *              - Neumann: prescribed normal derivative
     * @param delta Grid spacing in the relevant direction
     * @param condition The type of the condition.
     */
    __device__ __host__ BoundaryCondition(Real value, Real delta, ConditionType condition) : value(value), inverseDeltaSquared(1/(delta*delta)), inverseDelta(1/delta), condition(condition) {};

    /**
     * @brief Apply boundary condition contribution to system.
     *
     * Modifies the matrix L and RHS vector b for a grid point adjacent to the boundary.
     *
     * @param L Discrete Laplacian matrix
     * @param gridIndFlattened Index of the current grid point
     * @param diagOffset Index offset to neighbor in the boundary-normal direction
     * @param rhs Right-hand side vector
     */
    __device__ void setLAndRHS(DeviceData2d<Real> L,
                     const size_t gridIndFlattened,
                     const size_t primaryDiagonalCol,
                     const size_t secondaryDiagonalCol,
                     const DeviceData1d<Real> rhs
    ) const {
        switch (condition) {
            case ConditionType::NeumannStaggered:
                L(gridIndFlattened, secondaryDiagonalCol) = this->inverseDeltaSquared;
                L(gridIndFlattened, primaryDiagonalCol) -= this->inverseDeltaSquared;
                break;
            case  ConditionType::DirichletStaggered:
                L(gridIndFlattened, secondaryDiagonalCol) = this->inverseDeltaSquared;
                L(gridIndFlattened, primaryDiagonalCol) -= 3*this->inverseDeltaSquared;
                break;
            case ConditionType::NeumannNodeCentered:
                L(gridIndFlattened, secondaryDiagonalCol) = this->inverseDeltaSquared;
                L(gridIndFlattened, primaryDiagonalCol) -= this->inverseDeltaSquared;
                break;
            case ConditionType::DirichletNodeCentered:
                L(gridIndFlattened, secondaryDiagonalCol) = this->inverseDeltaSquared;
                L(gridIndFlattened, primaryDiagonalCol) -= 2 * this->inverseDeltaSquared;
                break;
        }
        setBoundaryRHSContribution(gridIndFlattened, rhs);
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
    __device__ void setBoundaryRHSContribution(const size_t gridIndFlattened, DeviceData1d<Real> rhs) const {
        Real contribution = 0;

        switch (condition) {
            case ConditionType::NeumannStaggered:
                contribution = -this->value * this->inverseDelta;
                break;
            case ConditionType::DirichletStaggered:
                contribution = 2*this->value * this->inverseDeltaSquared;
                break;
            case ConditionType::NeumannNodeCentered:
                contribution = this->value * this->inverseDelta;
                break;
            case ConditionType::DirichletNodeCentered:
                contribution = -this->value * this->inverseDeltaSquared;
                break;
        }

        atomicAdd(&rhs[gridIndFlattened], contribution);
    }

};


/**
 * @struct BoundaryConfig
 * @brief Stores boundary condition configuration for a 3D domain.
 *
 * Holds pointers to boundary condition objects for each face of the domain.
 * Faces can be null if not applicable (e.g., in 2D problems).
 */
template<typename Real>
struct BoundaryConfig {

    /// Boundary conditions for each face
    const BoundaryCondition<Real> left, right, top, bottom, front, back;

    /**
     * @brief Construct boundary configuration.
     *
     * @param left   Boundary at minimum x
     * @param right  Boundary at maximum x
     * @param top    Boundary at maximum y
     * @param bottom Boundary at minimum y
     * @param front  Boundary at minimum z (optional)
     * @param back   Boundary at maximum z (optional)
     */
    __host__ __device__ BoundaryConfig(
        const BoundaryCondition<Real>& left, const BoundaryCondition<Real>& right,
        const BoundaryCondition<Real>& top, const BoundaryCondition<Real>& bottom,
        const BoundaryCondition<Real>& front, const BoundaryCondition<Real>& back
        ) : left(left), right(right), top(top), bottom(bottom), front(front), back(back){};

    __host__ BoundaryConfig(ConditionType type, Real value, Real3d delta):
        left(value, delta.x, type), right(value, delta.x, type),
        top(value, delta.y, type), bottom(value, delta.y, type),
        front(value, delta.z, type), back(value, delta.z, type){}

    __host__ BoundaryConfig(ConditionType type, Real value, Real delta): BoundaryConfig(type, value, Real3d(delta, delta, delta)){}

};