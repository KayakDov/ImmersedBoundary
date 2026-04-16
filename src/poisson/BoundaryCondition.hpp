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
     */
    __device__ void setL(Real& mainDiagVal, Real& offDiagVal) const {
        switch (condition) {
            case ConditionType::NeumannStaggered:
                offDiagVal = this->inverseDeltaSquared;
                mainDiagVal -= this->inverseDeltaSquared;
                break;
            case  ConditionType::DirichletStaggered:
                offDiagVal = this->inverseDeltaSquared;
                mainDiagVal -= 3*this->inverseDeltaSquared;
                break;
            case ConditionType::NeumannNodeCentered:
                offDiagVal = this->inverseDeltaSquared;
                mainDiagVal -= this->inverseDeltaSquared;
                break;
            case ConditionType::DirichletNodeCentered:
                offDiagVal = this->inverseDeltaSquared;
                mainDiagVal -= 2 * this->inverseDeltaSquared;
                break;
        }
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
    __device__ void setBoundaryRHSContribution(Real& rhsVal) const {
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
            case ConditionType::DirichletNodeCentered:
                contribution = -this->value * this->inverseDeltaSquared;
                break;
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
        setBoundaryRHSContribution(rhsVal);
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

    /**
     * @brief Retrieve a boundary condition by dimension and position.
     *
     * @param[in] dim               Dimension: 0=row/y, 1=col/x, 2=layer/z.
     * @param[in] isStart           If true, return the boundary at the start (left/top/front).
     *                              If false, return the boundary at the end (right/bottom/back).
     *
     * @return Reference to the requested boundary condition.
     *
     * @throws std::out_of_range if dim is not 0, 1, or 2.
     */
    __host__ __device__ const BoundaryCondition<Real>& operator()(size_t dim, bool isStart) const {
        switch (dim) {
            case 0: return isStart ? left : right;
            case 1: return isStart ? top : bottom;
            case 2: return isStart ? front : back;
            default:
        #ifdef __CUDA_ARCH__
                asm("trap;");  // Device-side trap for invalid access
                return left;   // Unreachable, but satisfies return requirement
        #else
                throw std::out_of_range("Invalid dimension: must be 0, 1, or 2");
        #endif
        }

    }

};