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
#include <set>

#include "deviceArrays/headers/DeviceData.cuh"
#include "solvers/Event.h"
#include <limits>

/**
 * Dirchlet is fixed value while Neumann is fixed gradient.
 */
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
     * An empty conditon, similar to a null pointer.
     */
    __device__ __host__ BoundaryCondition() : value(std::numeric_limits<Real>::quiet_NaN()), inverseDeltaSquared(0), inverseDelta(0), condition(ConditionType::NeumannNodeCentered) {};

    /**
     * @brief Apply boundary condition contribution to system.
     *
     * Modifies the matrix L for a grid point adjacent to the boundary.
     *
     */
    __device__ void setL(Real& mainDiagVal, Real& offDiagVal) const {
        switch (condition) {
            case ConditionType::NeumannNodeCentered:
            case ConditionType::NeumannStaggered:
                mainDiagVal -= this->inverseDeltaSquared;
                break;
            case  ConditionType::DirichletStaggered:
                mainDiagVal -= 3*this->inverseDeltaSquared;
                break;
            case ConditionType::DirichletNodeCentered:
                mainDiagVal -= 2 * this->inverseDeltaSquared;
                break;
        }
        offDiagVal = this->inverseDeltaSquared;
    }
    /**
    * @brief Equality operator implemented as a hidden friend.
    * Compares all precomputed factors and the condition type.
    */
    __host__ __device__ friend bool operator==(const BoundaryCondition& lhs, const BoundaryCondition& rhs) {
        return (lhs.condition == rhs.condition) &&
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
    __device__ __host__ bool isUndefined() {
        return inverseDelta == 0;
    }
};

/**
 * @class BoundaryConditionPair
 * @brief Encapsulates the boundary conditions for both ends of a 1D grid segment.
 *
 * This class stores the physical and numerical constraints for the start and end
 * of a dimension (e.g., Left/Right, Top/Bottom). By storing these by value,
 * the pair maintains its own valid copy of the conditions, preventing dangling
 * references during kernel execution.
 * * @tparam T The floating-point type (float or double).
 */
template<typename T>
class BoundaryConditionPair {
public:
    /** @brief The condition applied at the start (lower index) of the segment. */
    const BoundaryCondition<T> start;

    /** @brief The condition applied at the end (higher index) of the segment. */
    const BoundaryCondition<T> end;

    /** @brief The length of the dimension.*/
    const size_t dimLength;

    __host__ __device__ BoundaryConditionPair() : dimLength(static_cast<size_t>(-1)){}

    /**
     * @brief Construct from existing BoundaryCondition objects.
     * @param start Condition for the beginning of the segment.
     * @param end Condition for the end of the segment.
     */
    __device__ __host__ BoundaryConditionPair(const BoundaryCondition<T>& start, const BoundaryCondition<T>& end, size_t dimLength)
        : start(start), end(end), dimLength(dimLength) {}

    /**
     * @brief Construct by providing raw parameters for both boundaries.
     * * This constructor initializes the internal BoundaryCondition objects directly.
     *
     * @param valStart       Physical value at the start (e.g., Temperature or Flux).
     * @param valEnd         Physical value at the end.
     * @param startCondition Type of condition (Dirichlet/Neumann) for the start.
     * @param endCondition   Type of condition for the end.
     * @param delta          Grid spacing used to precompute gradient factors.
     */
    __device__ __host__ BoundaryConditionPair(const T valStart,
                                              const T valEnd,
                                              const ConditionType startCondition,
                                              const ConditionType endCondition,
                                              T delta, size_t dimLength)
        : start(valStart, delta, startCondition),
          end(valEnd, delta, endCondition), dimLength(dimLength) {}

    /**
     * @brief Equality operator for condition pairs.
     * Required for deduplication in spectral or multi-dimensional setups.
     */
    __host__ __device__ friend bool operator==(const BoundaryConditionPair& lhs,
                                               const BoundaryConditionPair& rhs) {
        return (lhs.start == rhs.start) && (lhs.end == rhs.end) && (lhs.dimLength == rhs.dimLength);
    }

    __host__ __device__ friend bool operator!=(const BoundaryConditionPair& lhs,
                                               const BoundaryConditionPair& rhs) {
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
    __device__ bool setL(T& mainDiagVal, T& leftDiagonal, T& rightDiagonal, const size_t indexInLine, const size_t lineLength) const {
        if (indexInLine == 0) start.setL(mainDiagVal, rightDiagonal);
        else if (indexInLine == lineLength - 1) end.setL(mainDiagVal, leftDiagonal);
        else return false;
        return true;
    }

    /**
     * @brief Apply boundary condition contribution to RHS only.
     *
     * Adds the boundary-condition-induced contribution to the right-hand side
     * vector, assuming the operator (L) is assembled independently.
     *
     * @param gridIndFlattened Index of the current grid point
     * @param rhs Right-hand side vector (modisfied in place)
     * @return true if a modification is made, false otherwise.
     */
    __device__ bool setBoundaryRHS1d(DeviceData1d<T>& rhs, const size_t indexInLine, const size_t lineLength) const {
        if (indexInLine == 0) start.setBoundaryRHS(rhs[0]);
        else if (indexInLine == lineLength - 1) end.setBoundaryRHS(rhs[lineLength - 1]);
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
    __device__ bool setLAndRHS(T& mainDiagVal, T& startDiagVal, T& endDiagonalVal, DeviceData1d<T> rhs, const size_t indexInLine, const size_t linelength) const {
        setL(mainDiagVal, startDiagVal, endDiagonalVal, indexInLine, linelength);
        return setBoundaryRHS1d(rhs, indexInLine, linelength);
    }

    /**
     *
     * @param cond The condition.
     * @return True if the condition is Neumann.  False otherwise.
     */
    __host__ static bool isNeumann(ConditionType cond) {
        return cond == ConditionType::NeumannNodeCentered || cond == ConditionType::NeumannStaggered;
    }
    /**
     *
     * @param cond The condition
     * @return True if the condition is node centered,false otherwise.
     */
    __host__ bool isNodeCentered() const {
        return start.condition == ConditionType::DirichletNodeCentered || start.condition == ConditionType::NeumannNodeCentered;
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
    const BoundaryConditionPair<Real> leftRight, topBottom, frontBack; //TODO: Can xyz be used here?  It's not currently implemented on the device.

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
        const BoundaryCondition<Real>& front, const BoundaryCondition<Real>& back,
        const GridDim& dim
        ) : leftRight(left, right, dim.cols), topBottom(top, bottom, dim.rows), frontBack(front, back, dim.layers){}

    __host__ BoundaryConfig(ConditionType type, Real value, Real3d delta, const GridDim& dim):
        BoundaryConfig(
            {value, static_cast<Real>(delta.x), type},
            {value, static_cast<Real>(delta.x), type},
            {value, static_cast<Real>(delta.y), type},
            {value, static_cast<Real>(delta.y), type},
            {value, static_cast<Real>(delta.z), type},
            {value, static_cast<Real>(delta.z), type},
            dim
        ){}

    __host__ BoundaryConfig(const ConditionType& type, Real value, Real delta, const GridDim& dim): BoundaryConfig(type, value, Real3d(delta, delta, delta), dim){}

    /**
     * @brief Retrieve a boundary condition by dimension and position.
     *
     * @param[in] dim               Dimension: 0=row/y, 1=col/x, 2=layer/z.
     * @param[in] isEnd           If true, return the boundary at the start (left/top/front).
     *                              If false, return the boundary at the end (right/bottom/back).
     *
     * @return Reference to the requested boundary condition.
     *
     * @throws std::out_of_range if dim is not 0, 1, or 2.
     */
    __host__ __device__ const BoundaryConditionPair<Real>& operator[](size_t dim) const {
        switch (dim) {
            case 0: return leftRight;
            case 1: return topBottom;
            case 2: return frontBack;
            default:
        #ifdef __CUDA_ARCH__
                asm("trap;");  // Device-side trap for invalid access
                return leftRight;   // Unreachable, but satisfies return requirement
        #else
                throw std::out_of_range("Invalid dimension: must be 0, 1, or 2");
        #endif
        }

    }

    /**
     * True if i is the first index that these conditions appear at.
     * @param i The index to be checked.
     * @return the value of the index repeated, or -1 if this is the first appearence.
     */
    __host__ int repeat(int i) const {
        for (size_t j = 0; j < i; ++j) if ((*this)[j] == (*this)[i]) return j;
        return -1;

    }

    /**
     * @brief Efficiently creates unique objects based on 1D Laplacian boundary conditions.
     *
     * This utility iterates through the X, Y, and Z dimensions of a Laplacian setup.
     * If the boundary conditions (dimensions, spacing, and types) for two dimensions
     * are identical, the function reuses the existing object by sharing ownership
     * via std::shared_ptr. This prevents redundant memory allocations and unnecessary
     * re-computation of Laplacian matrices or eigenvalue vectors.
     *
     * @tparam T          The floating-point type (e.g., float or double).
     * @tparam ResultType The type of object to be created (e.g., Mat<T> or Vec<T>).
     * @tparam BoundaryPairToResultType    A callable type (lambda or function) that returns ResultType by value.
     *
     * @param[in]  conds    A reference to an array of 3 LaplacianConditions (X, Y, Z).
     * @param[out] outputs  An array of 3 shared_ptrs to be populated with the results.
     * @param[in]  factory  A factory function: `ResultType factory(const LaplacianConditions<T>&)`.
     * * @note Because ResultType is wrapped in a shared_ptr, ResultType does not need
     * to be assignable, which is critical for classes with const data members.
     */
    template <typename ResultType>
    __host__ void createUnique(std::shared_ptr<ResultType> (&outputs)[3], std::function<ResultType(const BoundaryConditionPair<Real>&)> factory) const {
        for (size_t i = 0; i < 3; ++i) {
            int repeatInd = repeat(i);
            if (repeatInd == -1) outputs[i] = std::make_shared<ResultType>(factory((*this)[i]));
            else outputs[i] = outputs[repeatInd];
        }
    }


    /**
     * Generates, including memory allocation, eigen values and vectors.  The matrices pointed to, that are retruned,
     * hold the values in the last column, and the vectors in the first nxn cells.
     * @param hands Used to create the different vectors in parrallel.  The number of handles should be equal to the number of dimentisons.
     * @param events The number of events should be equal to the number of dimesnions.
     * @param preAllocatedForL_iX3
     * @return pointers to matrices containing the eigen values and vectors.
     */
    void generateEigen(Handle *hands, Event *events, std::shared_ptr<Mat<Real>> (&preAllocatedForL_iX3)[3]) const;
};