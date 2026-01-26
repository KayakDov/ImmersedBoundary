#include <memory>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "immersedBoundary/ImerssedEquation.h"

/**
 * @file ImmersedEquationInterface.hpp
 * @brief Explicitly typed interface for Shroud compatibility.
 * * Functions are suffixed by types:
 * d = double, s = float (single)
 * i32 = int32_t, i64 = int64_t
 */

// Global instances for each type combination
template<typename Real, typename Int>
std::unique_ptr<ImmersedEq<Real, Int> > eq = nullptr;

template<typename Real, typename Int>
void initImmersedEq(size_t gridHeight, size_t gridWidth, size_t gridDepth, size_t forceSize, size_t nnzMaxB, Real *p, Real *f, double dx, double dy,
                    double dz, double tol, size_t iter) {
    eq<Real, Int> = std::make_unique<ImmersedEq<Real, Int> >(GridDim(gridHeight, gridWidth, gridDepth), forceSize, nnzMaxB, p, f, Real3d(dx, dy, dz),
                                                             tol, iter);
}

template<typename Real, typename Int>
void solveImmersedEq(Real *res, size_t nnzB, Int *rowPointersB, Int *colOffsetsB, Real *val, bool multi = true) {
    if (!eq<Real, Int>) throw std::runtime_error(
        "The solver is not initialized.  Be sure you're using consistent types.");
    eq<Real, Int>->solve(res, nnzB, rowPointersB, colOffsetsB, val, multi);
}

// --- Initialization Functions ---
extern "C" {
inline void initImmersedEq_d_i32(size_t gridHeight, size_t gridWidth, size_t gridDepth, size_t forceSize, size_t nnzMaxB, double *p, double *f, double dx, double dy, double dz, double tol, size_t iter) {
        initImmersedEq<double, int32_t>(gridHeight, gridWidth, gridDepth, forceSize, nnzMaxB, p, f, dx, dy, dz, tol, iter);
    }

inline void initImmersedEq_s_i32(size_t gridHeight, size_t gridWidth, size_t gridDepth, size_t forceSize, size_t nnzMaxB, float *p, float *f, double dx, double dy, double dz, double tol, size_t iter) {
        initImmersedEq<float, int32_t>(gridHeight, gridWidth, gridDepth, forceSize, nnzMaxB, p, f, dx, dy, dz, tol, iter);
    }

inline void initImmersedEq_d_i64(size_t gridHeight, size_t gridWidth, size_t gridDepth, size_t forceSize, size_t nnzMaxB, double *p, double *f, double dx, double dy, double dz, double tol, size_t iter) {
        initImmersedEq<double, int64_t>(gridHeight, gridWidth, gridDepth, forceSize, nnzMaxB, p, f, dx, dy, dz, tol, iter);
    }

inline void initImmersedEq_s_i64(size_t gridHeight, size_t gridWidth, size_t gridDepth, size_t forceSize, size_t nnzMaxB, float *p, float *f, double dx, double dy, double dz, double tol, size_t iter) {
        initImmersedEq<float, int64_t>(gridHeight, gridWidth, gridDepth, forceSize, nnzMaxB, p, f, dx, dy, dz, tol, iter);
    }

    // --- Solve Functions ---

inline void solveImmersedEq_d_i32(double *res, size_t nnzB, int32_t *rowPointersB, int32_t *colOffsetsB, double *val, bool multi = true) {
        solveImmersedEq<double, int32_t>(res, nnzB, rowPointersB, colOffsetsB, val, multi);
    }

inline void solveImmersedEq_s_i32(float *res, size_t nnzB, int32_t *rowPointersB, int32_t *colOffsetsB, float *val, bool multi = true) {
        solveImmersedEq<float, int32_t>(res, nnzB, rowPointersB, colOffsetsB, val, multi);
    }

inline void solveImmersedEq_d_i64(double *res, size_t nnzB, int64_t *rowPointers, int64_t *colOffsets, double *val, bool multi = true) {
        solveImmersedEq<double, int64_t>(res, nnzB, rowPointers, colOffsets, val, multi);
    }

inline void solveImmersedEq_s_i64(float *res, size_t nnzB, int64_t *rowPointers, int64_t *colOffsets, float *val, bool multi = true) {
        solveImmersedEq<float, int64_t>(res, nnzB, rowPointers, colOffsets, val, multi);
    }
}
