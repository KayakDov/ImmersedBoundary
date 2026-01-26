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
void initImmersedEq(size_t h, size_t w, size_t d, size_t fs, size_t nnzMax, Real *p, Real *f, double dx, double dy,
                    double dz, double tol, size_t iter) {
    eq<Real, Int> = std::make_unique<ImmersedEq<Real, Int> >(GridDim(h, w, d), fs, nnzMax, p, f, Real3d(dx, dy, dz),
                                                             tol, iter);
}

template<typename Real, typename Int>
void solveImmersedEq(Real *res, size_t nnz, Int *rp, Int *cp, Real *val, bool multi = true) {
    if (!eq<Real, Int>) throw std::runtime_error(
        "The solver is not initialized.  Be sure you're using consistent types.");
    eq<Real, Int>->solve(res, nnz, rp, cp, val, multi);
}

// --- Initialization Functions ---
extern "C" {
inline void initImmersedEq_d_i32(size_t h, size_t w, size_t d, size_t fs, size_t nnz, double *p, double *f, double dx,
                                 double dy, double dz, double tol, size_t iter) {
        initImmersedEq<double, int32_t>(h, w, d, fs, nnz, p, f, dx, dy, dz, tol, iter);
    }

inline void initImmersedEq_s_i32(size_t h, size_t w, size_t d, size_t fs, size_t nnz, float *p, float *f, double dx, double dy,
                                 double dz, double tol, size_t iter) {
        initImmersedEq<float, int32_t>(h, w, d, fs, nnz, p, f, dx, dy, dz, tol, iter);
    }

inline void initImmersedEq_d_i64(size_t h, size_t w, size_t d, size_t fs, size_t nnz, double *p, double *f, double dx,
                                 double dy, double dz, double tol, size_t iter) {
        initImmersedEq<double, int64_t>(h, w, d, fs, nnz, p, f, dx, dy, dz, tol, iter);
    }

inline void initImmersedEq_s_i64(size_t h, size_t w, size_t d, size_t fs, size_t nnz, float *p, float *f, double dx, double dy,
                                 double dz, double tol, size_t iter) {
        initImmersedEq<float, int64_t>(h, w, d, fs, nnz, p, f, dx, dy, dz, tol, iter);
    }

    // --- Solve Functions ---

inline void solveImmersedEq_d_i32(double *res, size_t nnz, int32_t *rp, int32_t *cp, double *val, bool multi = true) {
        solveImmersedEq<double, int32_t>(res, nnz, rp, cp, val, multi);
    }

inline void solveImmersedEq_s_i32(float *res, size_t nnz, int32_t *rp, int32_t *cp, float *val, bool multi = true) {
        solveImmersedEq<float, int32_t>(res, nnz, rp, cp, val, multi);
    }

inline void solveImmersedEq_d_i64(double *res, size_t nnz, int64_t *rp, int64_t *cp, double *val, bool multi = true) {
        solveImmersedEq<double, int64_t>(res, nnz, rp, cp, val, multi);
    }

inline void solveImmersedEq_s_i64(float *res, size_t nnz, int64_t *rp, int64_t *cp, float *val, bool multi = true) {
        solveImmersedEq<float, int64_t>(res, nnz, rp, cp, val, multi);
    }
}
