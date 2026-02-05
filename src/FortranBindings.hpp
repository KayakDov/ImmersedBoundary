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
void initImmersedEq(size_t gridHeight, size_t gridWidth, size_t gridDepth, size_t forceSize, size_t nnzMax, Real *p, Real *f, double dx, double dy, double dz, double dt, double tol, size_t iter) {
    eq<Real, Int> = std::make_unique<ImmersedEq<Real, Int> >(
        GridDim(gridHeight, gridWidth, gridDepth),
        forceSize,
        nnzMax,
        p,
        f,
        Real3d(dx, dy, dz),
        dt,
        tol,
        iter
    );
}

template<typename Real, typename Int>
void solveImmersedEq(Real *result, size_t nnzB, Int *rowOffsetsB, Int *colIndsB, Real *valB, bool multi = true) {
    if (!eq<Real, Int>) throw std::runtime_error(
        "The solver is not initialized.  Be sure you're using consistent types.");
    eq<Real, Int>->solve(result, nnzB, rowOffsetsB, colIndsB, valB, multi);
}

template<typename Real, typename Int>
void solveImmersedEq(
    Real* resultPPrime,
    Real* resultFPrime,
    size_t nnzB,
    Int *rowOffsetsB,
    Int *colIndsB,
    Real *valuesB,
    size_t nnzR,
    Int *colOffsetsR,
    Int *rowIndsR,
    Real *valuesR,
    Real *UGamma,
    Real* uStar,
    bool multiStream
) {
    if (!eq<Real, Int>) throw std::runtime_error(
        "The solver is not initialized.  Be sure you're using consistent types.");
    eq<Real, Int>->solve(resultPPrime, resultFPrime, nnzB, rowOffsetsB, colIndsB, valuesB,nnzR, colOffsetsR, rowIndsR, valuesR, UGamma, uStar, multiStream);
}

// void ImmersedEq<Real, Int>::solve(




// --- Initialization Functions ---
extern "C" {
inline void initImmersedEq_d_i32(size_t gridHeight, size_t gridWidth, size_t gridDepth, size_t forceSize, size_t nnzMaxB, double *p, double *f, double dx, double dy, double dz, double dt, double tol, size_t iter) {
        initImmersedEq<double, int32_t>(gridHeight, gridWidth, gridDepth, forceSize, nnzMaxB, p, f, dx, dy, dz, dt, tol, iter);
    }

inline void initImmersedEq_s_i32(size_t gridHeight, size_t gridWidth, size_t gridDepth, size_t forceSize, size_t nnzMaxB, float *p, float *f, double dx, double dy, double dz, double dt, double tol, size_t iter) {
        initImmersedEq<float, int32_t>(gridHeight, gridWidth, gridDepth, forceSize, nnzMaxB, p, f, dx, dy, dz, dt, tol, iter);
    }

inline void initImmersedEq_d_i64(size_t gridHeight, size_t gridWidth, size_t gridDepth, size_t forceSize, size_t nnzMaxB, double *p, double *f, double dx, double dy, double dz, double dt, double tol, size_t iter) {
        initImmersedEq<double, int64_t>(gridHeight, gridWidth, gridDepth, forceSize, nnzMaxB, p, f, dx, dy, dz, dt, tol, iter);
    }

inline void initImmersedEq_s_i64(size_t gridHeight, size_t gridWidth, size_t gridDepth, size_t forceSize, size_t nnzMaxB, float *p, float *f, double dx, double dy, double dz, double dt, double tol, size_t iter) {
        initImmersedEq<float, int64_t>(gridHeight, gridWidth, gridDepth, forceSize, nnzMaxB, p, f, dx, dy, dz, dt, tol, iter);
    }

    // --- Solve Functions ---

inline void solveImmersedEq_d_i32(double *result, size_t nnzB, int32_t *rowOffsetsB, int32_t *colIndsB, double *val, bool multi = true) {
        solveImmersedEq<double, int32_t>(result, nnzB, rowOffsetsB, colIndsB, val, multi);
    }

inline void solveImmersedEq_s_i32(float *result, size_t nnzB, int32_t *rowOffsetsB, int32_t *colIndsB, float *val, bool multi = true) {
        solveImmersedEq<float, int32_t>(result, nnzB, rowOffsetsB, colIndsB, val, multi);
    }

inline void solveImmersedEq_d_i64(double *result, size_t nnzB, int64_t *rowOffsetsB, int64_t *colIndsB, double *val, bool multi = true) {
        solveImmersedEq<double, int64_t>(result, nnzB, rowOffsetsB, colIndsB, val, multi);
    }

inline void solveImmersedEq_s_i64(float *result, size_t nnzB, int64_t *rowOffsetsB, int64_t *colIndsB, float *val, bool multi = true) {
        solveImmersedEq<float, int64_t>(result, nnzB, rowOffsetsB, colIndsB, val, multi);
    }

    //-----------prime functions -------------
inline void solveImmersedEqPrimes_d_i32(
    double* resultPPrime,
    double* resultFPrime,
    size_t nnzB,
    int32_t *rowOffsetsB,
    int32_t *colIndsB,
    double *valuesB,
    size_t nnzR,
    int32_t *colOffsetsR,
    int32_t *rowIndsR,
    double *valuesR,
    double *UGamma,
    double* uStar,
    bool multiStream
) {
    solveImmersedEq<double, int32_t>(resultPPrime, resultFPrime, nnzB, rowOffsetsB, colIndsB, valuesB, nnzR, colOffsetsR, rowIndsR, valuesR, UGamma, uStar, multiStream);
}

inline void solveImmersedEqPrimes_s_i32(
   float* resultPPrime,
   float* resultFPrime,
   size_t nnzB,
   int32_t *rowOffsetsB,
   int32_t *colIndsB,
   float *valuesB,
   size_t nnzR,
   int32_t *colOffsetsR,
   int32_t *rowIndsR,
   float *valuesR,
   float *UGamma,
   float* uStar,
   bool multiStream
   ) {
    solveImmersedEq<float, int32_t>(resultPPrime, resultFPrime, nnzB, rowOffsetsB, colIndsB, valuesB, nnzR, colOffsetsR, rowIndsR, valuesR, UGamma, uStar, multiStream);
}

inline void solveImmersedEqPrimes_d_i64(
    double* resultPPrime,
    double* resultFPrime,
    size_t nnzB,
    int64_t *rowOffsetsB,
    int64_t *colIndsB,
    double *valuesB,
    size_t nnzR,
    int64_t *colOffsetsR,
    int64_t *rowIndsR,
    double *valuesR,
    double *UGamma,
    double* uStar,
    bool multiStream
) {
    solveImmersedEq<double, int64_t>(resultPPrime, resultFPrime, nnzB, rowOffsetsB, colIndsB, valuesB, nnzR, colOffsetsR, rowIndsR, valuesR, UGamma, uStar, multiStream);
}

inline void solveImmersedEqPrimes_s_i64(
    float* resultPPrime,
    float* resultFPrime,
    size_t nnzB,
    int64_t *rowOffsetsB,
    int64_t *colIndsB,
    float *valuesB,
    size_t nnzR,
    int64_t *colOffsetsR,
    int64_t *rowIndsR,
    float *valuesR,
    float *UGamma,
    float* uStar,
    bool multiStream
) {
    solveImmersedEq<float, int64_t>(resultPPrime, resultFPrime, nnzB, rowOffsetsB, colIndsB, valuesB, nnzR, colOffsetsR, rowIndsR, valuesR, UGamma, uStar, multiStream);
}
}
