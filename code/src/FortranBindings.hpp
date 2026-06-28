#include <memory>
#include <cstddef>
#include <cstdint>
#include <stdexcept>



#include "solvers/EigenDecomp/EigenDecompForFortran.h"
#include "immersedBoundary/ImerssedEquation.h"

/**
 * @file ImmersedEquationInterface.hpp
 * @brief Explicitly typed interface for Shroud compatibility.
 * Functions are suffixed by types:
 * d = double, s = float (single)
 * i32 = int32_t, i64 = int64_t
 * @ingroup public_api
 */

//TODO: export a synchronization method into the wrappe and remove synchronization from methods to give caller control over synchronization.
//TODO: All of these methods currently

namespace ImEq {

    template<typename Real, typename Int>
    std::unique_ptr<ImmersedEq<Real, Int>> eq = nullptr;

    template<typename Real, typename Int>
    void initImmersedEq(
        size_t gridHeight, size_t gridWidth, size_t gridDepth,
        bool leftIsNeumann, bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool backIsNeumann, bool frontIsNeumann,
        Real leftVal, Real rightVal, Real topVal, Real bottomVal, Real frontVal, Real backVal,
        bool isStaggered,
        size_t forceSize,
        size_t nnzMax,
        Real *p, Real *f,
        Real* dx, Real* dy, Real* dz, double dt,
        bool uniformDeltaX, bool uniformDeltaY, bool uniformDeltaZ,
        double tol, size_t maxIterations
    ) {
        XYZ<std::vector<Real>> delta(
            std::vector<Real>(dx, dx + (uniformDeltaX ? 1 : gridWidth + 1)),
            std::vector<Real>(dy, dy + (uniformDeltaY ? 1 : gridHeight + 1)),
            std::vector<Real>(dz, dz + (uniformDeltaZ ? 1 : gridDepth + 1))
        );

        buildBoundaryConfigAndLaunch(
            GridDim(gridHeight, gridWidth, gridDepth),
            delta,
            XYZ<bool>(leftIsNeumann, topIsNeumann, frontIsNeumann),
            XYZ<bool>(rightIsNeumann, bottomIsNeumann, backIsNeumann),
            XYZ<Real>(leftVal, topVal, frontVal),
            XYZ<Real>(rightVal, bottomVal, backVal),
            isStaggered,
            0,
            [&](const auto& boundary) {
                    eq<Real, Int> = std::make_unique<ImmersedEq<Real, Int>>(boundary, forceSize, nnzMax, p, f, dt, tol, maxIterations);
                }
        );

        //The old code.
        // BoundaryConfig<Real> boundary(
        //     {leftIsNeumann, topIsNeumann, frontIsNeumann},
        //     {rightIsNeumann, bottomIsNeumann, backIsNeumann},
        //     {leftVal, topVal, frontVal}, {rightVal, bottomVal, backVal},
        //     delta,
        //     GridDim(gridHeight, gridWidth, gridDepth),
        //     isStaggered
        // );
        // eq<Real, Int> = std::make_unique<ImmersedEq<Real, Int> >(boundary, forceSize, nnzMax, p, f, delta, dt, tol, maxIterations);
    }

    template<typename Real, typename Int>
    void solveImmersedEq(Real *result, size_t nnzB, Int *rowOffsetsB, Int *colIndsB, Real *valB) {
        if (!eq<Real, Int>) throw std::runtime_error("The solver is not initialized.  Be sure you're using consistent types.");
        eq<Real, Int>->solve(result, nnzB, rowOffsetsB, colIndsB, valB);
        cudaDeviceSynchronize();
    }

    template<typename Real, typename Int>
    void solveImmersedEq(Real* resultPPrime, Real* resultFPrime, size_t nnzB, Int *rowOffsetsB, Int *colIndsB, Real *valuesB, size_t nnzR, Int *colOffsetsR, Int *rowIndsR, Real *valuesR, Real *UGamma, Real* uStar) {
        if (!eq<Real, Int>) throw std::runtime_error("The solver is not initialized.  Be sure you're using consistent types.");
        eq<Real, Int>->solve(resultPPrime, resultFPrime, nnzB, rowOffsetsB, colIndsB, valuesB,nnzR, colOffsetsR, rowIndsR, valuesR, UGamma, uStar);
        cudaDeviceSynchronize();
    }

    template<typename Real, typename Int>
    void finalizeImmersedEq() {
        if (eq<Real, Int>) eq<Real, Int>.reset();
    }

    extern "C" {
        inline void initImmersedEq_d_i32(
            size_t gridHeight, size_t gridWidth, size_t gridDepth,
            bool leftIsNeumann, bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool backIsNeumann, bool frontIsNeumann,
            double leftVal, double rightVal, double topVal, double bottomVal, double frontVal, double backVal,
            bool isStaggered,
            size_t forceSize,
            size_t nnzMax,
            double *p, double *f,
            double* dx, double* dy, double* dz, double dt,
            bool uniformDeltaX, bool uniformDeltaY, bool uniformDeltaZ,
            double tol, size_t maxIterations
        ) {
            initImmersedEq<double, int32_t>(
                gridHeight, gridWidth, gridDepth,
                leftIsNeumann, rightIsNeumann, topIsNeumann, bottomIsNeumann, backIsNeumann, frontIsNeumann,
                leftVal, rightVal, topVal, bottomVal, frontVal, backVal,
                isStaggered,
                forceSize,
                nnzMax,
                p, f,
                dx, dy, dz, dt,
                uniformDeltaX, uniformDeltaY, uniformDeltaZ,
                tol, maxIterations
            );
        }

        inline void initImmersedEq_s_i32(
            size_t gridHeight, size_t gridWidth, size_t gridDepth,
            bool leftIsNeumann, bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool backIsNeumann, bool frontIsNeumann,
            double leftVal, double rightVal, double topVal, double bottomVal, double frontVal, double backVal,
            bool isStaggered,
            size_t forceSize,
            size_t nnzMax,
            float *p, float *f,
            float* dx, float* dy, float* dz, double dt,
            bool uniformDeltaX, bool uniformDeltaY, bool uniformDeltaZ,
            double tol, size_t maxIterations
        ) {
            initImmersedEq<float, int32_t>(
                gridHeight, gridWidth, gridDepth,
                leftIsNeumann, rightIsNeumann, topIsNeumann, bottomIsNeumann, backIsNeumann, frontIsNeumann,
                leftVal, rightVal, topVal, bottomVal, frontVal, backVal,
                isStaggered,
                forceSize,
                nnzMax,
                p, f,
                dx, dy, dz, dt,
                uniformDeltaX, uniformDeltaY, uniformDeltaZ,
                tol, maxIterations
            );
        }

        inline void initImmersedEq_d_i64(
            size_t gridHeight, size_t gridWidth, size_t gridDepth,
            bool leftIsNeumann, bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool backIsNeumann, bool frontIsNeumann,
            double leftVal, double rightVal, double topVal, double bottomVal, double frontVal, double backVal,
            bool isStaggered,
            size_t forceSize,
            size_t nnzMax,
            double *p, double *f,
            double* dx, double* dy, double* dz, double dt,
            bool uniformDeltaX, bool uniformDeltaY, bool uniformDeltaZ,
            double tol, size_t maxIterations
        ) {
            initImmersedEq<double, int64_t>(
                gridHeight, gridWidth, gridDepth,
                leftIsNeumann, rightIsNeumann, topIsNeumann, bottomIsNeumann, backIsNeumann, frontIsNeumann,
                leftVal, rightVal, topVal, bottomVal, frontVal, backVal,
                isStaggered,
                forceSize,
                nnzMax,
                p, f,
                dx, dy, dz, dt,
                uniformDeltaX, uniformDeltaY, uniformDeltaZ,
                tol, maxIterations
            );
        }

        inline void initImmersedEq_s_i64(
            size_t gridHeight, size_t gridWidth, size_t gridDepth,
            bool leftIsNeumann, bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool backIsNeumann, bool frontIsNeumann,
            double leftVal, double rightVal, double topVal, double bottomVal, double frontVal, double backVal,
            bool isStaggered,
            size_t forceSize,
            size_t nnzMax,
            float *p, float *f,
            float* dx, float* dy, float* dz, double dt,
            bool uniformDeltaX, bool uniformDeltaY, bool uniformDeltaZ,
            double tol, size_t maxIterations
        ) {
            initImmersedEq<float, int64_t>(
                gridHeight, gridWidth, gridDepth,
                leftIsNeumann, rightIsNeumann, topIsNeumann, bottomIsNeumann, backIsNeumann, frontIsNeumann,
                leftVal, rightVal, topVal, bottomVal, frontVal, backVal,
                isStaggered,
                forceSize,
                nnzMax,
                p, f,
                dx, dy, dz, dt,
                uniformDeltaX, uniformDeltaY, uniformDeltaZ,
                tol, maxIterations
            );
        }

        inline void solveImmersedEq_d_i32(double *result, size_t nnzB, int32_t *rowOffsetsB, int32_t *colIndsB, double *val, bool multi = true) {
            solveImmersedEq<double, int32_t>(result, nnzB, rowOffsetsB, colIndsB, val);
        }

        inline void solveImmersedEq_s_i32(float *result, size_t nnzB, int32_t *rowOffsetsB, int32_t *colIndsB, float *val, bool multi = true) {
            solveImmersedEq<float, int32_t>(result, nnzB, rowOffsetsB, colIndsB, val);
        }

        inline void solveImmersedEq_d_i64(double *result, size_t nnzB, int64_t *rowOffsetsB, int64_t *colIndsB, double *val, bool multi = true) {
            solveImmersedEq<double, int64_t>(result, nnzB, rowOffsetsB, colIndsB, val);
        }

        inline void solveImmersedEq_s_i64(float *result, size_t nnzB, int64_t *rowOffsetsB, int64_t *colIndsB, float *val, bool multi = true) {
            solveImmersedEq<float, int64_t>(result, nnzB, rowOffsetsB, colIndsB, val);
        }

        inline void solveImmersedEqPrimes_d_i32(double* resultPPrime, double* resultFPrime, size_t nnzB, int32_t *rowOffsetsB, int32_t *colIndsB, double *valuesB, size_t nnzR, int32_t *colOffsetsR, int32_t *rowIndsR, double *valuesR, double *UGamma, double* uStar) {
            solveImmersedEq<double, int32_t>(resultPPrime, resultFPrime, nnzB, rowOffsetsB, colIndsB, valuesB, nnzR, colOffsetsR, rowIndsR, valuesR, UGamma, uStar);
        }

        inline void solveImmersedEqPrimes_s_i32(float* resultPPrime, float* resultFPrime, size_t nnzB, int32_t *rowOffsetsB, int32_t *colIndsB, float *valuesB, size_t nnzR, int32_t *colOffsetsR, int32_t *rowIndsR, float *valuesR, float *UGamma, float* uStar) {
            solveImmersedEq<float, int32_t>(resultPPrime, resultFPrime, nnzB, rowOffsetsB, colIndsB, valuesB, nnzR, colOffsetsR, rowIndsR, valuesR, UGamma, uStar);
        }

        inline void solveImmersedEqPrimes_d_i64(double* resultPPrime, double* resultFPrime, size_t nnzB, int64_t *rowOffsetsB, int64_t *colIndsB, double *valuesB, size_t nnzR, int64_t *colOffsetsR, int64_t *rowIndsR, double *valuesR, double *UGamma, double* uStar) {
            solveImmersedEq<double, int64_t>(resultPPrime, resultFPrime, nnzB, rowOffsetsB, colIndsB, valuesB, nnzR, colOffsetsR, rowIndsR, valuesR, UGamma, uStar);
        }

        inline void solveImmersedEqPrimes_s_i64(float* resultPPrime, float* resultFPrime, size_t nnzB, int64_t *rowOffsetsB, int64_t *colIndsB, float *valuesB, size_t nnzR, int64_t *colOffsetsR, int64_t *rowIndsR, float *valuesR, float *UGamma, float* uStar) {
            solveImmersedEq<float, int64_t>(resultPPrime, resultFPrime, nnzB, rowOffsetsB, colIndsB, valuesB, nnzR, colOffsetsR, rowIndsR, valuesR, UGamma, uStar);
        }

        inline void finalizeImmersedEq_d_i32() {
            finalizeImmersedEq<double, int32_t>();
        }

        inline void finalizeImmersedEq_s_i32() {
            finalizeImmersedEq<float, int32_t>();
        }

        inline void finalizeImmersedEq_d_i64() {
            finalizeImmersedEq<double, int64_t>();
        }

        inline void finalizeImmersedEq_s_i64() {
            finalizeImmersedEq<float, int64_t>();
        }
    }
}

namespace eigen {

    template<typename Real>
    std::unique_ptr<EigenDecompForFortran<Real>> eds = nullptr;

    template<typename Real>
    void initEigenDecompSolver(
        size_t rows, size_t cols, size_t layers,
        Real* dx, Real* dy, Real* dz,
        bool uniformDeltaX, bool uniformDeltaY, bool uniformDeltaZ,
        bool leftIsNeumann, bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool backIsNeumann, bool frontIsNeumann,
        Real leftVal, Real rightVal, Real topVal, Real bottomVal, Real frontVal, Real backVal,
        bool isStaggered,
        bool thomas
    ) {
        auto xb = Mat<Real>::create(rows * cols * layers, 3);



        eds<Real> = std::make_unique<EigenDecompForFortran<Real>>(
            rows, cols, layers,
            std::vector<Real>(dx, dx + (uniformDeltaX ? 1 : cols + 1)),
            std::vector<Real>(dy, dy + (uniformDeltaY ? 1 : rows + 1)),
            std::vector<Real>(dz, dz + (uniformDeltaZ ? 1 : layers + 1)),
            leftIsNeumann, rightIsNeumann, topIsNeumann, bottomIsNeumann, frontIsNeumann, backIsNeumann,
            leftVal, rightVal, topVal, bottomVal, frontVal, backVal,
            isStaggered,
            thomas, xb.col(0), xb.col(1), xb.col(2));
    }

    template<typename Real>
    void runDecompSolver(Real* xHost, Real* bHost) {
        if (!eds<Real>) throw std::runtime_error(
            "The solver is not initialized.  Be sure you're using consistent types.");
        eds<Real>->solve(xHost, bHost);
        cudaDeviceSynchronize();
    }

    template<typename Real>
    void finalizeEigenDecomp() {
        if (eds<Real>) eds<Real>.reset();
    }

    // --- Initialization Functions ---
    extern "C" {
        inline void initEigenDecomp_d(
            size_t rows, size_t cols, size_t layers,
            double* dx, double* dy, double* dz,
            bool uniformDeltaX, bool uniformDeltaY, bool uniformDeltaZ,
            bool leftIsNeumann, bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool backIsNeumann, bool frontIsNeumann,
            double leftVal, double rightVal, double topVal, double bottomVal, double frontVal, double backVal,
            bool isStaggered,
            bool thomas
        ) {
            initEigenDecompSolver<double>(
                rows, cols, layers,
                dx, dy, dz,
                uniformDeltaX, uniformDeltaX, uniformDeltaZ,
                leftIsNeumann, rightIsNeumann, topIsNeumann, bottomIsNeumann, frontIsNeumann, backIsNeumann,
                leftVal, rightVal, topVal, bottomVal, frontVal, backVal,
                isStaggered, thomas);
        }

        inline void initEigenDecomp_s(
            size_t rows, size_t cols, size_t layers,
            float* dx, float* dy, float* dz,
            bool uniformDeltaX, bool uniformDeltaY, bool uniformDeltaZ,
            bool leftIsNeumann, bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool backIsNeumann, bool frontIsNeumann,
            float leftVal, float rightVal, float topVal, float bottomVal, float frontVal, float backVal,
            bool isStaggered,
            bool thomas
        ) {
            initEigenDecompSolver<float>(
                rows, cols, layers,
                dx, dy, dz,
                uniformDeltaX, uniformDeltaX, uniformDeltaZ,
                leftIsNeumann, rightIsNeumann, topIsNeumann, bottomIsNeumann, frontIsNeumann, backIsNeumann,
                leftVal, rightVal, topVal, bottomVal, frontVal, backVal,
                isStaggered, thomas
            );
        }

        inline void solveEigenDecomp_d(double *x, double* b) {
            runDecompSolver(x, b);
        }

        inline void solveEigenDecomp_s(float *x, float* b) {
            runDecompSolver(x, b);
        }

        inline void finalizeEigenDecomp_d() {
            finalizeEigenDecomp<double>();
        }

        inline void finalizeEigenDecomp_s() {
            finalizeEigenDecomp<float>();
        }
    }
}