#pragma once

#include <atomic>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "wrapper/LaplOperatorType.h"

#include "immersedBoundary/ImerssedEquation.h"
#include "wrapper/EigenDecompForFortran.h"


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
        size_t height, size_t width, size_t depth,
        bool leftIsNeumann, bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool frontIsNeumann, bool backIsNeumann,
        Real leftVal, Real rightVal, Real topVal, Real bottomVal, Real frontVal, Real backVal,
        size_t xSegSpacing, size_t ySegSpacing, size_t zSegSpacing,
        size_t forceSize,
        size_t nnzMax,
        Real *p, Real *f,
        Real* dx, Real* dy, Real* dz, double dt,
        bool xUniformDelta, bool yUniformDelta, bool zUniformDelta,
        double tol, size_t maxIterations
    ) {
        XYZ<std::vector<Real>> delta(
            std::vector<Real>(dx, dx + (xUniformDelta ? 1 : width + 1)),
            std::vector<Real>(dy, dy + (yUniformDelta ? 1 : height + 1)),
            std::vector<Real>(dz, dz + (zUniformDelta ? 1 : depth + 1))
        );

        Handle hand;
        buildBoundaryConfigAndLaunch(
            GridDim(height, width, depth),
            delta,
            XYZ<bool>(leftIsNeumann, topIsNeumann, frontIsNeumann),
            XYZ<bool>(rightIsNeumann, bottomIsNeumann, backIsNeumann),
            XYZ<Real>(leftVal, topVal, frontVal),
            XYZ<Real>(rightVal, bottomVal, backVal),
            XYZ<eigen::LaplOperatorT>(
                static_cast<eigen::LaplOperatorT>(xSegSpacing),
                static_cast<eigen::LaplOperatorT>(ySegSpacing),
                static_cast<eigen::LaplOperatorT>(zSegSpacing)
            ),
            hand,
            [&](const auto& boundary) {
                    eq<Real, Int> = std::make_unique<ImmersedEq<Real, Int>>(boundary, forceSize, nnzMax, p, f, dt, tol, maxIterations);
                }
        );
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
            size_t dim1Length, size_t dim2Length, size_t dim3Length,
            bool dim1StartIsNeumann, bool dim1EndIsNeumann, bool dim2StartIsNeumann, bool dim2EndIsNeumann, bool dim3StartIsNeumann, bool dim3EndIsNeumann,
            double dim1StartVal, double dim1EndVal, double dim2StartVal, double dim2EndVal, double dim3StartVal, double dim3EndVal,
            size_t dim1SegSpacing, size_t dim2SegSpacing, size_t dim3SegSpacing,
            size_t forceSize,
            size_t nnzMax,
            double *p, double *f,
            double* dim1Delta, double* dim2Delta, double* dim3Delta, double dt,
            bool dim1UniformDelta, bool dim2UniformDelta, bool dim3UniformDelta,
            double tol, size_t maxIterations
        ) {
            initImmersedEq<double, int32_t>(
                dim1Length, dim3Length, dim2Length,
                dim3StartIsNeumann, dim3EndIsNeumann, dim1StartIsNeumann, dim1EndIsNeumann, dim2StartIsNeumann, dim2EndIsNeumann,
                dim3StartVal, dim3EndVal, dim1StartVal, dim1EndVal, dim2StartVal, dim2EndVal,
                dim3SegSpacing, dim1SegSpacing, dim2SegSpacing,
                forceSize,
                nnzMax,
                p, f,
                dim3Delta, dim1Delta, dim2Delta, dt,
                dim3UniformDelta, dim1UniformDelta, dim2UniformDelta,
                tol, maxIterations
            );
        }

        inline void initImmersedEq_s_i32(
            size_t dim1Length, size_t dim2Length, size_t dim3Length,
            bool dim1StartIsNeumann, bool dim1EndIsNeumann, bool dim2StartIsNeumann, bool dim2EndIsNeumann, bool dim3StartIsNeumann, bool dim3EndIsNeumann,
            double dim1StartVal, double dim1EndVal, double dim2StartVal, double dim2EndVal, double dim3StartVal, double dim3EndVal,
            size_t dim1SegSpacing, size_t dim2SegSpacing, size_t dim3SegSpacing,
            size_t forceSize,
            size_t nnzMax,
            float *p, float *f,
            float* dim1Delta, float* dim2Delta, float* dim3Delta, double dt,
            bool dim1UniformDelta, bool dim2UniformDelta, bool dim3UniformDelta,
            double tol, size_t maxIterations
        ) {
            initImmersedEq<float, int32_t>(
                dim1Length, dim3Length, dim2Length,
                dim3StartIsNeumann, dim3EndIsNeumann, dim1StartIsNeumann, dim1EndIsNeumann, dim2StartIsNeumann, dim2EndIsNeumann,
                dim3StartVal, dim3EndVal, dim1StartVal, dim1EndVal, dim2StartVal, dim2EndVal,
                dim3SegSpacing, dim1SegSpacing, dim2SegSpacing,
                forceSize,
                nnzMax,
                p, f,
                dim3Delta, dim1Delta, dim2Delta, dt,
                dim3UniformDelta, dim1UniformDelta, dim2UniformDelta,
                tol, maxIterations
            );
        }

        inline void initImmersedEq_d_i64(
            size_t dim1Length, size_t dim2Length, size_t dim3Length,
            bool dim1StartIsNeumann, bool dim1EndIsNeumann, bool dim2StartIsNeumann, bool dim2EndIsNeumann, bool dim3StartIsNeumann, bool dim3EndIsNeumann,
            double dim1StartVal, double dim1EndVal, double dim2StartVal, double dim2EndVal, double dim3StartVal, double dim3EndVal,
            size_t dim1SegSpacing, size_t dim2SegSpacing, size_t dim3SegSpacing,
            size_t forceSize,
            size_t nnzMax,
            double *p, double *f,
            double* dim1Delta, double* dim2Delta, double* dim3Delta, double dt,
            bool dim1UniformDelta, bool dim2UniformDelta, bool dim3UniformDelta,
            double tol, size_t maxIterations
        ) {
            initImmersedEq<double, int64_t>(
                dim1Length, dim3Length, dim2Length,
                dim3StartIsNeumann, dim3EndIsNeumann, dim1StartIsNeumann, dim1EndIsNeumann, dim2StartIsNeumann, dim2EndIsNeumann,
                dim3StartVal, dim3EndVal, dim1StartVal, dim1EndVal, dim2StartVal, dim2EndVal,
                dim3SegSpacing, dim1SegSpacing, dim2SegSpacing,
                forceSize,
                nnzMax,
                p, f,
                dim3Delta, dim1Delta, dim2Delta, dt,
                dim3UniformDelta, dim1UniformDelta, dim2UniformDelta,
                tol, maxIterations
            );
        }

        inline void initImmersedEq_s_i64(
            size_t dim1Length, size_t dim2Length, size_t dim3Length,
            bool dim1StartIsNeumann, bool dim1EndIsNeumann, bool dim2StartIsNeumann, bool dim2EndIsNeumann, bool dim3StartIsNeumann, bool dim3EndIsNeumann,
            float dim1StartVal, float dim1EndVal, float dim2StartVal, float dim2EndVal, float dim3StartVal, float dim3EndVal,
            size_t dim1SegSpacing, size_t dim2SegSpacing, size_t dim3SegSpacing,
            size_t forceSize,
            size_t nnzMax,
            float *p, float *f,
            float* dim1Delta, float* dim2Delta, float* dim3Delta, double dt,
            bool dim1UniformDelta, bool dim2UniformDelta, bool dim3UniformDelta,
            double tol, size_t maxIterations
        ) {
            initImmersedEq<float, int64_t>(
                dim1Length, dim3Length, dim2Length,
                dim3StartIsNeumann, dim3EndIsNeumann, dim1StartIsNeumann, dim1EndIsNeumann, dim2StartIsNeumann, dim2EndIsNeumann,
                dim3StartVal, dim3EndVal, dim1StartVal, dim1EndVal, dim2StartVal, dim2EndVal,
                dim3SegSpacing, dim1SegSpacing, dim2SegSpacing,
                forceSize,
                nnzMax,
                p, f,
                dim3Delta, dim1Delta, dim2Delta, dt,
                dim3UniformDelta, dim1UniformDelta, dim2UniformDelta,
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
    std::vector<std::unique_ptr<EigenDecompForFortran<Real>>> solvers;

    // inline double currentTime() {
    //     return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    // }

    // std::atomic<double> totalEigenSolverTime{0.0};
    // inline void addSolverTime(double elapsed) {
    //     double current = totalEigenSolverTime.load();
    //     while (!totalEigenSolverTime.compare_exchange_weak(current, current + elapsed));
    // }

    template<typename Real>
    size_t initEigenDecompSolver(
        size_t rows, size_t cols, size_t layers,
        const Real* xDelta, const Real* yDelta, const Real* zDelta,
        size_t xSegSpacing, size_t ySegSpacing, size_t zSegSpacing,
        bool xStartIsNeumann, bool xEndIsNeumann, bool yStartIsNeumann, bool yEndIsNeumann, bool zStartIsNeumann, bool zEndIsNeumann,
        Real xStartVal, Real xEndVal, Real yStartVal, Real yEndVal, Real zStartVal, Real zEndVal,
        bool thomas,
        Real helmholtzShift,
        size_t gpuIndex
    ) {

        Handle hand(gpuIndex);
        auto xb = Mat<Real>::create(rows * cols * layers, 3, hand);

        XYZ<std::vector<Real>> delta(
            std::vector<Real>(xDelta, xDelta + (hasVariableDelta(xSegSpacing) ? cols + 1 : 1)),
            std::vector<Real>(yDelta, yDelta + (hasVariableDelta(ySegSpacing) ? rows + 1 : 1)),
            std::vector<Real>(zDelta, zDelta + (hasVariableDelta(zSegSpacing) ? layers + 1 : 1))
        );

        size_t solverIndex = solvers<Real>.size();

        solvers<Real>.push_back(
            std::make_unique<EigenDecompForFortran<Real>>(
                GridDim(rows, cols, layers),
                delta,
                XYZ<bool>(xStartIsNeumann, yStartIsNeumann, zStartIsNeumann),
                XYZ<bool>(xEndIsNeumann, yEndIsNeumann, zEndIsNeumann),
                XYZ<Real>(xStartVal, yStartVal, zStartVal),
                XYZ<Real>(xEndVal, yEndVal, zEndVal),
                XYZ<eigen::LaplOperatorT>(
                    static_cast<eigen::LaplOperatorT>(xSegSpacing),
                    static_cast<eigen::LaplOperatorT>(ySegSpacing),
                    static_cast<eigen::LaplOperatorT>(zSegSpacing)
                ),
                thomas,
                helmholtzShift,
                xb.col(0), xb.col(1), xb.col(2),
                gpuIndex
            )
        );

        // addSolverTime(currentTime() - startTime);

        return solverIndex;
    }

    template<typename Real>
    void runDecompSolver(size_t solverHandle, Real* bHost) {
        // double startTime = currentTime();

        if (solverHandle >= solvers<Real>.size() || !solvers<Real>[solverHandle])
            throw std::runtime_error("Invalid eigen solver handle.");

        solvers<Real>[solverHandle]->solve(bHost);

        // addSolverTime(currentTime() - startTime);
    }

    template<typename Real>
    void synch(size_t solverHandle, Real* x) {
        // double startTime = currentTime();
        if (solverHandle >= solvers<Real>.size() || !solvers<Real>[solverHandle])
            throw std::runtime_error("Invalid eigen solver handle.");
        solvers<Real>[solverHandle]->retrieveSoltion(x);
        // addSolverTime(currentTime() - startTime);
    }

    // --- Initialization Functions ---
    extern "C" {
        inline size_t initEigenDecomp_d(
            size_t dim1Length, size_t dim2Length, size_t dim3Length,
            const double* dim1Delta, const double* dim2Delta, const double* dim3Delta,
            int dim1SegType, int dim2SegType, int dim3SegType,
            bool dim1StartIsNeumann, bool dim1EndIsNeumann, bool dim2StartIsNeumann, bool dim2EndIsNeumann, bool dim3StartIsNeumann, bool dim3EndIsNeumann,
            double dim1StartVal, double dim1EndVal, double dim2StartVal, double dim2EndVal, double dim3StartVal, double dim3EndVal,
            bool thomas,
            double helmholtzShift,
            size_t gpuIndex
        ) {
            return initEigenDecompSolver<double>(
                dim1Length, dim3Length, dim2Length,
                dim3Delta, dim1Delta, dim2Delta,
                dim3SegType, dim1SegType, dim2SegType,
                dim3StartIsNeumann, dim3EndIsNeumann, dim1StartIsNeumann, dim1EndIsNeumann, dim2StartIsNeumann, dim2EndIsNeumann,
                dim3StartVal, dim3EndVal, dim1StartVal, dim1EndVal, dim2StartVal, dim2EndVal,
                thomas, helmholtzShift, gpuIndex);
        }

        inline size_t initEigenDecomp_s(
            size_t dim1Length, size_t dim2Length, size_t dim3Length,
            const float* dim1Delta, const float* dim2Delta, const float* dim3Delta,
            int dim1SegType, int dim2SegType, int dim3SegType,
            bool dim1StartIsNeumann, bool dim1EndIsNeumann, bool dim2StartIsNeumann, bool dim2EndIsNeumann, bool dim3StartIsNeumann, bool dim3EndIsNeumann,
            float dim1StartVal, float dim1EndVal, float dim2StartVal, float dim2EndVal, float dim3StartVal, float dim3EndVal,
            bool thomas, float helmholtzShift, size_t gpuIndex
        ) {
            return initEigenDecompSolver<float>(
                dim1Length, dim3Length, dim2Length,
                dim3Delta, dim1Delta, dim2Delta,
                dim3SegType, dim1SegType, dim2SegType,
                dim3StartIsNeumann, dim3EndIsNeumann, dim1StartIsNeumann, dim1EndIsNeumann, dim2StartIsNeumann, dim2EndIsNeumann,
                dim3StartVal, dim3EndVal, dim1StartVal, dim1EndVal, dim2StartVal, dim2EndVal,
                thomas, helmholtzShift, gpuIndex
            );
        }

        inline void solveEigenDecomp_d(size_t solverHandle, double* b) {
            runDecompSolver(solverHandle, b);
        }

        inline void solveEigenDecomp_s(size_t solverHandle, float* b) {
            runDecompSolver(solverHandle, b);
        }

        inline void synch_d(size_t solverHandle, double* x) {
            synch<double>(solverHandle, x);
        }

        inline void synch_s(size_t solverHandle, float* x) {
            synch<float>(solverHandle, x);
        }
    }

    void finalizeEigenDecomp() {
        // double startTime = currentTime();
        solvers<float>.clear();
        solvers<double>.clear();

        GPUScalar<float>::finalize();
        GPUScalar<double>::finalize();
        // addSolverTime(currentTime() - startTime);
        // std::cout << "Total eigen decomp time: " << totalEigenSolverTime.load() << std::endl;
    }
};