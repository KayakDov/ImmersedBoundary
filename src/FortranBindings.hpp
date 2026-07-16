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
        size_t height, size_t width, size_t depth,
        bool leftIsNeumann, bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool frontIsNeumann, bool backIsNeumann,
        Real leftVal, Real rightVal, Real topVal, Real bottomVal, Real frontVal, Real backVal,
        bool isStaggered,
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

        buildBoundaryConfigAndLaunch(
            GridDim(height, width, depth),
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
            bool isStaggered,
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
                isStaggered,
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
            bool isStaggered,
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
                isStaggered,
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
            bool isStaggered,
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
                isStaggered,
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
            bool isStaggered,
            size_t forceSize,
            size_t nnzMax,
            float *p, float *f,
            float* dim1Delta, float* dim2Delta, double dt, float* dim3Delta,
            bool dim1UniformDelta, bool dim2UniformDelta, bool dim3UniformDelta,
            double tol, size_t maxIterations
        ) {
            initImmersedEq<float, int64_t>(
                dim1Length, dim3Length, dim2Length,
                dim3StartIsNeumann, dim3EndIsNeumann, dim1StartIsNeumann, dim1EndIsNeumann, dim2StartIsNeumann, dim2EndIsNeumann,
                dim3StartVal, dim3EndVal, dim1StartVal, dim1EndVal, dim2StartVal, dim2EndVal,
                isStaggered,
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

    template<typename Real>
    size_t initEigenDecompSolver(
        size_t rows, size_t cols, size_t layers,
        Real* dim3Delta, Real* dim1Delta, Real* dim2Delta,
        bool dim3UniformDelta, bool dim1UniformDelta, bool dim2UniformDelta,
        bool dim3StartIsNeumann, bool dim3EndIsNeumann, bool dim1StartIsNeumann, bool dim1EndIsNeumann, bool dim2StartIsNeumann, bool dim2EndIsNeumann,
        Real dim3StartVal, Real dim3EndVal, Real dim1StartVal, Real dim1EndVal, Real dim2StartVal, Real dim2EndVal,
        bool isStaggered,
        bool thomas,
        Real helmholtzShift
    ) {

    std::cout << "\n========== EigenDecompForFortran Constructor ==========\n";

     std::cout << "Dimensions:\n";
     std::cout << "  rows   = " << rows   << '\n';
     std::cout << "  cols   = " << cols   << '\n';
     std::cout << "  layers = " << layers << '\n';



     std::cout << "=======================================================" << std::endl;


        auto xb = Mat<Real>::create(rows * cols * layers, 3);

        size_t solverHandle = solvers<Real>.size();
        solvers<Real>.push_back(
            std::make_unique<EigenDecompForFortran<Real>>(
                rows, cols, layers,
                std::vector<Real>(dim3Delta, dim3Delta + (dim3UniformDelta ? 1 : cols + 1)),
                std::vector<Real>(dim1Delta, dim1Delta + (dim1UniformDelta ? 1 : rows + 1)),
                std::vector<Real>(dim2Delta, dim2Delta + (dim2UniformDelta ? 1 : layers + 1)),
                dim3StartIsNeumann, dim3EndIsNeumann, dim1StartIsNeumann, dim1EndIsNeumann, dim2StartIsNeumann, dim2EndIsNeumann,
                dim3StartVal, dim3EndVal, dim1StartVal, dim1EndVal, dim2StartVal, dim2EndVal,
                isStaggered,
                thomas,
                helmholtzShift,
                xb.col(0), xb.col(1), xb.col(2)
            )
        );
        return solverHandle;
    }

    template<typename Real>
    void runDecompSolver(size_t solverHandle, Real* xHost, Real* bHost) {
        if (solverHandle >= solvers<Real>.size() || !solvers<Real>[solverHandle])
            throw std::runtime_error("Invalid eigen solver handle.");

        std::cout << "\n========== EigenDecompForFortran Solve ==========\n";

        std::cout << "Dimensions:\n";
        std::cout << "  solver Handle   = " << solverHandle   << " out of " << solvers<Real>.size() << '\n';


        std::cout << "=======================================================" << std::endl;


        solvers<Real>[solverHandle]->solve(xHost, bHost);
        cudaDeviceSynchronize();
    }

    template<typename Real>
    void finalizeEigenDecomp() {
        solvers<Real>.clear();
    }

    // --- Initialization Functions ---
    extern "C" {
        inline size_t initEigenDecomp_d(
            size_t dim1Length, size_t dim2Length, size_t dim3Length,
            double* dim1Delta, double* dim2Delta, double* dim3Delta,
            bool dim1UniformDelta, bool dim2UniformDelta, bool dim3UniformDelta,
            bool dim1StartIsNeumann, bool dim1EndIsNeumann, bool dim2StartIsNeumann, bool dim2EndIsNeumann, bool dim3StartIsNeumann, bool dim3EndIsNeumann,
            double dim1StartVal, double dim1EndVal, double dim2StartVal, double dim2EndVal, double dim3StartVal, double dim3EndVal,
            bool isStaggered,
            bool thomas,
            double helmholtzShift
        ) {
            return initEigenDecompSolver<double>(
                dim1Length, dim3Length, dim2Length,
                dim3Delta, dim1Delta, dim2Delta,
                dim3UniformDelta, dim1UniformDelta, dim2UniformDelta,
                dim3StartIsNeumann, dim3EndIsNeumann, dim1StartIsNeumann, dim1EndIsNeumann, dim2StartIsNeumann, dim2EndIsNeumann,
                dim3StartVal, dim3EndVal, dim1StartVal, dim1EndVal, dim2StartVal, dim2EndVal,
                isStaggered, thomas, helmholtzShift);
        }

        inline size_t initEigenDecomp_s(
            size_t dim1Length, size_t dim2Length, size_t dim3Length,
            float* dim1Delta, float* dim2Delta, float* dim3Delta,
            bool dim1UniformDelta, bool dim2UniformDelta, bool dim3UniformDelta,
            bool dim1StartIsNeumann, bool dim1EndIsNeumann, bool dim2StartIsNeumann, bool dim2EndIsNeumann, bool dim3StartIsNeumann, bool dim3EndIsNeumann,
            float dim1StartVal, float dim1EndVal, float dim2StartVal, float dim2EndVal, float dim3StartVal, float dim3EndVal,
            bool isStaggered,
            bool thomas,
            float helmholtzShift
        ) {
            return initEigenDecompSolver<float>(
                dim1Length, dim3Length, dim2Length,
                dim3Delta, dim1Delta, dim2Delta,
                dim3UniformDelta, dim1UniformDelta, dim2UniformDelta,
                dim3StartIsNeumann, dim3EndIsNeumann, dim1StartIsNeumann, dim1EndIsNeumann, dim2StartIsNeumann, dim2EndIsNeumann,
                dim3StartVal, dim3EndVal, dim1StartVal, dim1EndVal, dim2StartVal, dim2EndVal,
                isStaggered, thomas, helmholtzShift
            );
        }

        inline void solveEigenDecomp_d(size_t solverHandle, double *x, double* b) {
            runDecompSolver(solverHandle, x, b);
        }

        inline void solveEigenDecomp_s(size_t solverHandle, float *x, float* b) {
            runDecompSolver(solverHandle, x, b);
        }

        inline void finalizeEigenDecomp_d() {
            finalizeEigenDecomp<double>();
        }

        inline void finalizeEigenDecomp_s() {
            finalizeEigenDecomp<float>();
        }
    }
};