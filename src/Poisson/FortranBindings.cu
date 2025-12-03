#include <chrono>

#include "EigenDecompSolver.cu"
#include "BiCGSTAB/RunDirectSolver.cu"





/**
 * @brief Solve the 3D eigen-decomposition linear system on GPU data.
 *
 * This method wraps raw device pointers into Mat<T>, Vec<T>, and CubeBoundary<T>
 * objects and executes an EigenDecompSolver. No memory is allocated or deallocated.
 *
 * ## Data Format (IMPORTANT)
 * (unchanged — omitted for brevity)
 */
template<typename T>
void eigenDecompSolver(const T *frontBack, const size_t fbLd,
                       const T *leftRight, const size_t lrLd,
                       const T *topBottom, const size_t tbLd,
                       T *f, const size_t fStride,
                       T *x, const size_t xStride,
                       const size_t height,
                       const size_t width,
                       const size_t depth) {
    // Construct faces: rows, cols, leading dimension, data
    Mat<T> fb(2 * height, width, fbLd, nonOwningGpuPtr(const_cast<T *>(frontBack)));
    Mat<T> lr(2 * height, depth, lrLd, nonOwningGpuPtr(const_cast<T *>(leftRight)));
    Mat<T> tb(2 * depth, width, tbLd, nonOwningGpuPtr(const_cast<T *>(topBottom)));

    CubeBoundary<T> boundary(fb, lr, tb);

    const size_t n = height * width * depth;

    Vec<T> xVec(n, nonOwningGpuPtr(x), xStride);
    Vec<T> fVec(n, nonOwningGpuPtr(const_cast<T *>(f)), fStride);

    Handle hand;

    cudaDeviceSynchronize();
    // EigenDecompSolver eds(boundary, xVec, fVec, hand);  TODO: ask for prealocated memory and pass to here.
    cudaDeviceSynchronize();
}


// ============================================================================
//                    EXPORTED SYMBOLS FOR FORTRAN
// ============================================================================

extern "C" {
void eigenDecompSolver_float_(
    const float *frontBack, const size_t *fbLd,
    const float *leftRight, const size_t *lrLd,
    const float *topBottom, const size_t *tbLd,
    float *f, const size_t *fStride,
    float *x, const size_t *xStride,
    const size_t *height, const size_t *width,
    const size_t *depth) {
    eigenDecompSolver<float>(
        frontBack, *fbLd,
        leftRight, *lrLd,
        topBottom, *tbLd,
        f, *fStride,
        x, *xStride,
        *height, *width, *depth
    );
}


void eigenDecompSolver_double_(
    const double *frontBack, const size_t *fbLd,
    const double *leftRight, const size_t *lrLd,
    const double *topBottom, const size_t *tbLd,
    double *f, const size_t *fStride,
    double *x, const size_t *xStride,
    const size_t *height, const size_t *width,
    const size_t *depth) {
    eigenDecompSolver<double>(
        frontBack, *fbLd,
        leftRight, *lrLd,
        topBottom, *tbLd,
        f, *fStride,
        x, *xStride,
        *height, *width, *depth
    );
}
} // extern "C"

using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

template<typename T>
void benchMarkEigenDecompSolver(size_t height, size_t width, size_t depth, Handle hand3[]) {
    const auto boundary = CubeBoundary<double>::ZeroTo1(height, width, depth, hand3[0]);

    auto memAlocFX = Mat<double>::create(boundary.internalSize(), 2);


    size_t maxDim = std::max(width, std::max(height, depth));

    Mat<T> eigenStorage = Mat<T>::create(maxDim, width + height + depth + 3);
    SquareMat<T> eX = eigenStorage.sqSubMat(0, 0, width),
            eY = eigenStorage.sqSubMat(0, width, height),
            eZ = eigenStorage.sqSubMat(0, width + height, depth);

    Mat<T> vals = eigenStorage.subMat(0, width + height + depth, maxDim, 3);

    auto x = memAlocFX.col(0);
    auto f = memAlocFX.col(1);

    f.fill(0, hand3[0]);

    cudaDeviceSynchronize();

    TimePoint start = std::chrono::steady_clock::now();
    EigenDecompSolver<double> fdm(boundary, x, f, eX, eY, eZ, vals, hand3);
    cudaDeviceSynchronize();
    TimePoint end = std::chrono::steady_clock::now();


    double iterationTime = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();

    std::cout << iterationTime << ", ";

    // std::cout << "x = \n" << GpuOut<double>(x.tensor(height, depth), hand3[0]) << std::endl;
}

template<typename T>
void benchMarkEigenDecompSolver(size_t dim, Handle hand3[]) {
    benchMarkEigenDecompSolver<T>(dim, dim, dim, hand3);
}

/**
 * @brief Main entry point to demonstrate the FastDiagonalizationMethod
 *        for a 2x2x2 grid.
 */
int main() {
    Handle hand[3]{};

    constexpr size_t maxDimensions = 5000;

    std::cout << "dim, time" << std::endl;

    for (size_t dim = 355; dim < maxDimensions; dim++) {
        std::cout << dim << ", ";
        benchMarkEigenDecompSolver<double>(dim, hand);
        cudaDeviceSynchronize();

        testPoisson(dim, hand[0]);
        std::cout << std::endl;
    }


    // for (size_t i = 0; i < numTests; i++) {
    //     testPoisson(dim, hand[0]);
    //     cudaDeviceSynchronize();
    // }

    return 0;
}
