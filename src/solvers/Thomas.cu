//
// Created by usr on 2/8/26.
//

#include "Thomas.cuh"

#include <vector>


template<typename Real>
Thomas<Real>::Thomas(Mat<Real> heightX2TimesNumSys) :
    cPrime(heightX2TimesNumSys.subMat(
        0,
        0,
        heightX2TimesNumSys._rows,
        heightX2TimesNumSys._cols / 2
    )),
    dPrime(heightX2TimesNumSys.subMat(
        0,
        heightX2TimesNumSys._cols / 2,
        heightX2TimesNumSys._rows,
        heightX2TimesNumSys._cols / 2
    )) {
}

template<typename Real>
Thomas<Real>::Thomas(size_t height, size_t numSystems) : Thomas(Mat<Real>::create(height, numSystems * 2)) {
}


/**
 * @brief CUDA kernel for the parallel execution of the Thomas Algorithm across multiple systems.
 * * Maps each independent tridiagonal system to a unique GPU thread to compute
 * the solution vector $x$ from the provided diagonals and Right-Hand Side.
 * * @tparam Real Floating-point precision type.
 * @param triDiags   [in] 3D tensor containing the three diagonals (sub, primary, super).
 * @param x          [out] 2D matrix where the computed solution vectors are written.
 * @param b          [in] 2D matrix containing the Right-Hand Side (RHS) vectors.
 * @param superPrime [workspace] 2D matrix for intermediate coefficient storage.
 * @param bPrime     [workspace] 2D matrix for intermediate RHS storage.
 */
template<typename Real>
__global__ void solveThomasKernel(const DeviceData3d<Real> triDiags, DeviceData2d<Real> x, const DeviceData2d<Real> b,
                                  DeviceData2d<Real> superPrime, DeviceData2d<Real> bPrime) {
    if (size_t system = idx(); system < triDiags.layers) {
        Real super = triDiags(0, 2, system);
        Real primary = triDiags(0, 1, system);
        Real rhs = b(0, system);
        Real sub, denom;
        superPrime(0, system) = super / primary;
        bPrime(0, system) = rhs / primary;
        for (size_t row = 1; row < triDiags.rows; row++) {
            sub = triDiags(row, 0, system);
            primary = triDiags(row, 1, system);
            super = triDiags(row, 2, system);
            rhs = b(row, system);
            denom = 1 / (primary - sub * superPrime(row - 1, system));

            superPrime(row, system) = super * denom;
            bPrime(row, system) = (rhs - sub * bPrime(row - 1, system)) * denom;
        }
        size_t n = triDiags.rows - 1;
        x(n, system) = bPrime(n, system);
        for (int32_t row = n - 1; row >= 0; --row)
            x(row, system) = bPrime(row, system) - superPrime(row, system) * x(row + 1, system);
    }
}

/**
 * @brief Specialized kernel for 2D Laplacian systems oriented along rows (transposed).
 * Diagonal coefficients are hardcoded to 4 (primary) and -1 (off-diagonals).
 */
template<typename Real>
__global__ void solveThomas2dLaplacianKernelTranspose(DeviceData2d<Real> x, const DeviceData2d<Real> b,
                                                      DeviceData2d<Real> superPrime, DeviceData2d<Real> bPrime) {
    if (size_t system = idx(); system < x.rows) {
        Real rhs = b(system, 0);
        Real denom;
        superPrime(0, system) = -0.25;
        bPrime(0, system) = rhs / 4;
        for (size_t col = 1; col < x.cols; col++) {
            rhs = b(system, col);
            denom = 1 / (4 + superPrime(col - 1, system));

            superPrime(col, system) = -denom;
            bPrime(col, system) = (rhs + bPrime(col - 1, system)) * denom;
        }
        size_t n = x.cols - 1;
        x(system, n) = bPrime(n, system);
        for (int32_t col = n - 1; col >= 0; --col)
            x(system, col) = bPrime(col, system) - superPrime(col, system) * x(system, col + 1);
    }
}

/**
 * @brief Specialized kernel for 2D Laplacian systems oriented along columns.
 * Diagonal coefficients are hardcoded to 4 (primary) and -1 (off-diagonals).
 */
template<typename Real>
__global__ void solveThomas2dLaplacianKernel(DeviceData2d<Real> x, const DeviceData2d<Real> b,
                                             DeviceData2d<Real> superPrime, DeviceData2d<Real> bPrime) {
    if (size_t system = idx(); system < x.cols) {
        Real rhs = b(0, system);
        Real denom;
        superPrime(0, system) = -0.25;
        bPrime(0, system) = rhs / 4;
        for (size_t row = 1; row < x.rows; row++) {
            rhs = b(row, system);
            denom = 1 / (4 + superPrime(row - 1, system));

            superPrime(row, system) = -denom;
            bPrime(row, system) = (rhs + bPrime(row - 1, system)) * denom;
        }
        size_t n = x.rows - 1;
        x(n, system) = bPrime(n, system);
        for (int32_t row = n - 1; row >= 0; --row)
            x(row, system) = bPrime(row, system) - superPrime(row, system) * x(row + 1, system);
    }
}

/**
 * @brief Specialized kernel for 3D Laplacian systems oriented along rows (transposed).
 * Diagonal coefficients are hardcoded to 6 (primary) and -1 (off-diagonals).
 */
template<typename Real>
__global__ void solveThomas3dLaplacianKernelTranspose(DeviceData2d<Real> x, const DeviceData2d<Real> b,
                                                      DeviceData2d<Real> superPrime, DeviceData2d<Real> bPrime) {
    if (size_t system = idx(); system < x.rows) {
        Real rhs = b(system, 0);
        Real denom;
        superPrime(0, system) = -1.0 / 2;
        bPrime(0, system) = rhs / 2;
        for (size_t col = 1; col < x.cols; col++) {
            rhs = b(system, col);
            denom = 1 / (2 + superPrime(col - 1, system));

            superPrime(col, system) = -denom;
            bPrime(col, system) = (rhs + bPrime(col - 1, system)) * denom;
        }
        size_t n = x.cols - 1;
        x(system, n) = bPrime(n, system);
        for (int32_t col = n - 1; col >= 0; --col)
            x(system, col) = bPrime(col, system) - superPrime(col, system) * x(system, col + 1);
    }
}

/**
 * @brief Specialized kernel for 3D Laplacian systems oriented along the depth (Z-axis).
 * Uses a 2D grid where each thread processes a vertical column of the tensor.
 */
template<typename Real>
__global__ void solveThomas3dLaplacianKernel(DeviceData2d<Real> x, const DeviceData2d<Real> b,
                                             DeviceData2d<Real> superPrime, DeviceData2d<Real> bPrime) {
    if (size_t system = idx(); system < x.cols) {
        Real rhs = b(0, system);
        Real denom;
        superPrime(0, system) = -1.0 / 2;
        bPrime(0, system) = rhs / 2;
        for (size_t row = 1; row < x.rows; row++) {
            rhs = b(row, system);
            denom = 1 / (2 + superPrime(row - 1, system));

            superPrime(row, system) = -denom;
            bPrime(row, system) = (rhs + bPrime(row - 1, system)) * denom;
        }
        size_t n = x.rows - 1;
        x(n, system) = bPrime(n, system);
        for (int32_t row = n - 1; row >= 0; --row)
            x(row, system) = bPrime(row, system) - superPrime(row, system) * x(row + 1, system);
    }
}

template<typename Real>
__global__ void solveThomas3dLaplacianDepthsKernel(DeviceData3d<Real> x, DeviceData3d<Real> b,
                                                   DeviceData2d<Real> superPrime, DeviceData2d<Real> bPrime) {
    GridInd3d system(idy(), idx(), 0);
    if (system.row >= x.rows || system.col >= x.cols) return;
//__device__  DeviceData1d(size_t size, DeviceData3d<T> &src, GridInd3d &ind0, size_t dRow, size_t dCol, size_t dLayer);
    DeviceData1d<Real> depthX(x.layers, x, system, 0, 0, 1);
    DeviceData1d<Real> depthB(b.layers, b, system, 0, 0, 1);

    Real rhs = depthB[0];
    Real denom;

    size_t sysFlatInd = system.col * x.rows +system.row;

    superPrime(0, sysFlatInd) = -1.0 / 2;
    bPrime(0, sysFlatInd) = rhs / 2;
    for (size_t layer = 1; layer < x.layers; layer++) {
        rhs = depthB[layer];
        denom = 1 / (2 + superPrime(layer - 1, sysFlatInd));

        superPrime(layer, sysFlatInd) = -denom;
        bPrime(layer, sysFlatInd) = (rhs + bPrime(layer - 1, sysFlatInd)) * denom;
    }
    size_t n = x.layers - 1;
    depthX[n] = bPrime(n, sysFlatInd);
    for (int32_t layer = n - 1; layer >= 0; --layer)
        depthX[layer] = bPrime(layer, sysFlatInd) - superPrime(layer, sysFlatInd) * depthX[layer + 1];
}

template<typename Real>
void Thomas<Real>::solve(const Tensor<Real> &triDiags, Mat<Real> &result, Mat<Real> &b, Handle &hand) {
    KernelPrep kp(triDiags._layers);
    solveThomasKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        triDiags.toKernel3d(),
        result.toKernel2d(),
        b.toKernel2d(),
        cPrime.toKernel2d(),
        dPrime.toKernel2d()
    );
}

template<typename Real>
void Thomas<Real>::solveLaplacian(Mat<Real> &result, Mat<Real> &b, bool is3d, Handle &hand) {
    KernelPrep kp(result._cols);
    if (is3d) solveThomas3dLaplacianKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        result.toKernel2d(),
        b.toKernel2d(),
        cPrime.toKernel2d(),
        dPrime.toKernel2d()
    );
    else solveThomas2dLaplacianKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        result.toKernel2d(),
        b.toKernel2d(),
        cPrime.toKernel2d(),
        dPrime.toKernel2d()
    );
}

template<typename Real>
void Thomas<Real>::solveLaplacianTranspose(Mat<Real> &result, Mat<Real> &b, bool is3d, Handle &hand) {
    KernelPrep kp(result._rows);
    if (is3d) solveThomas3dLaplacianKernelTranspose<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        result.toKernel2d(),
        b.toKernel2d(),
        cPrime.toKernel2d(),
        dPrime.toKernel2d()
    );
    else solveThomas2dLaplacianKernelTranspose<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        result.toKernel2d(),
        b.toKernel2d(),
        cPrime.toKernel2d(),
        dPrime.toKernel2d()
    );
}

template<typename Real>
void Thomas<Real>::solveLaplacianDepths(Tensor<Real> &result, Tensor<Real> &b, Handle &hand) {
    KernelPrep kp(result.layerSize());
    solveThomas3dLaplacianDepthsKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
        result.toKernel3d(),
        b.toKernel3d(),
        cPrime.toKernel2d(),
        dPrime.toKernel2d()
    );

}


template<typename Real>
bool Thomas<Real>::test(size_t systemSize, size_t numSystems) {
    Handle hand;

    Mat<Real> solverWorkspace = Mat<Real>::create(systemSize, 2 * numSystems, hand);

    Thomas<Real> solver(solverWorkspace);
    Tensor<Real> triDiags = Tensor<Real>::create(systemSize, 3, numSystems, hand);

    Mat<Real> rhs = Mat<Real>::create(systemSize, numSystems, hand);

    Mat<Real> solution = Mat<Real>::create(systemSize, numSystems, hand);

    // ---------------------------------------------------------------------
    // 3. Prepare host-side test problem
    //
    // Each system solves:
    //
    //   A x = b
    //
    // where A is the standard 1D Laplacian:
    //   [-1,  2, -1]
    //
    // and the exact solution is chosen as:
    //   x = [1, 2, 3, ..., systemSize]
    //
    // We compute b = A * x analytically so the expected solution is known.
    // ---------------------------------------------------------------------
    std::vector<Real> hostTriDiags(systemSize * 3 * numSystems);
    std::vector<Real> hostRhs(systemSize * numSystems);
    std::vector<Real> hostExpectedSolution(systemSize * numSystems);

    for (size_t s = 0; s < numSystems; ++s) {
        // Define exact solution x(r) = r + 1
        for (size_t r = 0; r < systemSize; ++r) hostExpectedSolution[s * systemSize + r] = static_cast<Real>(r + 1);

        // Fill tridiagonal coefficients and compute RHS = A * x
        for (size_t r = 0; r < systemSize; ++r) {
            const size_t triBase = s * systemSize * 3 + r * 3;
            const size_t vecIdx = s * systemSize + r;

            const Real x_r = hostExpectedSolution[vecIdx];
            const Real x_l = (r > 0) ? hostExpectedSolution[vecIdx - 1] : Real(0);
            const Real x_u = (r + 1 < systemSize) ? hostExpectedSolution[vecIdx + 1] : Real(0);

            // Tridiagonal matrix entries
            hostTriDiags[triBase + 0] = (r == 0) ? Real(0) : Real(-1); // sub
            hostTriDiags[triBase + 1] = Real(2); // diag
            hostTriDiags[triBase + 2] =
                    (r + 1 == systemSize) ? Real(0) : Real(-1); // super

            // RHS entry: b = 2*x_r - x_l - x_u
            hostRhs[vecIdx] = Real(2) * x_r - x_l - x_u;
        }
    }

    // ---------------------------------------------------------------------
    // 4. Upload data and solve on GPU
    // ---------------------------------------------------------------------
    triDiags.set(hostTriDiags.data(), hand);
    rhs.set(hostRhs.data(), hand);

    solver.solve(triDiags, solution, rhs, hand);
    hand.synch();

    // ---------------------------------------------------------------------
    // 5. Download result and verify
    //
    // We check the maximum absolute error across all systems.
    // ---------------------------------------------------------------------
    std::vector<Real> hostSolution(systemSize * numSystems);
    solution.get(hostSolution.data(), hand);
    hand.synch();

    Real maxError = Real(0);
    for (size_t i = 0; i < hostSolution.size(); ++i) {
        maxError =
                std::max(maxError,
                         std::abs(hostSolution[i] -
                                  hostExpectedSolution[i]));
    }

    return maxError < Real(1e-5);
}


template class Thomas<double>;
template class Thomas<float>;
