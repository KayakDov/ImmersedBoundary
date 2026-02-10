#ifndef CUDABANDED_THOMAS_H
#define CUDABANDED_THOMAS_H
#include "Mat.h"

/**
 * @class Thomas
 * @brief A high-performance batch solver for tridiagonal systems of linear equations using TDMA.
 * * This class provides the pre-allocated GPU scratch space required to solve multiple
 * independent tridiagonal systems in parallel. It is designed for use in multi-dimensional
 * problems (like the Immersed Boundary Method) where many 1D lines are solved simultaneously.
 * * @tparam Real Precision type (float, double).
 */
template <typename Real>
class Thomas {

    Mat<Real> cPrime;  ///< GPU workspace for modified super-diagonal coefficients.
    Mat<Real> dPrime;  ///< GPU workspace for modified RHS coefficients.
public:

    /**
     * @brief Initializes the solver with the necessary scratch space.
     * @param heightX2TimesNumSys A matrix providing the temporary storage for the
     * forward sweep. Dimensions must be [SystemSize x (2 * NumSystems)].
     */
    explicit Thomas(Mat<Real> heightX2TimesNumSys);

    /**
     * @brief Initializes the solver and the necessary scratch space.
     * @param height The height of each system.
     * @param numSystems The number of systems.
     */
    explicit Thomas(size_t height, size_t numSystems);

    /**
     * @brief Computes the solution for a batch of tridiagonal systems.
     * * @param triDiags [in] 3D Tensor [SystemSize x 3 x NumSystems] containing
     * sub (col 0), primary (col 1), and super (col 2) diagonals.
     * @param result   [out] 2D Matrix [SystemSize x NumSystems] to store the solution.
     * @param b        [in] 2D Matrix [SystemSize x NumSystems] containing the Right-Hand Side.
     * @param hand     CUDA Handle for stream-ordered execution.
     */
    void solve(const Tensor<Real>& triDiags, Mat<Real>& result, Mat<Real>& b, Handle &hand);

    /**
     * @brief Solves a batch of Laplacian systems along the columns of a 2D matrix.
     * * Uses optimized constant coefficients for the Laplacian operator.
     * @param result [out] Matrix to store solution.
     * @param b      [in] Right-hand side matrix.
     * @param is3d   If true, uses 3D Laplacian coefficients (1/6), else 2D (1/4).
     * @param hand   CUDA Handle for execution. //TODO: check that default values of 4 (6) and -1 are correct everywhere, including for points that are at the edge of the grid.
     */
    void solveLaplacian(Mat<Real> &result, Mat<Real> &b, bool is3d, Handle &hand);

    /**
     * @brief Solves a batch of Laplacian systems along the rows of a 2D matrix (transposed orientation).
     * * @param result [out] Matrix to store solution.
     * @param b      [in] Right-hand side matrix.
     * @param is3d   If true, uses 3D Laplacian coefficients, else 2D.
     * @param hand   CUDA Handle for execution.
     */
    void solveLaplacianTranspose(Mat<Real> &result, Mat<Real> &b, bool is3d, Handle &hand);

    /**
     * @brief Solves a batch of Laplacian systems along the Z-axis (depths) of a 3D Tensor.
     * * Each (x, y) coordinate is treated as an independent tridiagonal system.
     * @param result [out] 3D Tensor to store solution.
     * @param b      [in] 3D Tensor Right-hand side.
     * @param hand   CUDA Handle for execution.
     */
    void solveLaplacianDepths(Tensor<Real> &result, Tensor<Real> &b, Handle &hand);

    /**
     * @brief Performs a numerical validation of the Thomas solver.
     * * Generates a batch of 1D Poisson-like tridiagonal systems where the
     * exact solution is known, executes the GPU kernel, and verifies
     * the results against a tolerance.
     * * @param gridHeight The number of nodes in the 1D system (N).
     * @param numSys The number of independent systems to test.
     * @return true if all systems converge to the correct solution within 1e-6.
     */
    static bool test(size_t gridHeight, size_t numSys);
};



#endif //CUDABANDED_THOMAS_H
