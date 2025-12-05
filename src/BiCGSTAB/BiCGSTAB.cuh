#ifndef BICGSTAB_H
#define BICGSTAB_H

#include "deviceArrays/headers/BandedMat.h"

#include "Event.h"
#include <iostream>
#include <chrono>
#include <array>
#include <functional> // For std::reference_wrapper

#include "deviceArrays/headers/Streamable.h"

// Forward declaration of required types (assuming they are defined elsewhere)
template<typename T> class DeviceData1d;
template<typename T> class Vec;
template<typename T> class Mat;
template<typename T> class BandedMat;
class Event;
class Handle; // Assuming Handle is a wrapper for cudaStream_t

using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

/**
 * @brief Implements the Bi-Conjugate Gradient Stabilized (BiCGSTAB) iterative solver
 * for sparse linear systems $A\mathbf{x} = \mathbf{b}$ on the GPU using CUDA streams
 * and cuBLAS for high performance.
 *
 * The implementation uses multiple CUDA streams and events to overlap communication,
 * computation, and I/O operations, improving overall solver efficiency.
 *
 * @tparam T Floating-point type used for the computation (float or double).
 */
template<typename T>
class BiCGSTAB {
private:
    const T tolerance;
    Handle handle[4]{};
    Event alphaReady, sReady, hReady, omegaReady, rReady, xReady, prodTS;
    const Vec<T> b;
    Mat<T> paM;
    Vec<T> r, r_tilde, p, v, s, t, h;
    Vec<T> paV;
    Singleton<T> rho, alpha, omega, rho_new, beta;
    std::array<Singleton<T>, 4> temp;

    const size_t maxIterations;

    /**
     * @brief Waits for a list of events to complete on a specified CUDA stream.
     */
    void wait(const size_t streamIndex, const std::initializer_list<std::reference_wrapper<Event> > evs) const;

    /**
    * @brief Renews (resets) a list of events, preparing them for the next iteration.
    */
    static void renew(const std::initializer_list<std::reference_wrapper<Event> > evs);

    /**
    * @brief Records an event on a specified CUDA stream.
    */
    void record(size_t streamIndex, Event &e) const;

    /**
     * @brief Synchronizes a specific CUDA stream handle.
     */
    void synch(const size_t streamInd) const;

    /**
     * @brief Checks if the squared $L_2$ norm of a vector is smaller than the tolerance.
     */
    bool isSmall(const Vec<T> &v, Singleton<T> preAlocated, const size_t streamInd);

    /**
     * @brief Sets the content of the destination vector to be equal to the source vector.
     */
    void set(Vec<T> &dst, const Vec<T> &src, const size_t streamInd);

    /**
     * @brief Executes the BiCGSTAB P vector update: p = r + beta * (p - omega * v).
     */
    void pUpdate(const size_t streamInd);

    /**
 *
 * @brief Initalizes variables r_tilde, r, b, p, and rx.ho
 */
    void preamable(const BandedMat<T> &A, Vec<T> &x);

public:
    /**
     * @brief Constructor for the BiCGSTAB solver.
     */
    explicit BiCGSTAB(
        const Vec<T> &b,
        Mat<T> *preAllocated = nullptr,
        const T tolerance = std::is_same_v<T, double> ? T(1e-15) : T(1e-6),
        const size_t maxIterations = 1500
    );

    /**
     * @brief Static method to solve the equation $A\mathbf{x} = \mathbf{b}$.
     */
    static void solve(
        const BandedMat<T> &A,
        Vec<T> &x,
        const Vec<T> &b,
        Mat<T> *preAllocated = nullptr,
        const T tolerance = std::is_same_v<T, double> ? T(1e-15) : T(1e-6),
        const size_t maxIterations = 1500
    );

    /**
     * Solves Ax = b
     * @param A The banded matrix.  Each column represents a diagonal of a sparse matrix.  Shorter diagonals will have
     * trailing padding, but never leading padding.  There should be as many columns as there are diagonals in the
     * square sparse matrix, and as many rows as there are rows in the square sparse matrix.
     * @param aLd The leading dimension of A.  It is the distance between the first elements of each column.  Must be
     * at least the number of rows in A, but may be more if there's padding.
     * @param inds The ith element is the diagonal index of the ith column in A.  Super diagonals have positive indices,
     * and subdiagonals have negative indices.  The absolute value of the index is the distance of the diagonal from the
     * primary diagonal.
     * @param indsStride  The distance between elements of inds.  This is usually 1.
     * @param numInds The number of diagonals.
     * @param x The solution will be put here.
     * @param xStride The distance between elements of x.
     * @param b The RHS of Ax=b.
     * @param bStride The distance between elements of b.
     * @param bSize The number of elements in b, x, and the number of rows in A.
     * @param prealocatedSizeX7 should have bSize rows and 7 columns.  Will be overwritten.
     * @param prealocatedLd The distance between the first elements of each column of prealocatedSizeX7.
     * @param maxIterations The maximum number of iterations.
     * @param tolerance What's close enough to 0.
     */
    static void solve(
        T *A, size_t aLd,
        int32_t *inds, size_t indsStride, size_t numInds,
        T *x, size_t xStride,
        T *b, size_t bStride, size_t bSize,
        T *prealocatedSizeX7,
        size_t prealocatedLd,
        size_t maxIterations = 1500,
        T tolerance = std::is_same_v<T, double> ? T(1e-15) : T(1e-6)
    );

    /**
     * @brief Solves the linear system $A\mathbf{x} = \mathbf{b}$ using the
     * unpreconditioned BiCGSTAB algorithm.
     */
    void solveUnpreconditionedBiCGSTAB(const BandedMat<T> &A, Vec<T> &x);
};

#endif // BICGSTAB_H
