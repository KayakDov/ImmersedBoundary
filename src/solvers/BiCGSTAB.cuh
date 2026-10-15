/**
 * @file BiCGSTAB.cuh
 * @brief Declares BiCGSTAB iterative solver interfaces.
 * @ingroup solvers
 *
 * @details
 * Solver interfaces consume the GPU container layer and CUDA library handles to implement direct, iterative, and eigen-decomposition based algorithms.
 */

#ifndef BICGSTAB_H
#define BICGSTAB_H

#include "../deviceArrays/headers/sparse/BandedMat.h"
#include "deviceArrays/headers/SquareMat.h"

#include "deviceArrays/headers/Handles.h"
#include <iostream>
#include <chrono>
#include <array>
#include <functional> // For std::reference_wrapper

#include "../deviceArrays/headers/Support/Streamable.h"

// Forward declaration of required types (assuming they are defined elsewhere)
template<typename T> class DeviceData1d;
template<typename T> class Vec;
template<typename T> class Mat;
template<typename T> class BandedMat;
class Handle;

using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

/**
 * @brief Implements the Bi-Conjugate Gradient Stabilized (BiCGSTAB) iterative solver
 * for sparse linear systems $A\mathbf{x} = \mathbf{b}$ on the GPU using CUDA streams
 * and cuBLAS for high performance.
 *
 * The implementation uses multiple CUDA streams (a Handles group) and Handles::waitFor to overlap communication,
 * computation, and I/O operations, improving overall solver efficiency.
 *
 * @tparam T Floating-point type used for the computation (float or double).
 */
template<typename T>
class BiCGSTAB {
protected:
    /// Stream k of the algorithm runs on hands_[k]; k < numStreams.
    Handles &hands_;

    /**
     * Marks on stream 1 that stream 0 waits for later in the code.
     *
     * pReadMark is set once stream 1 has read p (h = x + alpha p); pUpdate overwrites p and must wait for it.
     * xWrittenMark is set once stream 1 has read s and omega and written x (x = h + omega s); the next
     * iteration overwrites s and omega on stream 0 and must wait for it.  The first iteration finds no
     * xWrittenMark and waits for nothing.
     */
    static constexpr size_t pReadMark = 0, xWrittenMark = 1;
private:
    const T tolerance;
    Mat<T> bHeightX7;
    SimpleArray<T> r, r_tilde, p, v, s, t, h;
    Vec<T> a9;
    Singleton<T> rho, alpha, omega, rho_new, beta;
    std::array<Singleton<T>, 4> temp;

    const size_t maxIterations;






    /**
     * @brief Synchronizes a specific CUDA stream handle.
     */
    void synch(const size_t streamInd = 0) const;

    void synchAll() const;

    /**
     * @brief Checks if the squared $L_2$ norm of a vector is smaller than the tolerance.
     *
     * Runs on stream @p streamInd and blocks the host until the norm has been read back.  There is
     * deliberately no default stream.  The caller must already have made that stream wait for
     * whatever last wrote @p v.
     *
     * @param v Vector to measure.
     * @param preAlocated Scratch singleton that receives the squared norm.
     * @param streamInd Index of the stream to run on.
     * @return True if the squared norm is below the tolerance.
     */
    bool isSmall(const Vec<T> &v, Singleton<T> preAlocated, const size_t streamInd);

    /**
     * @brief Sets the content of the destination vector to be equal to the source vector.
     */
    void set(Vec<T> &dst, const Vec<T> &src, const size_t streamInd = 0);

    /**
     * @brief Executes the BiCGSTAB P vector update: p = r + beta * (p - omega * v).
     */
    void pUpdate(const size_t streamInd = 0);

    /**
 *
 * @brief Initalizes variables r_tilde, r, b, p, and rx.ho
 */
    void preamble(Vec<T> &x);
protected:
    /**
     * The linear operation on the LHS.  Note, all such operations must take place on stream[0].
     * y <- multProduct A vec + premultResult * y
     * @param vec
     * @param product
     * @param multProduct
     * @param premultResult
     */
    virtual void mult(Vec<T>& vec, Vec<T>& product, Singleton<T> multProduct, Singleton<T> premultResult) const = 0;
public:
    const Vec<T> b;

    constexpr static size_t numStreams = 4;
    virtual ~BiCGSTAB() = default;

    /**
     * @brief Constructor for the BiCGSTAB solver.
     */
    BiCGSTAB(const Vec<T> &b, Handles &hands, Mat<T> allocatedBHeightX7, Vec<T> allocated9, T tolerance, size_t maxIterations);

    /**
     * @brief Solves the linear system $A\mathbf{x} = \mathbf{b}$ using the
     * unpreconditioned BiCGSTAB algorithm.
     * @param x The result will be placed here.  This may be the same as the b vector if you'd like to overwrite it.
     */
    void solveUnpreconditioned(Vec<T> &initGuess);
};

template<typename T>
class BCGBanded:  public BiCGSTAB<T>{
    BandedMat<T> A;

    void mult(Vec<T>& vec, Vec<T>& product, Singleton<T> multProduct,
              Singleton<T> premultResult) const override;


public:



    /**
     * @brief Static method to solve the equation $A\mathbf{x} = \mathbf{b}$.
     */
    BCGBanded(Handles &hands, BandedMat<T> A, const Vec<T> &b, Mat<T> bHeightX7, Vec<T> allocated9,
              const T &tolerance, size_t maxIterations);

    static void solve(Handles &hands, const BandedMat<T> &A, Vec<T> &result, const Vec<T> &b,
               Mat<T> allocatedBHeightX7, Vec<T> allocated9, T tolerance, size_t maxIterations);

};

template<typename T>
class BCGDense:  public BiCGSTAB<T>{
    SquareMat<T> A;

    void mult(Vec<T>& vec, Vec<T>& product, Singleton<T> multProduct,
              Singleton<T> premultResult) const override;
public:
    BCGDense(Handles &hands, SquareMat<T> A, const Vec<T> &b, Mat<T> allocatedBSizeX7, Vec<T> allocated9, T tolerance, size_t maxIterations);

    /**
     * @brief Static method to solve the equation $A\mathbf{x} = \mathbf{b}$.
     */
    static void solve(Handles &hands, const SquareMat<T> &A, Vec<T> &result, const Vec<T> &b, Mat<T> bHeightX7,
               Vec<T> allocated9, T tolerance, size_t maxIterations);

    static void test();

};


#endif // BICGSTAB_H
