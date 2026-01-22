#ifndef BICGSTAB_H
#define BICGSTAB_H

#include "deviceArrays/headers/BandedMat.h"

#include "solvers/Event.h"
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
protected:
    Handle* hand4;
private:
    const T tolerance;
    Event alphaReady, sReady, hReady, omegaReady, rReady, xReady, prodTS;
    const Vec<T> b;
    Mat<T> bHeightX7;
    Vec<T> r, r_tilde, p, v, s, t, h;
    Vec<T> a9;
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
    void preamble(Vec<T> &x);
protected:
    virtual void mult(Vec<T>& vec, Vec<T>& product, Handle& hand, Singleton<T> multProduct = Singleton<T>::ONE, Singleton<T> premultResult = Singleton<T>::ZERO) const = 0;
public:
    virtual ~BiCGSTAB() = default;

    /**
     * @brief Constructor for the BiCGSTAB solver.
     */
    explicit BiCGSTAB(
        const Vec<T> &b,
        Handle* hand4,
        Mat<T> *allocatedBHeightX7 = nullptr,
        Vec<T> *allocated9 = nullptr,
        T tolerance = std::is_same_v<T, double> ? T(1e-15) : T(1e-6),
        size_t maxIterations = 1500
    );

    /**
     * @brief Solves the linear system $A\mathbf{x} = \mathbf{b}$ using the
     * unpreconditioned BiCGSTAB algorithm.
     * @param x The result will be placed here.  This may be the same as the b vector if you'd like to overwrite it.
     */
    void solveUnpreconditionedBiCGSTAB(Vec<T>& x);
};

template<typename T>
class BCGBanded:  public BiCGSTAB<T>{
    BandedMat<T> A;

    void mult(Vec<T>& vec, Vec<T>& product, Handle& hand, Singleton<T> multProduct,
              Singleton<T> premultResult) const override;
public:
    BCGBanded(Handle *hand4, BandedMat<T> A, const Vec<T> &b, Mat<T> *allocatedBSizeX7, Vec<T> *allocated9, const T &tolerance, size_t maxIterations);


    /**
     * @brief Static method to solve the equation $A\mathbf{x} = \mathbf{b}$.
     */
    static void solve(
        Handle* hand4,
        const BandedMat<T> &A,
        Vec<T>& result,
        const Vec<T> &b,
        Mat<T> *allocatedSizeX7 = nullptr,
        Vec<T> *allocated9 = nullptr,
        T tolerance = std::is_same_v<T, double> ? T(1e-15) : T(1e-6),
        size_t maxIterations = 1500
    );


};


#endif // BICGSTAB_H
