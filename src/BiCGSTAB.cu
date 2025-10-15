#include "deviceArrays/headers/deviceArrays.h"
#include "deviceArrays/headers/singleton.h"
#include <cmath>
#include "Event.h"
#include <iostream>
#include <chrono>


using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;


/**
 * @brief Implements the Bi-Conjugate Gradient Stabilized (BiCGSTAB) iterative solver
 * for sparse linear systems $A\mathbf{x} = \mathbf{b}$ on the GPU using CUDA streams
 * and cuBLAS for high performance.
 *
 * The implementation uses multiple CUDA streams and events to overlap communication,
 * computation, and I/O operations, improving overall solver efficiency.
 *
 * @tparam Float Floating-point type used for the computation (float or double).
 */
template <typename Float>
class BiCGSTAB{
private:
    const Float tolerance;
    Handle handle[4]{};
    Event alphaReady, sReady, hReady, omegaReady, rReady, xReady, prodTS;
    const Vec<Float> b;
    Mat<Float> paM;
    Vec<Float> r, r_tilde, p, v, s, t, h;
    Vec<Float> paV;
    Singleton<Float> rho, alpha, omega, rho_new, beta, temp;

    const size_t maxIterations;

    /**
     * @brief Waits for a list of events to complete on a specified CUDA stream.
     *
     * @param[in] streamIndex The index of the stream handle to wait on.
     * @param[in] evs A list of Event references to wait for.
     */
    void wait(const size_t streamIndex, const std::initializer_list<std::reference_wrapper<Event>> evs) const{
        for (auto& ref_e : evs)
            ref_e.get().wait(handle[streamIndex]);

    }

    /**
    * @brief Renews (resets) a list of events, preparing them for the next iteration.
    *
    * @param[in] evs A list of Event references to renew.
    */
    static void renew(const std::initializer_list<std::reference_wrapper<Event>> evs) {
        for (auto& ref_e : evs)
            ref_e.get().renew();
    }

    /**
    * @brief Records an event on a specified CUDA stream.
    *
    * @param[in] streamIndex The index of the stream handle to record the event on.
    * @param[out] e The Event object to record.
    */
    void record(size_t streamIndex, Event& e) const {
        e.record(handle[streamIndex]);
    }

    /**
     * @brief Synchronizes a specific CUDA stream handle.
     *
     * @param[in] streamInd The index of the stream handle to synchronize.
     */
     void synch(const size_t streamInd) const{
        handle[streamInd].synch();
    }

    /**
     * @brief Checks if the squared $L_2$ norm of a vector is smaller than the tolerance.
     *
     * @param[in] v The vector to check.
     * @param[in] streamInd The stream index to perform the operation on.
     * @return true if $\|\mathbf{v}\|^2 < \text{tolerance}$, false otherwise.
     */
    bool isSmall(const Vec<Float>& v, const size_t streamInd){
        Singleton<Float> vSquared = Singleton<Float>::create();
            v.mult(v, &vSquared, handle + streamInd);
            Float vSq = vSquared.get(handle[streamInd].stream);
            synch(streamInd);
            return vSq < tolerance;
    }

    /**
     * @brief Sets the content of the destination vector to be equal to the source vector.
     *
     * @param[out] dst The destination vector.
     * @param[in] src The source vector.
     * @param[in] streamInd The stream index to perform the copy on.
     */
    void set(Vec<Float>& dst, const Vec<Float>& src, const size_t streamInd){
        dst.set(src, handle[streamInd].stream);
    }
    /**
     * @brief Computes the quotient of two Singletons: $\text{dst} = \text{numerator} / \text{denom}$.
     *
     * @param[out] dst The Singleton to store the result.
     * @param[in] numerator The numerator Singleton.
     * @param[in] denom The denominator Singleton.
     * @param[in] streamInd The stream index to perform the operation on.
     */
    void setQuotient(Singleton<Float>& dst, const Singleton<Float>& numerator, const Singleton<Float>& denom, const size_t streamInd){
        dst.set(denom, handle[streamInd].stream);
        dst.EBEPow(numerator, Singleton<Float>::MINUS_ONE, handle[streamInd].stream);
    }

public:
    /**
     * @brief Constructor for the BiCGSTAB solver.
     *
     * Initializes state vectors, scalar singletons, stream handles, and sets
     * the pointer mode for cuBLAS to CUBLAS_POINTER_MODE_DEVICE.
     *
     * @param[in] b The right-hand side vector of the linear system.
     * @param[in] tolerance The stopping tolerance (defaults to $10^{-12}$ for double, $10^{-6}$ for float).
     * @param[in] maxIterations Maximum number of iterations (defaults to $5 \times$ vector size).
     * @param[in] preAllocated Optional pointer to a pre-allocated matrix for internal vectors.
     */
    explicit BiCGSTAB(
        const Vec<Float>& b,
        Float tolerance = std::is_same_v<Float,double> ? Float(1e-12) : Float(1e-6),
        size_t maxIterations = 0,
        Mat<Float>* preAllocated = nullptr
    ):tolerance(tolerance),
      b(b),
      paM(preAllocated ? *preAllocated : Mat<Float>::create(b.size(), 7)),
      r(paM.col(0)), r_tilde(paM.col(1)), p(paM.col(2)), v(paM.col(3)), s(paM.col(4)), t(paM.col(5)), h(paM.col(6)),
      paV(Vec<Float>::create(6, handle[0].stream)),
      rho(paV.get(0)), alpha(paV.get(1)), omega(paV.get(2)), rho_new(paV.get(3)), beta(paV.get(4)), temp(paV.get(5)),
      maxIterations(maxIterations <= 0 ? 5*b.size() : maxIterations)
    {
        static_assert(std::is_same_v<Float,float> || std::is_same_v<Float,double>,
                "Algorithms.cu unpreconditionedBiCGSTAB: T must be float or double");
        cudaDeviceSynchronize();
        for (const auto& h : handle)
            cublasSetPointerMode(h.handle, CUBLAS_POINTER_MODE_DEVICE);
    }

    /**
     * @brief Fills preaolcated memory of length size with random values.  If no p
     * @param x Pre-allocated memory that will be filled with random values.  Pass nullptr to have this method allocate
     * memory for you.
     * @return A random initial guess for x.
     */
    Vec<Float> initialGuess(Vec<Float>* x) {
        Vec<Float> result = x ? *x : Vec<Float>::create(b.size(), handle[0].stream);
        result.fillRandom(&handle[0]); // set x randomly
        record(0, xReady);
        return result;
    }

    /**
     * @brief Initalizes variables r_tilde, r, b, p, and rho
     * @param A The left hand side of the equation Ax = b
     * @param result The x of the equation Ax = b
     */
    void preamable(BandedMat<Float>& A, Vec<Float>& result) {
        r_tilde.fillRandom(&handle[0]); // set r_tilde randomly

        set(r, b, 0);
        A.mult(result, &r, handle, &Singleton<Float>::MINUS_ONE, &Singleton<Float>::ONE); // r = b - A * x

        set(p, r, 0);

        r_tilde.mult(r, &rho, handle);

        wait(1, {xReady});
    }

    /**
     * @brief Solves the linear system $A\mathbf{x} = \mathbf{b}$ using the
     * unpreconditioned BiCGSTAB algorithm.
     *
     * Initializes the state, enters the iterative loop, performs matrix-vector
     * products and vector updates asynchronously across multiple streams, and
     * checks for convergence.
     *
     * @param[in] A The system matrix, provided as a BandedMat object.
     * @param[in,out] x Optional pointer to the initial guess vector. If nullptr, a random
     * initial guess is generated and returned as the result.
     * @return The final solution vector $\mathbf{x}$.
     */
    Vec<Float> solveUnpreconditionedBiCGSTAB(const BandedMat<Float>& A, Vec<Float>* x = nullptr){

        Vec<Float> result = initialGuess(x);

        preamable(A, result);

        double totalTime = 0;
        size_t numIterations = 0;
        for(;numIterations < maxIterations; numIterations++) {

            TimePoint start = std::chrono::steady_clock::now();

            A.mult(p, &v, handle, &Singleton<Float>::ONE, &Singleton<Float>::ZERO); // v = A * p

            r_tilde.mult(v, &alpha, handle);
            alpha.EBEPow(rho, Singleton<Float>::MINUS_ONE, handle[0].stream); //alpha = rho / (r_tilde * v)
            record(0, {alphaReady});

            synch(1);
            omegaReady.renew();
            wait(1, {alphaReady});

            set(h, result, 1);
            synch(1);
            renew({alphaReady, xReady});
            h.add(p, &alpha, handle + 1); // h = x + alpha * p

            temp.set(alpha, handle[0].stream);
            temp.mult(Singleton<Float>::MINUS_ONE, handle);
            s.setSum(r, v, &Singleton<Float>::ONE, &temp, handle); // s = r - alpha * v
            record(0, {sReady});

            set(result, h, 1);

            wait(2, {sReady, hReady});
            if(isSmall(s, 2)) break;
            renew({sReady, hReady});

            A.mult(s, &t, handle); // t = A * s

            t.mult(s, &temp, handle + 3);
            record(3, {prodTS});
            t.mult(t, &omega, handle);
            wait(0, {prodTS});
            omega.EBEPow(temp, Singleton<Float>::MINUS_ONE, handle[0].stream); //omega = t * s / t * t;

            record(0, {omegaReady});

            wait(1, {omegaReady});
            result.add(s, &omega, handle + 1); // x = h + omega * s
            record(1, {xReady});

            synch(0);
            prodTS.renew();
            temp.set(omega, handle[0].stream);
            temp.mult(Singleton<Float>::MINUS_ONE, handle);
            r.setSum(s, t, &Singleton<Float>::ONE, &temp, handle); // r = s - omega * t
            record(0, {rReady});

            wait(2, {xReady, rReady});

            if(isSmall(r, 2)) break;
            rReady.renew();

            r_tilde.mult(r, &rho_new, handle);

            setQuotient(temp, rho_new, rho, 0);
            setQuotient(beta, alpha, omega, 0);
            beta.EBEPow(temp, Singleton<Float>::ONE, handle[0].stream); // beta = (rho_new / rho) * (alpha / omega);

            set(rho, rho_new, 0);

            p.mult(beta, handle);
            p.add(r, &Singleton<Float>::ONE, handle); // p = r + beta * p
            temp.set(beta, handle[0].stream);
            temp.mult(omega, handle);
            p.sub(v, &temp, handle); // p = p - beta * omega * v

            TimePoint end = std::chrono::steady_clock::now();
            double iterationTime = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();
            totalTime += iterationTime;
        }

        std::cout << "algorithms.cu unpreconditionedBiCGSTAB Number of iterations: " << numIterations << std::endl;
        std::cout << "algorithms.cu unpreconditionedBiCGSTAB Average time per iteration in milliseconds: " << totalTime / numIterations << std::endl;
        
        return result;
    }

};

