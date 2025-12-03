
#include "../deviceArrays/headers/Singleton.h"
#include "Event.h"
#include <iostream>
#include <chrono>
#include <array>

#include "deviceArrays/headers/Streamable.h"


using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;


/**
 * @brief Implements the full BiCGSTAB update for the P (search direction) vector:
 *
 * p_new = r + beta * (p_old - omega * v)
 *
 * This operation is performed in place on d_p, which holds p_old.
 *
 * @tparam T Floating point type.
 * @param p The P vector (input/output).
 * @param r The R residual vector (input).
 * @param v The V vector (input).
 * @param beta Device pointer to the scalar beta (input).
 * @param omega Device pointer to the scalar omega (input).
 * @param N The size of the vectors.
 */
template <typename T>
__global__ void updatePKernel(
    DeviceData1d<T> p,
    const DeviceData1d<T> r,
    const DeviceData1d<T> v,
    const T* __restrict__ beta,
    const T* __restrict__ omega)
{
    if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < p.cols)
        p[idx] = r[idx] + *beta * (p[idx] - *omega * v[idx]);
}

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
template <typename T>
class BiCGSTAB{
private:
    const T tolerance;
    Handle handle[4]{};
    Event alphaReady, sReady, hReady, omegaReady, rReady, xReady, prodTS;
    const Vec<T> b;
    Mat<T> paM;
    Vec<T> r, r_tilde, p, v, s, t, h;
    Vec<T> paV;
    Singleton<T> rho, alpha, omega, rho_new, beta;
    std::array<Singleton<T>, 4> temp;//a temporary place for a single value on each stream.

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
    bool isSmall(const Vec<T>& v,  Singleton<T> preAlocated, const size_t streamInd){

            v.mult(v, preAlocated, handle + streamInd);
            T vSq = preAlocated.get(handle[streamInd]);
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
    void set(Vec<T>& dst, const Vec<T>& src, const size_t streamInd){
        dst.set(src, handle[streamInd]);
    }

    /**
     * @brief Executes the BiCGSTAB P vector update: p = r + beta * (p - omega * v).
     * * This is a single, highly efficient kernel launch using internal BiCGSTAB state vectors.
     *
     * @param[in] streamInd The index of the stream handle to perform the operation on.
     */
    void pUpdate(const size_t streamInd) {
        KernelPrep kp = p.kernelPrep();

        // Kernel launch performs: p = r + beta * (p - omega * v)
        updatePKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, handle[streamInd]>>>(
            p.toKernel1d(),       // d_p (Input/Output)
            r.toKernel1d(),       // d_r
            v.toKernel1d(),       // d_v
            beta.data(),            // d_beta (Device pointer from Singleton)
            omega.data()           // d_omega (Device pointer from Singleton)
        );
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
     * Should have at least 7 columns and b.size() rows.
     */
    explicit BiCGSTAB(
        const Vec<T>& b,
        Mat<T>* preAllocated = nullptr,
        const T tolerance = std::is_same_v<T,double> ? T(1e-15) : T(1e-6),
        const size_t maxIterations = 1500
        ):tolerance(tolerance),
      b(b),
      paM(preAllocated ? *preAllocated : Mat<T>::create(b.size(), 7)),
      r(paM.col(0)), r_tilde(paM.col(1)), p(paM.col(2)), v(paM.col(3)), s(paM.col(4)), t(paM.col(5)), h(paM.col(6)),
      paV(Vec<T>::create(9, handle[0])),
      rho(paV.get(0)), alpha(paV.get(1)), omega(paV.get(2)), rho_new(paV.get(3)), beta(paV.get(4)), temp{{paV.get(5), paV.get(6), paV.get(7), paV.get(8)}},
      maxIterations(maxIterations)
    {
        static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>,
                "Algorithms.cu unpreconditionedBiCGSTAB: T must be float or double");
        cudaDeviceSynchronize();
    }

    static void solve(
        const BandedMat<T>& A,
        Vec<T>& x,
        const Vec<T>& b,
        Mat<T>* preAllocated = nullptr,
        const T tolerance = std::is_same_v<T,double> ? T(1e-15) : T(1e-6),
        const size_t maxIterations = 1500
    ) {
        BiCGSTAB solver(b, preAllocated, tolerance, maxIterations);
        solver.solve(A, x);
    }

    static void solve(
        const T* A,
        const T* aLd,
        const size_t* inds,
        const size_t indsStride,
        const size_t numInds,
        const T* x,
        const size_t xStride,
        const T* b,
        const size_t bStride,
        const size_t bSize,
        T* prealocated, //size x 7 matrix
        const size_t prealocatedLd,
        size_t maxIterations = 1500,
        T tolerance = std::is_same_v<T,double> ? T(1e-15) : T(1e-6)
    ){
        Mat<T> preAlocatedMat = Mat<T>::create(bSize, 7, prealocatedLd, prealocated);
        Vec<T> bVec = Vec<T>::create(bSize, b, bStride);
        auto ABanded = BandedMat<T>::create(bSize, numInds, aLd, A, inds, indsStride);//rows cols pointer VecIndices
        auto xVec = Vec<T>::create(bSize, b, xStride);
        solve(ABanded, xVec, bVec, &preAlocatedMat, tolerance, maxIterations);
    }

    /**
     *
     * @brief Initalizes variables r_tilde, r, b, p, and rx.ho
     * @param A The left-hand side of the equation Ax = b
     * @param x The x of the equation Ax = b
     */
    void preamable(const BandedMat<T>& A, Vec<T>& x) {

        x.fillRandom(&handle[0]); // set x randomly
        // x.fill(1, handle[0]);

        record(0, xReady);

        set(r, b, 0);
        A.bandedMult(x, r, handle, Singleton<T>::MINUS_ONE, Singleton<T>::ONE); // r = b - A * x

        set(r_tilde, r, 0); //r_tilde = r

        r_tilde.mult(r, rho, handle); //rho = r_tilde * r

        set(p, r, 0);

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
    Vec<T> solveUnpreconditionedBiCGSTAB(const BandedMat<T>& A, Vec<T>& x){

        cudaDeviceSynchronize();
        TimePoint start = std::chrono::steady_clock::now();
        preamable(A, x);

        size_t numIterations = 0;
        for(;numIterations < maxIterations; numIterations++) {

            A.bandedMult(p, v, handle); // v = A * p

            r_tilde.mult(v, alpha, handle);
            alpha.EBEPow(rho, Singleton<T>::MINUS_ONE, handle[0]); //alpha = rho / (r_tilde * v)
            record(0, {alphaReady});

            synch(1);
            omegaReady.renew();
            wait(1, {alphaReady});

            set(h, x, 1);
            synch(1);
            renew({alphaReady, xReady});
            h.add(p, &alpha, handle + 1); // h = x + alpha * p

            s.setDifference(r, v, Singleton<T>::ONE, alpha, handle); // s = r - alpha * v
            record(0, {sReady});

            wait(2, {sReady, hReady});
            if(isSmall(s, temp[2], 2)) {
                set(x, h, 2);
                break;
            }
            renew({sReady, hReady});

            A.bandedMult(s, t, handle); // t = A * s

            t.mult(s, temp[3], handle + 3);
            record(3, {prodTS});
            t.mult(t, omega, handle);
            wait(0, {prodTS});
            omega.EBEPow(temp[3], Singleton<T>::MINUS_ONE, handle[0]); //omega = t * s / t * t;

            record(0, {omegaReady});

            wait(1, {omegaReady});
            x.setSum(h, s, Singleton<T>::ONE, omega, handle + 1); // x = h + omega * s
            record(1, {xReady});

            synch(0);
            prodTS.renew();
            r.setDifference(s, t, Singleton<T>::ONE, omega, handle); // r = s - omega * t
            record(0, {rReady});

            wait(2, {xReady, rReady});

            if(isSmall(r, temp[2],2)) break;
            rReady.renew();

            r_tilde.mult(r, rho_new, handle);

            beta.setProductOfQuotients(rho_new, rho, alpha, omega, handle[0]); // beta = (rho_new / rho) * (alpha / omega);

            set(rho, rho_new, 0);

            pUpdate(0); // p = p - beta * omega * v


        }
        cudaDeviceSynchronize();
        const TimePoint end = std::chrono::steady_clock::now();
        const double time = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();

        if (numIterations >= maxIterations) std::cout << "WARNING: Maximum number of iterations reached.  Convergence failed.";
        std::cout << time << ", ";

        return x;
    }

};


//
// void testBiCGSTAB() {
//     Handle hand;
//
//     size_t denseMatDim = 4;
//
//     auto indices = Vec<int32_t>::create(2);
//     indices.get(0).set(1, hand);
//     indices.get(1).set(-1, hand);
//
//     auto bm = BandedMat<double>::create(denseMatDim, 2,indices);
//     bm.col(0).fill(3, hand);
//     bm.col(1).fill(-2, hand);
//
//     auto b = Vec<double>::create(denseMatDim, hand);
//     b.fill(1, hand);
//
//     auto x = Vec<double>::create(denseMatDim, hand);
//
//     BiCGSTAB<double> bs(b);
//
//     bs.solveUnpreconditionedBiCGSTAB(bm, x);
//
//     std::cout << "x = \n" << GpuOut<double>(x, hand) << std::endl;
// }
