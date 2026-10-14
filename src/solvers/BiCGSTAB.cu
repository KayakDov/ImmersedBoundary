#include "solvers/BiCGSTAB.cuh"

#include "../deviceArrays/headers/Support/Streamable.h"
#include <stdexcept>
#include <string>


using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;


template<typename T>
__global__ void updatePKernel(
    DeviceData1d<T> p,
    const DeviceData1d<T> r,
    const DeviceData1d<T> v,
    const T *__restrict__ beta,
    const T *__restrict__ omega) {
    if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < p.cols)
        p[idx] = r[idx] + *beta * (p[idx] - *omega * v[idx]);
}


template<typename T>
void BiCGSTAB<T>::synch(const size_t streamInd) const {
    hands_[streamInd].synch();
}
template<typename T>
void BiCGSTAB<T>::synchAll() const {
    for (size_t i = 0; i < numStreams; i++) hands_[i].synch();
}

template<typename T>
bool BiCGSTAB<T>::isSmall(const Vec<T> &v, Singleton<T> preAlocated, const size_t streamInd) {
    v.mult(v, preAlocated, &hands_[streamInd]);
    T vSq = preAlocated.get(hands_[streamInd]);
    return vSq < tolerance;
}

template<typename T>
void BiCGSTAB<T>::set(Vec<T> &dst, const Vec<T> &src, const size_t streamInd) {
    dst.set(src, hands_[streamInd]);
}

template<typename T>
void BiCGSTAB<T>::pUpdate(const size_t streamInd) {
    KernelPrep kp = p.kernelPrep();

    // Kernel launch performs: p = r + beta * (p - omega * v)
    updatePKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hands_[streamInd]>>>(
        p.toKernel1d(), // d_p (Input/Output)
        r.toKernel1d(), // d_r
        v.toKernel1d(), // d_v
        beta.data(), // d_beta (Device pointer from Singleton)
        omega.data() // d_omega (Device pointer from Singleton)
    );
}

template<typename T>
BiCGSTAB<T>::BiCGSTAB(
    const Vec<T> &b,
    Handles& hands,
    Mat<T> allocatedBHeightX7,
    Vec<T> allocated9,
    const T tolerance,
    const size_t maxIterations
) : hands_(hands),
    tolerance(tolerance),
    b(b),
    bHeightX7(allocatedBHeightX7),
    r(bHeightX7.col(0)), r_tilde(bHeightX7.col(1)), p(bHeightX7.col(2)), v(bHeightX7.col(3)), s(bHeightX7.col(4)), t(bHeightX7.col(5)), h(bHeightX7.col(6)),
    a9(allocated9),
    rho(a9.get(0)), alpha(a9.get(1)), omega(a9.get(2)), rho_new(a9.get(3)), beta(a9.get(4)),
    temp{{a9.get(5), a9.get(6), a9.get(7), a9.get(8)}},
    maxIterations(maxIterations)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Algorithms.cu unpreconditionedBiCGSTAB: T must be float or double");
    if (hands.size() < numStreams)
        throw std::invalid_argument("BiCGSTAB needs at least " + std::to_string(numStreams) + " handles.");
    hands_.waitFor(1, 0);
}

template<typename T>
void BiCGSTAB<T>::preamble(Vec<T>& x) {
    //TODO: multithread the preamble.

    set(r, b, 0);

    mult(x, r, GPUScalar<T>::get(-1, hands_[0]), GPUScalar<T>::get(1, hands_[0])); // r = b - A * x

    set(r_tilde, r, 0); //r_tilde = r

    r_tilde.mult(r, rho, &hands_[0]); //rho = r_tilde * r

    set(p, r, 0);
}

template<typename T>
void BiCGSTAB<T>::solveUnpreconditioned(Vec<T>& initGuess) {
    synch();
    TimePoint start = std::chrono::steady_clock::now();

    auto& x = initGuess;
    preamble(x);

    size_t i = 0;
    for (; i < maxIterations; i++) {
        mult(p, v, GPUScalar<T>::get(1, hands_[0]), GPUScalar<T>::get(0, hands_[0])); // v = A * p

        r_tilde.mult(v, alpha, &hands_[0]);
        hands_.waitFor(3, 0); // rho was set on stream 3 at the end of the last iteration.
        alpha.EBEPow(rho, GPUScalar<T>::get(-1, hands_[0]), hands_[0]); //alpha = rho / (r_tilde * v)

        hands_.waitFor(0, 1);

        set(h, x, 1);
        h.add(p, &alpha, &hands_[1]); // h = x + alpha * p
        hands_.markCompletion(1, pReadMark);

        hands_.waitForMark(1, 0, xWrittenMark);
        s.setDifference(r, v, GPUScalar<T>::get(1, hands_[0]), alpha, &hands_[0]); // s = r - alpha * v

        hands_.waitFor(0, 2);
        if (isSmall(s, temp[2], 2)) {
            set(x, h, 1);
            break;
        }

        mult(s, t, GPUScalar<T>::get(1, hands_[0]), GPUScalar<T>::get(0, hands_[0])); // t = A * s

        hands_.waitFor(0, 3);
        t.mult(s, temp[3], &hands_[3]); //temp 3 = ts
        t.mult(t, omega, &hands_[0]); //omega = t*t
        hands_.waitFor(3, 0);
        omega.EBEPow(temp[3], GPUScalar<T>::get(-1, hands_[0]), hands_[0]); //omega = t * s / t * t;

        hands_.waitFor(0, 1);
        x.setSum(h, s, GPUScalar<T>::get(1, hands_[1]), omega, &hands_[1]); // x = h + omega * s
        hands_.markCompletion(1, xWrittenMark);

        hands_.waitFor(2, 0); // Nothing on stream 2 may still be reading r.
        r.setDifference(s, t, GPUScalar<T>::get(1, hands_[0]), omega, &hands_[0]); // r = s - omega * t

        hands_.waitFor(0, 2);
        if (isSmall(r, temp[2])) break;

        r_tilde.mult(r, rho_new, &hands_[0]);
        beta.setProductOfQuotients(rho_new, rho, alpha, omega, hands_[0]); // beta = (rho_new / rho) * (alpha / omega);

        hands_.waitFor(0, 3);
        set(rho, rho_new, 3);

        hands_.waitForMark(1, 0, pReadMark);
        pUpdate(); // p = p - beta * omega * v
    }
    if (i >= maxIterations)
        std::cout << "WARNING: Maximum number of iterations reached.  Convergence failed.";

    synchAll();

    const TimePoint end = std::chrono::steady_clock::now();
    const double time = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();
    // std::cout<< "BiCGSTAB #iterations = " << i << " with tolderance = " << tolerance << std::endl;
    // std::cout << time << ", ";
}


template<typename T>
BCGBanded<T>::BCGBanded(Handles& hands, BandedMat<T> A, const Vec<T> &b, Mat<T> bHeightX7, Vec<T> allocated9, const T &tolerance,
size_t maxIterations): BiCGSTAB<T>(b, hands, bHeightX7, allocated9, tolerance, maxIterations), A(A){
}

template<typename T>
void BCGBanded<T>::mult(Vec<T>& vec, Vec<T>& product, Singleton<T> multProduct, Singleton<T> preMultResult) const {
    return A.bandedMult(vec, product, &this->hands_[0], multProduct, preMultResult);
}


template<typename T>
void BCGBanded<T>::solve(
    Handles& hands,
    const BandedMat<T> &A,
    Vec<T>& result,
    const Vec<T> &b,
    Mat<T> allocatedBHeightX7,
    Vec<T> allocated9,
    const T tolerance,
    const size_t maxIterations
) {
    BCGBanded<T> solver(hands, A, b, allocatedBHeightX7, allocated9, tolerance, maxIterations);
    solver.solveUnpreconditioned(result);
}

template<typename T>
void BCGDense<T>::mult(Vec<T> &vec, Vec<T> &product, Singleton<T> multProduct, Singleton<T> premultResult) const {
    A.mult(vec, product, &this->hands_[0], &multProduct, &premultResult, false);
}

template<typename T>
BCGDense<T>::BCGDense(Handles& hands, SquareMat<T> A, const Vec<T> &b, Mat<T> allocatedBSizeX7, Vec<T> allocated9, T tolerance, size_t maxIterations): BiCGSTAB<T>(b, hands, allocatedBSizeX7, allocated9, tolerance, maxIterations), A(A) {

}

template<typename T>
void BCGDense<T>::solve(Handles& hands, const SquareMat<T> &A, Vec<T> &result, const Vec<T> &b, Mat<T> bHeightX7, Vec<T> allocated9, T tolerance, size_t maxIterations) {
    BCGDense<T> solver(hands, A, b, bHeightX7, allocated9, tolerance, maxIterations);
    solver.solveUnpreconditioned(result);
}

template class BiCGSTAB<double>;
template class BiCGSTAB<float>;

template class BCGBanded<double>;
template class BCGBanded<float>;

template class BCGDense<double>;
template class BCGDense<float>;