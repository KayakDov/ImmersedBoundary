#include "solvers/BiCGSTAB.cuh"

#include "../deviceArrays/headers/Support/Streamable.h"


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
void BiCGSTAB<T>::hold(const size_t streamIndex, const std::initializer_list<std::reference_wrapper<Event> > evs) const {
    for (auto &ref_e: evs)
        ref_e.get().hold(hand4[streamIndex]);
}

template<typename T>
void BiCGSTAB<T>::record(size_t streamIndex, const std::initializer_list<std::reference_wrapper<Event> > evs) const {
    for (auto &ref_e: evs)
        ref_e.get().record(hand4[streamIndex]);
}

template<typename T>
void BiCGSTAB<T>::synch(const size_t streamInd) const {
    hand4[streamInd].synch();
}
template<typename T>
void BiCGSTAB<T>::synchAll() const {
    for (size_t i = 0; i < numStreams; i++) hand4[i].synch();
}

template<typename T>
bool BiCGSTAB<T>::isSmall(const Vec<T> &v, Singleton<T> preAlocated, const size_t streamInd) {
    v.mult(v, preAlocated, hand4 + streamInd);
    T vSq = preAlocated.get(hand4[streamInd]);
    return vSq < tolerance;
}

template<typename T>
void BiCGSTAB<T>::set(Vec<T> &dst, const Vec<T> &src, const size_t streamInd) {
    dst.set(src, hand4[streamInd]);
}

template<typename T>
void BiCGSTAB<T>::pUpdate(const size_t streamInd) {
    KernelPrep kp = p.kernelPrep();

    // Kernel launch performs: p = r + beta * (p - omega * v)
    updatePKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, hand4[streamInd]>>>(
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
    Handle* hand4,
    Event* events12,
    Mat<T> allocatedBHeightX7,
    Vec<T> allocated9,
    const T tolerance,
    const size_t maxIterations
) : hand4(hand4),
    tolerance(tolerance),
    alphaRAW(events12[0]), sRAW(events12[1]), pWAR(events12[2]), omegaRAW(events12[3]), rRAW(events12[4]), xRAW(events12[5]), rWAR(events12[6]), tRAW(events12[7]), tsRAW(events12[8]), betaRAW(events12[9]), rhoRAW(events12[10]), sWAR(events12[11]),
    b(b),
    bHeightX7(allocatedBHeightX7),
    r(bHeightX7.col(0)), r_tilde(bHeightX7.col(1)), p(bHeightX7.col(2)), v(bHeightX7.col(3)), s(bHeightX7.col(4)), t(bHeightX7.col(5)), h(bHeightX7.col(6)),
    a9(allocated9),
    rho(a9.get(0)), alpha(a9.get(1)), omega(a9.get(2)), rho_new(a9.get(3)), beta(a9.get(4)),
    temp{{a9.get(5), a9.get(6), a9.get(7), a9.get(8)}},
    maxIterations(maxIterations)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Algorithms.cu unpreconditionedBiCGSTAB: T must be float or double");
    bHeightX7.fill(0, hand4[0]);
    a9.fill(0, hand4[1]);
    record(1, {events12[0]});
    hold(0, {events12[0]});
}

template<typename T>
void BiCGSTAB<T>::preamble(Vec<T>& x) {
    record(0, {rWAR, sWAR, xRAW, rhoRAW});//TODO: multithread the preamble.

    set(r, b, 0);

    mult(x, r, GPUScalar<T>::get(-1), GPUScalar<T>::get(1)); // r = b - A * x

    set(r_tilde, r, 0); //r_tilde = r

    r_tilde.mult(r, rho, hand4); //rho = r_tilde * r

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
        mult(p, v); // v = A * p

        r_tilde.mult(v, alpha, hand4);
        hold(0, {rhoRAW});
        alpha.EBEPow(rho, GPUScalar<T>::get(-1), hand4[0]); //alpha = rho / (r_tilde * v)

        record(0, {alphaRAW});
        hold(1, {alphaRAW});

        set(h, x, 1);
        h.add(p, &alpha, hand4 + 1); // h = x + alpha * p
        record(1, {pWAR});

        hold(0, {xRAW, sWAR});
        s.setDifference(r, v, GPUScalar<T>::get(1), alpha, hand4); // s = r - alpha * v
        record(0, {sRAW});

        hold(2, {sRAW});
        if (isSmall(s, temp[2], 2)) {
            set(x, h, 1);
            break;
        }
        record(0, {sWAR});


        mult(s, t); // t = A * s
        record(0, {tRAW});

        hold(3, {tRAW});
        t.mult(s, temp[3], hand4+3); //temp 3 = ts
        record(3, {tsRAW});
        t.mult(t, omega, hand4); //omega = t*t
        hold(0, {tsRAW});
        omega.EBEPow(temp[3], GPUScalar<T>::get(-1), hand4[0]); //omega = t * s / t * t;
        record(0, {omegaRAW});

        hold(1, {omegaRAW});
        x.setSum(h, s, GPUScalar<T>::get(1), omega, hand4 + 1); // x = h + omega * s
        record(1, {xRAW});

        hold(0, {rWAR});
        r.setDifference(s, t, GPUScalar<T>::get(1), omega, hand4); // r = s - omega * t
        record(0, {rRAW});

        hold(2, {rRAW});
        if (isSmall(r, temp[2])) break;
        record(2, {rWAR});

        r_tilde.mult(r, rho_new, hand4);
        beta.setProductOfQuotients(rho_new, rho, alpha, omega, hand4[0]); // beta = (rho_new / rho) * (alpha / omega);
        record(0, {betaRAW});

        hold(3, {betaRAW});
        set(rho, rho_new, 3);
        record(3, {rhoRAW});

        hold(0, {pWAR});
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
BCGBanded<T>::BCGBanded(Handle* hand4, BandedMat<T> A, const Vec<T> &b, Event* events11, Mat<T> bHeightX7, Vec<T> allocated9, const T &tolerance,
size_t maxIterations): BiCGSTAB<T>(b, hand4, events11, bHeightX7, allocated9, tolerance, maxIterations), A(A){
}

template<typename T>
void BCGBanded<T>::mult(Vec<T>& vec, Vec<T>& product, Singleton<T> multProduct, Singleton<T> preMultResult) const {
    return A.bandedMult(vec, product, this->hand4, multProduct, preMultResult);
}


template<typename T>
void BCGBanded<T>::solve(
    Handle* hand4,
    const BandedMat<T> &A,
    Vec<T>& result,
    const Vec<T> &b,
    Event* events11,
    Mat<T> allocatedBHeightX7,
    Vec<T> allocated9,
    const T tolerance,
    const size_t maxIterations
) {
    BCGBanded<T> solver(hand4, A, b, events11, allocatedBHeightX7, allocated9, tolerance, maxIterations);
    solver.solveUnpreconditioned(result);
}

template<typename T>
void BCGDense<T>::mult(Vec<T> &vec, Vec<T> &product, Singleton<T> multProduct, Singleton<T> premultResult) const {
    A.mult(vec, product, this->hand4, &multProduct, &premultResult, false);
}

template<typename T>
BCGDense<T>::BCGDense(Handle *hand4, SquareMat<T> A, const Vec<T> &b, Event* events11, Mat<T> allocatedBSizeX7, Vec<T> allocated9, T tolerance, size_t maxIterations): BiCGSTAB<T>(b, hand4, events11, allocatedBSizeX7, allocated9, tolerance, maxIterations), A(A) {

}

template<typename T>
void BCGDense<T>::solve(Handle *hand4, const SquareMat<T> &A, Vec<T> &result, const Vec<T> &b, Event* events11, Mat<T> bHeightX7, Vec<T> allocated9, T tolerance, size_t maxIterations) {
    BCGDense<T> solver(hand4, A, b, events11, bHeightX7, allocated9, tolerance, maxIterations);
    solver.solveUnpreconditioned(result);
}

template class BiCGSTAB<double>;
template class BiCGSTAB<float>;

template class BCGBanded<double>;
template class BCGBanded<float>;

template class BCGDense<double>;
template class BCGDense<float>;