#include "solvers/BiCGSTAB.cuh"

#include "deviceArrays/headers/Streamable.h"


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
    Mat<T> *allocatedBHeightX7,
    Vec<T> *allocated9,
    const T tolerance,
    const size_t maxIterations
) : tolerance(tolerance),
    b(b),
    hand4(hand4),
    bHeightX7(allocatedBHeightX7 ? *allocatedBHeightX7 : Mat<T>::create(b.size(), 7)),
    r(bHeightX7.col(0)), r_tilde(bHeightX7.col(1)), p(bHeightX7.col(2)), v(bHeightX7.col(3)), s(bHeightX7.col(4)), t(bHeightX7.col(5)), h(bHeightX7.col(6)),
    a9(allocated9 ? *allocated9 : Vec<T>::create(9, hand4[0])),
    rho(a9.get(0)), alpha(a9.get(1)), omega(a9.get(2)), rho_new(a9.get(3)), beta(a9.get(4)), temp{{a9.get(5), a9.get(6), a9.get(7), a9.get(8)}},
    maxIterations(maxIterations) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "Algorithms.cu unpreconditionedBiCGSTAB: T must be float or double");
}

template<typename T>
void BiCGSTAB<T>::preamble(Vec<T>& x) {
    record(0, {rWAR, xRAW, rhoRAW});//TODO: multithread the preamble.

    set(r, b, 0);

    mult(x, r, Singleton<T>::MINUS_ONE, Singleton<T>::ONE); // r = b - A * x

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

    size_t iteration = 0;
    for (; iteration < maxIterations; iteration++) {
        mult(p, v); // v = A * p

        r_tilde.mult(v, alpha, hand4);
        alpha.EBEPow(rho, Singleton<T>::MINUS_ONE, hand4[0]); //alpha = rho / (r_tilde * v)

        set(h, x);
        h.add(p, &alpha, hand4); // h = x + alpha * p

        s.setDifference(r, v, Singleton<T>::ONE, alpha, hand4); // s = r - alpha * v

        if (isSmall(s, temp[2])) {
            set(x, h);
            break;
        }

        mult(s, t); // t = A * s

        t.mult(s, temp[3], hand4);
        t.mult(t, omega, hand4);
        omega.EBEPow(temp[3], Singleton<T>::MINUS_ONE, hand4[0]); //omega = t * s / t * t;

        x.setSum(h, s, Singleton<T>::ONE, omega, hand4); // x = h + omega * s

        r.setDifference(s, t, Singleton<T>::ONE, omega, hand4); // r = s - omega * t

        if (isSmall(r, temp[2])) break;

        r_tilde.mult(r, rho_new, hand4);
        beta.setProductOfQuotients(rho_new, rho, alpha, omega, hand4[0]); // beta = (rho_new / rho) * (alpha / omega);

        set(rho, rho_new);
        pUpdate(); // p = p - beta * omega * v
    }
    if (iteration >= maxIterations)
        std::cout << "WARNING: Maximum number of iterations reached.  Convergence failed.";

    synch();

    const TimePoint end = std::chrono::steady_clock::now();
    const double time = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();
    // std::cout<< "BiCGSTAB #iterations = " << iteration << std::endl;
    // std::cout << time << ", ";
}

template<typename T>
void BiCGSTAB<T>::solveUnconditionedMultiStream(Vec<T>& initGuess) {
    synch();
    TimePoint start = std::chrono::steady_clock::now();

    auto& x = initGuess;
    preamble(x);

    size_t i = 0;
    for (; i < maxIterations; i++) {
        mult(p, v); // v = A * p

        r_tilde.mult(v, alpha, hand4);
        hold(0, {rhoRAW});
        alpha.EBEPow(rho, Singleton<T>::MINUS_ONE, hand4[0]); //alpha = rho / (r_tilde * v)

        record(0, {alphaRAW});
        hold(1, {alphaRAW});

        set(h, x, 1);
        h.add(p, &alpha, hand4 + 1); // h = x + alpha * p
        record(1, {hRAW});


        s.setDifference(r, v, Singleton<T>::ONE, alpha, hand4); // s = r - alpha * v
        record(0, {sRAW});

        hold(2, {sRAW});
        if (isSmall(s, temp[2], 2)) {
            set(x, h, 1);
            break;
        }

        mult(s, t); // t = A * s
        record(0, {tRAW});

        hold(3, {tRAW});
        t.mult(s, temp[3], hand4+3); //temp 3 = ts
        record(3, {tsRAW});
        t.mult(t, omega, hand4); //omega = t*t
        hold(0, {tsRAW});
        omega.EBEPow(temp[3], Singleton<T>::MINUS_ONE, hand4[0]); //omega = t * s / t * t;
        record(0, {omegaRAW});

        hold(1, {omegaRAW});
        x.setSum(h, s, Singleton<T>::ONE, omega, hand4 + 1); // x = h + omega * s

        hold(0, {rWAR});
        r.setDifference(s, t, Singleton<T>::ONE, omega, hand4); // r = s - omega * t
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

        hold(0, {hRAW});
        pUpdate(); // p = p - beta * omega * v
    }
    if (i >= maxIterations)
        std::cout << "WARNING: Maximum number of iterations reached.  Convergence failed.";

    synch();

    const TimePoint end = std::chrono::steady_clock::now();
    const double time = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();
    // std::cout<< "BiCGSTAB #iterations = " << iteration << std::endl;
    // std::cout << time << ", ";
}

template<typename T>
BCGBanded<T>::BCGBanded(Handle* hand4, BandedMat<T> A, const Vec<T> &b, Mat<T> *bHeightX7, Vec<T>* allocated9, const T &tolerance,
size_t maxIterations): BiCGSTAB<T>(b, hand4, bHeightX7, allocated9, tolerance, maxIterations), A(A){
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
    Mat<T> *allocatedBHeightX7,
    Vec<T>* allocated9,
    const T tolerance,
    const size_t maxIterations
) {
    BCGBanded<T> solver(hand4, A, b, allocatedBHeightX7, allocated9, tolerance, maxIterations);
    solver.solveUnpreconditioned(result);
}

template<typename T>
void BCGBanded<T>::test() {
    Handle hand4[4]{};

    size_t n = 5;
    size_t numDiagonals = 2;

    auto indices = SimpleArray<int32_t>::create(numDiagonals, hand4[0]);
    std::vector<int32_t>  indicesHost = {0, 1};
    indices.set(indicesHost.data(), hand4[0]);

    auto banded = BandedMat<double>::create(n, indicesHost.size(), indices);
    banded.col(0).fill(1, hand4[0]);
    banded.col(1).fill(2, hand4[0]);

    auto rhs = SimpleArray<double>::create(n, hand4[0]);
    std::vector<double>  rhsHost = {1,2,10,4,5};
    rhs.set(rhsHost.data(), hand4[0]);

    BCGBanded<double> bcg(hand4, banded, rhs, nullptr, nullptr, 1e-6, 100);

    auto result = SimpleArray<double>::create(n, hand4[0]);
    result.fillRandom(hand4);

    bcg.solveUnpreconditioned(result);

    std::cout << "result = " << GpuOut<double>(result, hand4[0]) << std::endl;
    std::cout << "expected: 85  -42, 22, -6, 5 " << std::endl;
}

// const Vec<T> &other,
//     Vec<T> &result,
//     Handle *handle,
//     const Singleton<T> *alpha,
//     const Singleton<T> *beta,
//     bool transpose

template<typename T>
void BCGDense<T>::mult(Vec<T> &vec, Vec<T> &product, Singleton<T> multProduct, Singleton<T> premultResult) const {
    A.mult(vec, product, this->hand4, &multProduct, &premultResult, false);
}

template<typename T>
BCGDense<T>::BCGDense(Handle *hand4, SquareMat<T> A, const Vec<T> &b, Mat<T> *allocatedBSizeX7, Vec<T> *allocated9, T tolerance, size_t maxIterations): BiCGSTAB<T>(b, hand4, allocatedBSizeX7, allocated9, tolerance, maxIterations), A(A) {

}
//(Handle *hand4, SquareMat<T> A, const Vec<T> &b, Mat<T> *allocatedBSizeX7, Vec<T> *allocated9, const T &tolerance, size_t maxIterations);
template<typename T>
void BCGDense<T>::solve(Handle *hand4, const SquareMat<T> &A, Vec<T> &result, const Vec<T> &b, Mat<T> *bHeightX7, Vec<T> *allocated9, T tolerance, size_t maxIterations) {
    BCGDense<T> solver(hand4, A, b, bHeightX7, allocated9, tolerance, maxIterations);
    solver.solveUnpreconditioned(result);
}

template<typename T>
void BCGDense<T>::test() {
    Handle hand4[4]{};
    size_t n = 6;
    auto A = SquareMat<T>::create(n);
    std::vector<T>  hostA = {0.410352, -0.186335, -0.0563147, -0.172257, -0.0993789, -0.0389234,
                            -0.186335, 0.354037, -0.186335, -0.0993789, -0.21118, -0.0993789,
                            0, 0, 1, 0, 0, 0,
                            0, 0, 0, 1, 0, 0,
                            0, 0, 0, 0, 1, 0,
                            0, 0, 0, 0, 0, 1};
    A.set(hostA.data(), hand4[0]);
    auto b = SimpleArray<T>::create(n, hand4[0]);
    std::vector<T>  hostB = {-1.51304, -1.56522, -0.313043, -0.486957, -0.434783, 0.313043};
    b.set(hostB.data(), hand4[0]);
    auto result = SimpleArray<T>::create(n, hand4[0]);
    auto bHeightX7 = Mat<T>::create(n, 7);
    auto aX9 = SimpleArray<T>::create(9, hand4[0]);
    T tolerance = 1.0e-6;
    size_t maxIterations = 100;
    solve(hand4, A, result, b, &bHeightX7, &aX9, tolerance, maxIterations);
    std::cout << "result = " << GpuOut<T>(result, hand4[0]) << std::endl;

    std:: cout << "expected: -7.48312639568, -8.35954534961, -2.29212890075, -2.60674032488, -2.94381665669, -0.80898814329" << std::endl;

}


template class BiCGSTAB<double>;
template class BiCGSTAB<float>;

template class BCGBanded<double>;
template class BCGBanded<float>;

template class BCGDense<double>;
template class BCGDense<float>;



//multi streamed version.


//
//
// synchAll();
//     record(0, {xRAW});
//     TimePoint start = std::chrono::steady_clock::now();
//
//     auto& x = initGuess;
//     preamble(x);
//
//     size_t numIterations = 0;
//     for (; numIterations < maxIterations; numIterations++) {
//         mult(p, v); // v = A * p
//
//         r_tilde.mult(v, alpha, hand4);
//         alpha.EBEPow(rho, Singleton<T>::MINUS_ONE, hand4[0]); //alpha = rho / (r_tilde * v)
//         record(0, {alphaRAW});
//
//
//         wait(1, {alphaRAW});
//
//         set(h, x, 1);
//
//         h.add(p, &alpha, hand4 + 1); // h = x + alpha * p
//         record(1, {hRAW});
//
//         wait(0, {xRAW});
//         s.setDifference(r, v, Singleton<T>::ONE, alpha, hand4); // s = r - alpha * v
//         record(0, {sRAW});
//
//         wait(2, {sRAW});
//         if (isSmall(s, temp[2], 2)) {
//             wait(2, {hRAW});
//             set(x, h, 2);
//             break;
//         }
//
//         mult(s, t); // t = A * s
//
//         t.mult(s, temp[3], hand4 + 3);
//         record(3, {prodTSRAW});
//
//         t.mult(t, omega, hand4);
//         wait(0, {prodTSRAW});
//         omega.EBEPow(temp[3], Singleton<T>::MINUS_ONE, hand4[0]); //omega = t * s / t * t;
//
//         record(0, {omegaRAW});
//
//         wait(1, {omegaRAW});
//         x.setSum(h, s, Singleton<T>::ONE, omega, hand4 + 1); // x = h + omega * s
//         record(1, {xRAW});
//
//
//         r.setDifference(s, t, Singleton<T>::ONE, omega, hand4); // r = s - omega * t
//         record(0, {rRAW});
//
//         wait(2, {xRAW, rRAW});
//
//         if (isSmall(r, temp[2], 2)) break;
//
//         r_tilde.mult(r, rho_new, hand4);
//
//         beta.setProductOfQuotients(rho_new, rho, alpha, omega, hand4[0]); // beta = (rho_new / rho) * (alpha / omega);
//
//         set(rho, rho_new, 0);
//
//         wait(0, {hRAW});
//         pUpdate(0); // p = r + beta(p - omega * v)
//     }
//
//     if (numIterations >= maxIterations) std::cout << "WARNING: Maximum number of iterations reached.  Convergence failed.";
//     synchAll();
//
//     const TimePoint end = std::chrono::steady_clock::now();
//     const double time = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();
//     // std::cout<< "BiCGSTAB #iterations = " << numIterations << std::endl;
//     // std::cout << time << ", ";