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
void BiCGSTAB<T>::wait(const size_t streamIndex,
                       const std::initializer_list<std::reference_wrapper<Event> > evs) const {
    for (auto &ref_e: evs)
        ref_e.get().wait(hand4[streamIndex]);
}

template<typename T>
void BiCGSTAB<T>::renew(const std::initializer_list<std::reference_wrapper<Event> > evs) {
    for (auto &ref_e: evs)
        ref_e.get().renew();
}


template<typename T>
void BiCGSTAB<T>::record(size_t streamIndex, Event &e) const {
    e.record(hand4[streamIndex]);
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
    synch(streamInd);
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
    rho(a9.get(0)), alpha(a9.get(1)), omega(a9.get(2)), rho_new(a9.get(3)), beta(a9.get(4)),
    temp{{a9.get(5), a9.get(6), a9.get(7), a9.get(8)}},
    maxIterations(maxIterations) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "Algorithms.cu unpreconditionedBiCGSTAB: T must be float or double");
}

template<typename T>
void BiCGSTAB<T>::preamble(Vec<T>& x) {

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

        if (isSmall(r, temp[2])) {
            std::cout << "Converged at r" << std::endl;
            break;
        }

        r_tilde.mult(r, rho_new, hand4);
        beta.setProductOfQuotients(rho_new, rho, alpha, omega, hand4[0]); // beta = (rho_new / rho) * (alpha / omega);
        std::cout << "rho_new: " << GpuOut<T>(rho_new, hand4[0])
                  << " | beta: " << GpuOut<T>(beta, hand4[0]) << std::endl;

        set(rho, rho_new);
        pUpdate(); // p = p - beta * omega * v
    }
    if (iteration >= maxIterations)
        std::cout << "WARNING: Maximum number of iterations reached.  Convergence failed.";

    synch();

    const TimePoint end = std::chrono::steady_clock::now();
    const double time = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();
    std::cout<< "BiCGSTAB #iterations = " << iteration << std::endl;
    std::cout << time << ", ";
}

template<typename T>
void BiCGSTAB<T>::solveUnconditionedMultiStream(Vec<T>& initGuess) {
    synchAll();
    TimePoint start = std::chrono::steady_clock::now();

    auto& x = initGuess;
    preamble(x);

    size_t numIterations = 0;
    for (; numIterations < maxIterations; numIterations++) {
        mult(p, v); // v = A * p

        r_tilde.mult(v, alpha, hand4);
        alpha.EBEPow(rho, Singleton<T>::MINUS_ONE, hand4[0]); //alpha = rho / (r_tilde * v)
        record(0, {alphaReady});

        synch(1);
        omegaReady.renew();
        wait(1, {alphaReady});

        set(h, x, 1);
        synch(1);
        renew({alphaReady, xReady});
        h.add(p, &alpha, hand4 + 1); // h = x + alpha * p

        s.setDifference(r, v, Singleton<T>::ONE, alpha, hand4); // s = r - alpha * v
        record(0, {sReady});

        wait(2, {sReady, hReady});
        if (isSmall(s, temp[2], 2)) {
            set(x, h, 2);
            break;
        }
        renew({sReady, hReady});

        mult(s, t); // t = A * s

        t.mult(s, temp[3], hand4 + 3);
        record(3, {prodTS});
        t.mult(t, omega, hand4);
        wait(0, {prodTS});
        omega.EBEPow(temp[3], Singleton<T>::MINUS_ONE, hand4[0]); //omega = t * s / t * t;

        record(0, {omegaReady});

        wait(1, {omegaReady});
        x.setSum(h, s, Singleton<T>::ONE, omega, hand4 + 1); // x = h + omega * s
        record(1, {xReady});

        synch(0);
        prodTS.renew();
        r.setDifference(s, t, Singleton<T>::ONE, omega, hand4); // r = s - omega * t
        record(0, {rReady});

        wait(2, {xReady, rReady});

        if (isSmall(r, temp[2], 2)) break;
        rReady.renew();

        r_tilde.mult(r, rho_new, hand4);

        beta.setProductOfQuotients(rho_new, rho, alpha, omega, hand4[0]); // beta = (rho_new / rho) * (alpha / omega);

        set(rho, rho_new, 0);

        pUpdate(0); // p = p - beta * omega * v
    }
    if (numIterations >= maxIterations)
        std::cout << "WARNING: Maximum number of iterations reached.  Convergence failed.";
    synchAll();

    const TimePoint end = std::chrono::steady_clock::now();
    const double time = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();
    std::cout<< "BiCGSTAB #iterations = " << numIterations << std::endl;
    std::cout << time << ", ";
}

template<typename T>
BCGBanded<T>::BCGBanded(Handle* hand4, BandedMat<T> A, const Vec<T> &b, Mat<T> *allocatedBSizeX7, Vec<T>* allocated9, const T &tolerance,
size_t maxIterations): BiCGSTAB<T>(b, hand4, allocatedBSizeX7, allocated9, tolerance, maxIterations), A(A){
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

template class BiCGSTAB<double>;
template class BiCGSTAB<float>;

template class BCGBanded<double>;
template class BCGBanded<float>;



//multi streamed version.
