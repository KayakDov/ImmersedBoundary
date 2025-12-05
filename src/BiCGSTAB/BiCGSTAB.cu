#include "BiCGSTAB.cuh"

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
        ref_e.get().wait(handle[streamIndex]);
}

template<typename T>
void BiCGSTAB<T>::renew(const std::initializer_list<std::reference_wrapper<Event> > evs) {
    for (auto &ref_e: evs)
        ref_e.get().renew();
}


template<typename T>
void BiCGSTAB<T>::record(size_t streamIndex, Event &e) const {
    e.record(handle[streamIndex]);
}

template<typename T>
void BiCGSTAB<T>::synch(const size_t streamInd) const {
    handle[streamInd].synch();
}

template<typename T>
bool BiCGSTAB<T>::isSmall(const Vec<T> &v, Singleton<T> preAlocated, const size_t streamInd) {
    v.mult(v, preAlocated, handle + streamInd);
    T vSq = preAlocated.get(handle[streamInd]);
    synch(streamInd);
    return vSq < tolerance;
}

template<typename T>
void BiCGSTAB<T>::set(Vec<T> &dst, const Vec<T> &src, const size_t streamInd) {
    dst.set(src, handle[streamInd]);
}

template<typename T>
void BiCGSTAB<T>::pUpdate(const size_t streamInd) {
    KernelPrep kp = p.kernelPrep();

    // Kernel launch performs: p = r + beta * (p - omega * v)
    updatePKernel<<<kp.numBlocks, kp.threadsPerBlock, 0, handle[streamInd]>>>(
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
    Mat<T> *preAllocated,
    const T tolerance,
    const size_t maxIterations
) : tolerance(tolerance),
    b(b),
    paM(preAllocated ? *preAllocated : Mat<T>::create(b.size(), 7)),
    r(paM.col(0)), r_tilde(paM.col(1)), p(paM.col(2)), v(paM.col(3)), s(paM.col(4)), t(paM.col(5)), h(paM.col(6)),
    paV(Vec<T>::create(9, handle[0])),
    rho(paV.get(0)), alpha(paV.get(1)), omega(paV.get(2)), rho_new(paV.get(3)), beta(paV.get(4)),
    temp{{paV.get(5), paV.get(6), paV.get(7), paV.get(8)}},
    maxIterations(maxIterations) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "Algorithms.cu unpreconditionedBiCGSTAB: T must be float or double");
    cudaDeviceSynchronize();
}

template<typename T>
void BiCGSTAB<T>::solve(
    const BandedMat<T> &A,
    Vec<T> &x,
    const Vec<T> &b,
    Mat<T> *preAllocated,
    const T tolerance,
    const size_t maxIterations
) {
    BiCGSTAB solver(b, preAllocated, tolerance, maxIterations);
    solver.solveUnpreconditionedBiCGSTAB(A, x);
}

template<typename T>
void BiCGSTAB<T>::solve(
    T *A,
    const size_t aLd,
    int32_t *inds,
    const size_t indsStride,
    const size_t numInds,
    T *x,
    const size_t xStride,
    T *b,
    const size_t bStride,
    const size_t bSize,
    T *prealocatedSizeX7, //size x 7 matrix
    const size_t prealocatedLd,
    size_t maxIterations,
    T tolerance
) {
    Mat<T> preAlocatedMat = Mat<T>::create(bSize, 7, prealocatedLd, prealocatedSizeX7);
    Vec<T> bVec = Vec<T>::create(bSize, bStride, b);
    const auto ABanded = BandedMat<T>::create(bSize, numInds, aLd, A, inds, indsStride); //rows cols pointer VecIndices
    auto xVec = Vec<T>::create(bSize, xStride, b);
    BiCGSTAB<T>::solve(ABanded, xVec, bVec, &preAlocatedMat, tolerance, maxIterations);
}

template<typename T>
void BiCGSTAB<T>::preamable(const BandedMat<T> &A, Vec<T> &x) {
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

template<typename T>
void BiCGSTAB<T>::solveUnpreconditionedBiCGSTAB(const BandedMat<T> &A, Vec<T> &x) {
    cudaDeviceSynchronize();
    TimePoint start = std::chrono::steady_clock::now();
    preamable(A, x);

    size_t numIterations = 0;
    for (; numIterations < maxIterations; numIterations++) {
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
        if (isSmall(s, temp[2], 2)) {
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

        if (isSmall(r, temp[2], 2)) break;
        rReady.renew();

        r_tilde.mult(r, rho_new, handle);

        beta.setProductOfQuotients(rho_new, rho, alpha, omega, handle[0]); // beta = (rho_new / rho) * (alpha / omega);

        set(rho, rho_new, 0);

        pUpdate(0); // p = p - beta * omega * v
    }
    cudaDeviceSynchronize();
    const TimePoint end = std::chrono::steady_clock::now();
    const double time = (static_cast<std::chrono::duration<double, std::milli>>(end - start)).count();

    if (numIterations >= maxIterations)
        std::cout << "WARNING: Maximum number of iterations reached.  Convergence failed.";
    std::cout << time << ", ";
}


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
template class BiCGSTAB<double>;
template class BiCGSTAB<float>;