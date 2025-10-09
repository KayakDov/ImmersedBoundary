#include "deviceArrays/deviceArrays.h"
#include <cmath>
#include "Event.h"
#include <iostream>
#include <chrono>


using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

template <typename T>
class BiCGSTAB{
private:
    T tolerance;
    Handle handle[4]{};
    Event alphaReady, sReady, hReady, omegaReady, rReady, xReady, prodTS;
    const Vec<T> b;
    Mat<T> paM;
    Vec<T> r, r_tilde, p, v, s, t, h;
    Vec<T> paV;
    Singleton<T> rho, alpha, omega, rho_new, beta, temp;

    size_t maxIterations;

    void wait(const size_t streamIndex, const std::initializer_list<std::reference_wrapper<Event>> evs) const{
        for (auto& ref_e : evs)
            ref_e.get().wait(handle[streamIndex]);

    }

    static void renew(const std::initializer_list<std::reference_wrapper<Event>> evs) {
        for (auto& ref_e : evs)
            ref_e.get().renew();
    }

    void record(size_t streamIndex, Event& e) const {
        e.record(handle[streamIndex]);
    }

     void synch(const size_t streamInd) const{
        handle[streamInd].synch();
    }

    bool isSmall(const Vec<T>& v, const size_t streamInd){
        Singleton<T> vSquared = Singleton<T>::create();
            v.mult(v, &vSquared, handle + streamInd);
            T vSq = vSquared.get(handle[streamInd].stream);
            synch(streamInd);
            return vSq < tolerance;
    }

    void set(Vec<T>& dst, const Vec<T>& src, const size_t streamInd){
        dst.set(src, handle[streamInd].stream);
    }

    void setQuotient(Singleton<T>& dst, const Singleton<T>& numerator, const Singleton<T>& denom, const size_t streamInd){
        dst.set(denom, handle[streamInd].stream);
        dst.EBEPow(numerator, Singleton<T>::MINUS_ONE, handle[streamInd].stream);
    }

public:
    explicit BiCGSTAB(
        const Vec<T>& b,
        T tolerance = std::is_same_v<T,double> ? T(1e-12) : T(1e-6),
        size_t maxIterations = 0,
        Mat<T>* preAllocated = nullptr
    ):tolerance(tolerance),
      b(b),
      paM(preAllocated ? *preAllocated : Mat<T>::create(b.size(), 7)),
      r(paM.col(0)), r_tilde(paM.col(1)), p(paM.col(2)), v(paM.col(3)), s(paM.col(4)), t(paM.col(5)), h(paM.col(6)),
      paV(Vec<T>::create(6, handle[0].stream)),
      rho(paV.get(0)), alpha(paV.get(1)), omega(paV.get(2)), rho_new(paV.get(3)), beta(paV.get(4)), temp(paV.get(5)),
      maxIterations(maxIterations)
    {
        static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>,
                "Algorithms.cu unpreconditionedBiCGSTAB: T must be float or double");
        if(this->maxIterations == 0) this->maxIterations = b.size()*5;
        cudaDeviceSynchronize();
        for (const auto& h : handle)
            cublasSetPointerMode(h.handle, CUBLAS_POINTER_MODE_DEVICE);
    }

    Vec<T> solveUnpreconditionedBiCGSTAB(const BandedMat<T>& A, Vec<T>* x = nullptr){

        Vec<T> result = x ? *x : Vec<T>::create(b.size(), handle[0].stream);
        result.fillRandom(&handle[0]); // set x randomly
        record(0, xReady);
        
        r_tilde.fillRandom(&handle[0]); // set r_tilde randomly    

        set(r, b, 0);
        A.mult(result, &r, handle, &Singleton<T>::MINUS_ONE, &Singleton<T>::ONE); // r = b - A * x
        
        set(p, r, 0);
        
        r_tilde.mult(r, &rho, handle);    
        
        wait(1, {xReady});

        double totalTime = 0;
        size_t numIterations = 0;
        for(;numIterations < maxIterations; numIterations++) {

            TimePoint start = std::chrono::steady_clock::now();

            A.mult(p, &v, handle, &Singleton<T>::ONE, &Singleton<T>::ZERO); // v = A * p

            r_tilde.mult(v, &alpha, handle);        
            alpha.EBEPow(rho, Singleton<T>::MINUS_ONE, handle[0].stream); //alpha = rho / (r_tilde * v)
            record(0, {alphaReady});

            synch(1);
            omegaReady.renew();
            wait(1, {alphaReady});
            
            set(h, result, 1);
            synch(1);
            renew({alphaReady, xReady});
            h.add(p, &alpha, handle + 1); // h = x + alpha * p

            temp.set(alpha, handle[0].stream);
            temp.mult(Singleton<T>::MINUS_ONE, handle);
            s.setSum(r, v, &Singleton<T>::ONE, &temp, handle); // s = r - alpha * v
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
            omega.EBEPow(temp, Singleton<T>::MINUS_ONE, handle[0].stream); //omega = t * s / t * t;

            record(0, {omegaReady});

            wait(1, {omegaReady});
            result.add(s, &omega, handle + 1); // x = h + omega * s
            record(1, {xReady});        
            
            synch(0);
            prodTS.renew();
            temp.set(omega, handle[0].stream);
            temp.mult(Singleton<T>::MINUS_ONE, handle);
            r.setSum(s, t, &Singleton<T>::ONE, &temp, handle); // r = s - omega * t
            record(0, {rReady});

            wait(2, {xReady, rReady});

            if(isSmall(r, 2)) break;
            rReady.renew();

            r_tilde.mult(r, &rho_new, handle);

            setQuotient(temp, rho_new, rho, 0);
            setQuotient(beta, alpha, omega, 0);
            beta.EBEPow(temp, Singleton<T>::ONE, handle[0].stream); // beta = (rho_new / rho) * (alpha / omega);
            
            set(rho, rho_new, 0);

            p.mult(beta, handle);
            p.add(r, &Singleton<T>::ONE, handle); // p = r + beta * p
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

