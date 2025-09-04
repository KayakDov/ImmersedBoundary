#include "deviceArrays.h"
#include <cmath>
#include "Event.h"

/**
 * Unpreconditioned BiCGSTAB algorithm to solve the linear system Ax = b.
 * @param A The coefficient matrix (CuArray2D).
 * @param diags The diagonals of the matrix A.
 * @param b The right-hand side vector (CuArray1D).
 * @param preAlocated Optional pre-allocated memory for intermediate computations (CuArray2D).  
 * There should be 7 columns and b.size() rows.
 * @param maxIterations Maximum number of iterations to perform. Default is
 * the size of vector b.
 * @param tolerance Convergence tolerance. Default is 1e-6 for float and 1e-12 for double.
 * @return The solution vector x (CuArray1D).
 * @throws std::runtime_error if T is not float or double.
 */
template<typename T>
Vec<T> unpreconditionedBiCGSTAB(
    const Mat<T>& A, 
    const Vec<int>& diags,
    const Vec<T>& b, 
    Vec<T>* x = nullptr, 
    size_t maxIterations = -1, 
    T tolerance = std::is_same<T,double>::value ? T(1e-12) : T(1e-6),
    Mat<T>* preAlocated = nullptr
){
    if(maxIterations == (size_t)-1) maxIterations = b.size()*10000;
    static_assert(std::is_same<T,float>::value || std::is_same<T,double>::value,
              "Algorithms.cu unpreconditionedBiCGSTAB: T must be float or double");
    Handle handle[4]{};
    Event alphaReady, sReady, hReady, omegaReady, rReady, xReady, prodTS;

    const Mat<T>& paM = preAlocated ? *preAlocated : Mat<T>(b.size(), 7);
    Vec<T> r(paM, 0, IndexType::Column), r_tilde(paM, 1, IndexType::Column), p(paM, 2, IndexType::Column), 
        v(paM, 3, IndexType::Column), s(paM, 4, IndexType::Column), t(paM, 5, IndexType::Column), h(paM, 6, IndexType::Column);
    Vec<T> paV(6); 
    Singleton<T> rho(paV, 0), alpha(paV, 1), omega(paV, 2), rho_new(paV, 3), beta(paV, 4), temp(paV, 5);

    Vec<T> result = x ? *x : Vec<T>(b.size());
    xReady.record(handle[0]);
    
    r_tilde.fillRandom(&handle[0]); // set r_tilde randomly    

    r.set(b, handle[0].stream);
    A.diagMult(diags, result, &r, handle, T(-1), T(1)); // r = b - A * x    
    
    p.set(r, handle[0].stream);
       
    r_tilde.mult(r, &rho, handle);    
    
    xReady.wait(handle[1]);

    int i = 0;
    for(;i < maxIterations; i++) {
       
        A.diagMult(diags, p, &v, handle, T(1), T(0)); // v = A * p

        r_tilde.mult(v, &alpha, handle);        
        alpha.EBEPow(rho, T(-1), handle[0].stream); //alpha = rho / (r_tilde * v)
        alphaReady.record(handle[0]);

        cudaStreamSynchronize(handle[1].stream);
        omegaReady.renew();
        alphaReady.wait(handle[1]);
        
        
        h.set(result, handle[1].stream);
        cudaStreamSynchronize(handle[1].stream);
        alphaReady.renew();
        xReady.renew();
        h.add(p, alpha.get(handle[1].stream), handle + 1); // h = x + alpha * p

        s.set(r, handle[0].stream);
        s.sub(v, alpha.get(handle[0].stream), &handle[0]); // s = r - alpha * v
        sReady.record(handle[0]);
        
        result.set(h, handle[1].stream);

        sReady.wait(handle[2]);
        hReady.wait(handle[2]);
        Singleton<T> sQuared;
        s.mult(s, &sQuared, handle + 2);
        T sSq = sQuared.get(handle[2].stream);
        cudaStreamSynchronize(handle[2].stream);
        if(sSq < tolerance) break;
        sReady.renew();
        hReady.renew();

        A.diagMult(diags, s, &t, handle); // t = A * s

        t.mult(s, &temp, handle + 3);
        prodTS.record(handle[3]);
        t.mult(t, &omega, handle);
        prodTS.wait(handle[0]);
        omega.EBEPow(temp, T(-1), handle[0].stream); //omega = t * s / t * t;
        cudaStreamSynchronize(handle[0].stream);
        prodTS.renew();

        omegaReady.record(handle[0]);

        omegaReady.wait(handle[1]);
        result.add(s, omega.get(handle[1].stream), handle + 1); // x = h + omega * s
        xReady.record(handle[1]);        
        
        r.set(s, handle[0].stream);
        r.sub(t, omega.get(handle[0].stream), handle); // r = s - omega * t
        rReady.record(handle[0]);

        xReady.wait(handle[2]);
        rReady.wait(handle[2]);

        Singleton<T> rSquared;
        r.mult(r, &rSquared, handle + 2);
        T rSq = rSquared.get(handle[2].stream);
        cudaStreamSynchronize(handle[2].stream);
        if(rSq < tolerance) break;        
        rReady.renew();

        r_tilde.mult(r, &rho_new, handle);

        temp.set(rho, handle[0].stream);
        temp.EBEPow(rho_new, T(-1), handle[0].stream);
        
        beta.set(omega, handle[0].stream);
        beta.EBEPow(alpha, T(-1), handle[0].stream);
        beta.EBEPow(temp, T(1), handle[0].stream); // beta = (rho_new / rho) * (alpha / omega);
        rho.set(rho_new, handle[0].stream);

        p.mult(beta.get(handle[0].stream), handle);
        p.add(r, T(1), handle); // p = r + beta * p
        p.sub(v, beta.get(handle[0].stream) * omega.get(handle[0].stream), handle); // p = p - beta * omega * v        
    }

    std::cout << "algorithms.cu unpreconditionedBiCGSTAB Number of iterations:" << i << std::endl;
    
    return result;
}

