//
// Created by usr on 1/30/26.
//

#include "PPrimeFPrime.cuh"

#include "handle.h"
#include "ImerssedEquation.h"
#include "deviceArrays/defFiles/DeviceData.cuh"

/**
 * @brief Computes the discrete divergence (\nabla \cdot u*) on a staggered MAC grid.
 *
 * This kernel calculates the divergence at the cell centers (Eulerian grid) using
 * the intermediate velocity components stored on the cell faces. This represents
 * the "volume error" or source term for the Pressure Poisson equation in the
 * SIMPLE-based Immersed Boundary Method.
 *
 * @tparam Real Floating point type (float or double).
 *
 * @param u           The x-velocity component grid (staggered).
 * @param v           The y-velocity component grid (staggered).
 * @param w           The z-velocity component grid (staggered).
 * @param dst         Output scalar grid (cell centers) where divergence is stored.
 * @param deltaCols   The grid spacing in the x-direction (dx).
 * @param deltaRows   The grid spacing in the y-direction (dy).
 * @param deltaLayers The grid spacing in the z-direction (dz).
 *
 * @note **Grid Dimension Requirements:**
 * To ensure every cell center in @p dst has a bounding pair of faces:
 * - @p u must have dimensions (dst.cols + 1, dst.rows, dst.layers).
 * - @p v must have dimensions (dst.cols, dst.rows + 1, dst.layers).
 * - @p w must have dimensions (dst.cols, dst.rows, dst.layers + 1).
 *
 * @details
 * The calculation follows the second-order central difference for staggered grids:
 * div = (u[i+1,j,k] - u[i,j,k])/dx + (v[i,j+1,k] - v[i,j,k])/dy + (w[i,j,k+1] - w[i,j,k])/dz
 */
template <typename Real>
__global__ void divergenceKernel3d(
    DeviceData3d<Real> u, DeviceData3d<Real> v, DeviceData3d<Real> w, DeviceData3d<Real> dst, const double deltaCols, const double deltaRows, const double deltaLayers, Real* scalar) {
    if (GridInd3d ind; ind < dst)
        dst[ind] = *scalar * (
            (u(ind, 0, 1, 0) - u[ind])/deltaCols +
            (v(ind, 1, 0, 0) - v[ind])/deltaRows +
            (w(ind, 0, 0, 1) - w[ind])/deltaLayers);

}

/**
 * @brief Computes the discrete divergence (∇·u*) on a 2D staggered MAC grid.
 *
 * @tparam T Floating point type (float or double).
 * @param u The x-velocity component grid.
 * @param v The y-velocity component grid.
 * @param dst Output scalar grid (cell centers) for divergence results.
 * @param deltaCols Grid spacing in x (dx).
 * @param deltaRows Grid spacing in y (dy).
 *
 * @note **Requirement:** @p u and @p v must have 1 more element in their
 * respective staggered dimension than @p dst.
 */
template <typename Real>
__global__ void divergenceKernel2d(DeviceData2d<Real> u, DeviceData2d<Real> v, DeviceData2d<Real> dst, const double deltaCols, const double deltaRows, Real* scalar) {

    if(GridInd2d ind;ind < dst)
        dst[ind] = *scalar * (
            (u(ind, 0, 1) - u[ind])/deltaCols +
            (v(ind, 1, 0) - v[ind])/deltaRows
        );
}


template<typename Real, typename Int> //TOOD: create another constructor that calls this one, but creates the arrays from CPU pointers.
PPrimeFprime<Real, Int>::PPrimeFprime(
    const GridDim &dim,
    const SimpleArray<Real> &uStar,
    const SparseCSC<Real, Int> R,
    const SimpleArray<Real> UGamma,
    const Real3d &delta,
    double deltaT,
    ImmersedEq<Real> imEq,
    Real* pPrimeHost,
    Real* fPrimeHost,
    bool multiStream

):  dim(dim),
    uStar(uStar),
    u(uStar.subArray(0, dim.rows * (dim.cols + 1) * dim.layers)),
    v(uStar.subArray(u.size(), (dim.rows + 1) * dim.cols * dim.layers)),
    w(uStar.subArray(u.size() + v.size(), dim.rows * dim.cols * (dim.layers + 1))),//TODO:verify this works for 2d u*
    R(R),
    UGamma(UGamma),
    delta(delta),
    dT(Singleton<Real>::create(3/(2 * deltaT), imEq.hand5[0])){

    auto& hand1 = imEq.hand5[0];
    auto& RHSPPrime = imEq.baseData.p;
    auto& RHSFPrime = imEq.baseData.f;

    setRHSPPrime(RHSPPrime, hand1);
    setRHSFPrime(RHSFPrime, imEq.sparseMultBuffer, hand1);

    auto pPrimeDevice = imEq.solve(multiStream);

    imEq.events11[0].record(0);
    imEq.events11[0].wait(1);
    pPrimeDevice.get(pPrimeHost, imEq.hand5[1]);

    auto& fPrimeDevice = imEq.baseData.allocatedFSize();
    imEq.baseData.B->multB(pPrimeDevice, fPrimeDevice, Singleton<Real>::TWO, Singleton<Real>::ZERO, false);
    fPrimeDevice.subtract(RHSFPrime, Singleton<Real>::TWO, imEq.sparseMultBuffer.get(0), hand1);
    pPrimeDevice.get(fPrimeHost, hand1);
}

/**
 * @brief Host-side wrapper to launch the appropriate divergence kernel.
 */
template <typename Real, typename Int>
void PPrimeFprime<Real, Int>::divergence(SimpleArray<Real> result, Singleton<Real> scalar, Handle &hand) {

    const KernelPrep kp = result.kernelPrep();

    if (dim.layers > 1) divergenceKernel3d<Real><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
            u.tensor(dim.rows, dim.layers).toKernel3d(),
            v.tensor(dim.rows, dim.layers).toKernel3d(),
            w.tensor(dim.rows, dim.layers).toKernel3d(),
            result.tensor(dim.rows, dim.layers).toKernel3d(),
            delta.x, delta.y, delta.z,
            scalar.data()
        );
    else divergenceKernel2d<Real><<<kp.numBlocks, kp.threadsPerBlock, 0, hand>>>(
            u.matrix(dim.rows).toKernel2d(),
            v.matrix(dim.rows).toKernel2d(),
            result.matrix(dim.rows).toKernel2d(),
            delta.x, delta.y,
            scalar.data()
        );
}

template<typename Real, typename Int>
void PPrimeFprime<Real, Int>::setRHSFPrime(SimpleArray<Real> result, SimpleArray<Real>& sparseMultBuffer, Handle &hand) {
    size_t minBufferSize = sparseMultBuffer->size() > R.multWorkspaceSize(uStar, result, dT, Singleton<Real>::ZERO, true, hand);
    if (!sparseMultBuffer || sparseMultBuffer->size() > minBufferSize) sparseMultBuffer = std::make_shared<SimpleArray<Real>>(SimpleArray<Real>::create(1.5 * minBufferSize, hand, true));

    R.mult(uStar, result, dT, Singleton<Real>::ZERO, true, *sparseMultBuffer, hand);

    result.subtract(UGamma, &dT, sparseMultBuffer->get(0), &hand);
}

template<typename Real, typename Int>
void PPrimeFprime<Real, Int>::setRHSPPrime(SimpleArray<Real> result, Handle &hand) {
    divergence(result, dT, hand);
}

template class PPrimeFprime<double, int32_t>;
template class PPrimeFprime<float, int32_t>;

template class PPrimeFprime<double, int64_t>;
template class PPrimeFprime<float, int64_t>;
