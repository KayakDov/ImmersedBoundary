
#include "EigenDecompForFortran.h"

#include <vector>
#include <memory>
#include "poisson/BoundaryConfig.cuh"
#include "poisson/Poisson.cuh"
#include "solvers/EigenDecomp/EigenDecomp2d.h"
#include "solvers/EigenDecomp/EigenDecomp3d.cuh"
#include "solvers/EigenDecomp/EigenDecompThomas.cuh"

/**
 * @brief Constructs the overarching Fortran interop wrapper for the Eigen Decomposition solver.
 * * Acts as the entry point from the Fortran/Shroud API, translating raw host vectors
 * and configuration flags into the heavily templated, GPU-accelerated C++ environment.
 * * @param rows The number of rows (Y-dimension) in the computational grid.
 * @param cols The number of columns (X-dimension) in the computational grid.
 * @param layers The number of layers (Z-dimension) in the computational grid.
 * @param dx Host vector containing grid spacing for the X-axis. Size 1 implies uniform.
 * @param dy Host vector containing grid spacing for the Y-axis. Size 1 implies uniform.
 * @param dz Host vector containing grid spacing for the Z-axis. Size 1 implies uniform.
 * @param leftIsNeumann True if the left boundary is Neumann (otherwise Dirichlet).
 * @param rightIsNeumann True if the right boundary is Neumann.
 * @param topIsNeumann True if the top boundary is Neumann.
 * @param bottomIsNeumann True if the bottom boundary is Neumann.
 * @param backIsNeumann True if the back boundary is Neumann.
 * @param frontIsNeumann True if the front boundary is Neumann.
 * @param leftVal The condition value (derivative or constant) at the left boundary.
 * @param rightVal The condition value at the right boundary.
 * @param topVal The condition value at the top boundary.
 * @param bottomVal The condition value at the bottom boundary.
 * @param frontVal The condition value at the front boundary.
 * @param backVal The condition value at the back boundary.
 * @param isStaggered True if using a staggered grid discretization.
 * @param thomas True to utilize the Thomas algorithm solver for the Z dimension.
 * @param sizeOfBForX Allocated device array for X-axis intermediate calculations.
 * @param sizeOfBForRHS Allocated device array for RHS intermediate calculations.
 * @param sizeOfBForBAdj Allocated device array for boundary correction adjustments.
 */
template<typename Real>
EigenDecompForFortran<Real>::EigenDecompForFortran(
    GridDim dim,
    const XYZ<std::vector<Real>> &delta,
    XYZ<bool> startIsNeumann, XYZ<bool> endIsNeumann, XYZ<Real> startVal, XYZ<Real> endVal,
    XYZ<eigen::LaplOperatorT> segType,
    bool thomas, Real helmholtzShift,
    SimpleArray<Real> sizeOfBForX, SimpleArray<Real> sizeOfBForRHS, SimpleArray<Real> sizeOfBForBAdj,
    size_t gpuIndex
    ) : x(sizeOfBForX), b(sizeOfBForRHS), adjToB(sizeOfBForBAdj),
        // Sized off x.size() alone -- guaranteed equal to b.size() (see the
        // class-level comment in the header for why), so one buffer covers
        // both directions.
        pinnedBuf(allocPinned(sizeOfBForX.size()), &cudaFreeHost), hand(gpuIndex) {
    
    Handle hands[3] = {Handle(gpuIndex), Handle(gpuIndex), Handle(gpuIndex)};
    Event events[3];

    buildBoundaryConfigAndLaunch<Real>(
        dim, delta, startIsNeumann, endIsNeumann, startVal, endVal, segType,
        hands[0],
        [&](const auto& boundaryHost) {
            poisson::boundaryCorrection(boundaryHost.forDevice(), sizeOfBForBAdj, hands[0]);
            using SegXType = typename std::decay_t<decltype(boundaryHost.x)>::SegmentType;

            if (dim.layers <= 1)
                eds = std::make_unique<EigenDecomp2d<Real>>(boundaryHost.forDevice(), hands, events[0]);
            else
                eds = thomas ?
                    std::make_unique<EigenDecompThomas<Real, SegXType>>(boundaryHost, hands, events, helmholtzShift) :
                    std::make_unique<EigenDecomp3d<Real>>(boundaryHost.forDevice(), hands, events, helmholtzShift);
        }
    );
    for (size_t i = 0; i < 3; ++i) {
        events[i].record(hands[i]);
        events[i].hold(hand);
    }
}


template<typename Real>
void EigenDecompForFortran<Real>::solve() {
    // No memcpy here: pinnedBuf already holds the RHS the caller wrote
    // directly into it (via the pointer from pinnedPtr()). This H2D read
    // is enqueued on hand before the D2H write-back below, so the two can
    // never race -- see the class-level comment in the header.
    b.set(pinnedBuf.get(), hand);

    b.add(adjToB, &GPUScalar<Real>::get(1, hand), &hand);
    eds->solve(x, b, hand);

    // Writes the solution back into the same buffer the RHS was just read
    // from. Safe for the same same-stream-ordering reason as above.
    x.get(pinnedBuf.get(), hand);
}

template<typename Real>
void EigenDecompForFortran<Real>::synch() {
    // No memcpy here either: the solution is already sitting in pinnedBuf
    // once the device work this waits for has completed. The caller reads
    // it directly from the same pointer pinnedPtr() gave them.
    hand.synch();
}

template class EigenDecompForFortran<double>;
template class EigenDecompForFortran<float>;
