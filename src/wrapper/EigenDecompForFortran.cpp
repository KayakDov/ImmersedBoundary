
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
    SimpleArray<Real> sizeOfBForX, SimpleArray<Real> sizeOfBForRHS, SimpleArray<Real> sizeOfBForBAdj
    ) : x(sizeOfBForX), b(sizeOfBForRHS), adjToB(sizeOfBForBAdj),pinnedX(allocPinned(sizeOfBForX.size()), &cudaFreeHost),
        pinnedB(allocPinned(sizeOfBForRHS.size()), &cudaFreeHost) {
    
    Handle hands[3];
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
void EigenDecompForFortran<Real>::solve(const Real *bHost)  {

    std::memcpy(pinnedB.get(), bHost, b.size() * sizeof(Real));
    b.set(pinnedB.get(), hand);

    b.add(adjToB, &GPUScalar<Real>::get(1), &hand);
    eds->solve(x, b, hand);

    x.get(pinnedX.get(), hand);
}

template<typename Real>
void EigenDecompForFortran<Real>::retrieveSoltion(Real *xHost) {
    hand.synch();
    std::memcpy(xHost, pinnedX.get(), x.size() * sizeof(Real));
}

template class EigenDecompForFortran<double>;
template class EigenDecompForFortran<float>;