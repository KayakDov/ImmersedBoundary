#include "EigenDecompForFortran.h"
#include <vector>
#include <memory>

#include "deviceArrays/headers/Support/Streamable.h"
#include "poisson/AxisSegment.cuh"
#include "poisson/BoundaryConfig.cuh"

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
    size_t rows, size_t cols, size_t layers,
    const std::vector<Real>& dx, const std::vector<Real>& dy, const std::vector<Real>& dz,
    bool leftIsNeumann, bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool backIsNeumann, bool frontIsNeumann,
    Real leftVal, Real rightVal, Real topVal, Real bottomVal, Real frontVal, Real backVal,
    bool isStaggered,
    bool thomas,
    SimpleArray<Real> sizeOfBForX,
    SimpleArray<Real> sizeOfBForRHS,
    SimpleArray<Real> sizeOfBForBAdj
) : x(sizeOfBForX), b(sizeOfBForRHS), adjToB(sizeOfBForBAdj) {
    //
    // std::cout << "\n========== EigenDecompForFortran Constructor ==========\n";
    //
    // std::cout << "Dimensions:\n";
    // std::cout << "  rows   = " << rows   << '\n';
    // std::cout << "  cols   = " << cols   << '\n';
    // std::cout << "  layers = " << layers << '\n';
    //
    // std::cout << "\nSpacing vectors:\n";
    //
    // std::cout << "dx (" << dx.size() << ") = ";
    // for (const auto& v : dx) std::cout << v << " ";
    // std::cout << '\n';
    //
    // std::cout << "dy (" << dy.size() << ") = ";
    // for (const auto& v : dy) std::cout << v << " ";
    // std::cout << '\n';
    //
    // std::cout << "dz (" << dz.size() << ") = ";
    // for (const auto& v : dz) std::cout << v << " ";
    // std::cout << '\n';
    //
    // std::cout << "\nBoundary conditions:\n";
    //
    // std::cout << "leftIsNeumann   = " << leftIsNeumann   << '\n';
    // std::cout << "rightIsNeumann  = " << rightIsNeumann  << '\n';
    // std::cout << "topIsNeumann    = " << topIsNeumann    << '\n';
    // std::cout << "bottomIsNeumann = " << bottomIsNeumann << '\n';
    // std::cout << "frontIsNeumann  = " << frontIsNeumann  << '\n';
    // std::cout << "backIsNeumann   = " << backIsNeumann   << '\n';
    //
    // std::cout << "\nBoundary values:\n";
    //
    // std::cout << "leftVal   = " << leftVal   << '\n';
    // std::cout << "rightVal  = " << rightVal  << '\n';
    // std::cout << "topVal    = " << topVal    << '\n';
    // std::cout << "bottomVal = " << bottomVal << '\n';
    // std::cout << "frontVal  = " << frontVal  << '\n';
    // std::cout << "backVal   = " << backVal   << '\n';
    //
    // std::cout << "\nOther flags:\n";
    //
    // std::cout << "isStaggered = " << isStaggered << '\n';
    // std::cout << "thomas      = " << thomas      << '\n';
    //
    // std::cout << "\nBuffer sizes:\n";
    //
    // std::cout << "sizeOfBForX   = " << sizeOfBForX.size()   << '\n';
    // std::cout << "sizeOfBForRHS = " << sizeOfBForRHS.size() << '\n';
    // std::cout << "sizeOfBForBAdj= " << sizeOfBForBAdj.size() << '\n';
    //
    // std::cout << "=======================================================\n";

    Handle hands[3];
    Event events[2];
    cudaStream_t defaultStream = 0;


    buildBoundaryConfigAndLaunch<Real>(
        GridDim{rows, cols, layers},
        XYZ<std::vector<Real>>{dx, dy, dz},
        XYZ<bool>{leftIsNeumann, topIsNeumann, frontIsNeumann},
        XYZ<bool>{rightIsNeumann, bottomIsNeumann, backIsNeumann},
        XYZ<Real>{leftVal, topVal, frontVal},
        XYZ<Real>{rightVal, bottomVal, backVal},
        isStaggered,
        defaultStream,
        [&](const auto& boundaryHost) {
            poisson::boundaryCorrection(boundaryHost.forDevice(), sizeOfBForBAdj, hands[0]);
            using SegXType = typename std::decay_t<decltype(boundaryHost.x)>::SegmentType;

            if (layers <= 1)
                eds = std::make_unique<EigenDecomp2d<Real>>(boundaryHost.forDevice(), hands, events[0]);
            else
                eds = thomas ?
                    std::make_unique<EigenDecompThomas<Real, SegXType>>(boundaryHost, hands, events) :
                    std::make_unique<EigenDecomp3d<Real>>(boundaryHost.forDevice(), hands, events);
        }
    );
}
template<typename Real>
void EigenDecompForFortran<Real>::solve(Real *xHost, Real *bHost)  {

    b.set(bHost, hand);

    b.add(adjToB, &GPUScalar<Real>::get(1), &hand);

    eds->solve(x, b, hand);
    x.get(xHost, hand);
}

template class EigenDecompForFortran<double>;
template class EigenDecompForFortran<float>;