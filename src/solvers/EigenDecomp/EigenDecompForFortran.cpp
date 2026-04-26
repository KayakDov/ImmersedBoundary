#include "EigenDecompForFortran.h"

template<typename Real>
EigenDecompForFortran<Real>::EigenDecompForFortran(
    size_t rows, size_t cols, size_t layers,
    double dx, double dy, double dz,
    bool leftIsNeumann, bool rightIsNeumann, bool topIsNeumann, bool bottomIsNeumann, bool backIsNeumann, bool frontIsNeumann,
    Real leftVal, Real rightVal, Real topVal, Real bottomVal, Real frontVal, Real backVal,
    bool isStaggered,
    bool thomas,
    SimpleArray<Real> x,
    SimpleArray<Real> b
) :x(x), b(b) {

    Handle hands[3];
    Event events[2];

    BoundaryConfig<Real> boundary(
        {leftIsNeumann, topIsNeumann, frontIsNeumann}, {rightIsNeumann, bottomIsNeumann, backIsNeumann},
        {leftVal, topVal, frontVal}, {rightVal, bottomVal, backVal},
        Real3d(dx, dy, dz),
        GridDim(rows, cols, layers),
        isStaggered
    );

    GridDim dim(rows, cols, layers);

    if (layers <= 1)
        eds = std::make_unique<EigenDecomp2d<Real>>(boundary, hands, events[0]);
    else
        eds = thomas ?
            std::make_unique<EigenDecompThomas<Real>>(boundary, dx, hands, events):
            std::make_unique<EigenDecomp3d<Real>>(boundary, hands, events);
}

template<typename Real>
void EigenDecompForFortran<Real>::solve(Real *xHost, Real *bHost)  {
    b.set(bHost, hand);
    eds->solve(x, b, hand);
    x.get(xHost, hand);
}

template class EigenDecompForFortran<double>;
template class EigenDecompForFortran<float>;