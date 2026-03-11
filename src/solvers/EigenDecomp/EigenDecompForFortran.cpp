#include "EigenDecompForFortran.h"

template<typename Real>
EigenDecompForFortran<Real>::EigenDecompForFortran(size_t rows, size_t cols, size_t layers, double dx, double dy,
double dz, bool thomas, SimpleArray<Real> x, SimpleArray<Real> b) :x(x), b(b) {

    Handle hands[3];
    Event events[2];

    GridDim dim(rows, cols, layers);

    if (layers <= 1)
        eds = std::make_unique<EigenDecomp2d<Real>>(dim, hands, Real2d(dx, dy), events[0]);
    else
        eds = thomas ?
            std::make_unique<EigenDecompThomas<Real>>(dim, hands, Real3d(dx, dy, dz), events):
            std::make_unique<EigenDecomp3d<Real>>(dim, hands, Real3d(dx, dy, dz), events);
}

template<typename Real>
void EigenDecompForFortran<Real>::solve(Real *xHost, Real *bHost)  {
    b.set(bHost, hand);
    eds->solve(x, b, hand);
    x.get(xHost, hand);
}

template class EigenDecompForFortran<double>;
template class EigenDecompForFortran<float>;