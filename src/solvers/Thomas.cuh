#ifndef CUDABANDED_THOMAS_H
#define CUDABANDED_THOMAS_H
#include "Mat.h"

template <typename Real>
class Thomas {
    Mat<Real> heightX_3XnumSystemsPlus4;
    Tensor<Real> triDiags;///< Must have a multiple of 3 number of columns.
    Mat<Real> x, b, cPrime, dPrime;
public:

    Thomas(Mat<Real> &heightX_7XnumSystems);

    /**
     * Note that the first element of each subdiagonal, and the last element of each superdiagonal is not read.
     * @param triDiagonals
     * @param b
     * @param hand
     */
    void set(Real *triDiagonals, Real *b, Handle &hand);

    void solve(Handle &hand);
};



#endif //CUDABANDED_THOMAS_H
