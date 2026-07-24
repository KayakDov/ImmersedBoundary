#include "EigenDecompSolver.h"

#include "../Event.h"
#include "deviceArrays/headers/Support/Streamable.h"
#include "deviceArrays/headers/Singleton.h"
#include "poisson/BoundaryConfig.cuh"

template<typename T>
bool EigenDecompSolver<T>::isInLColSpace(const Vec<T> &rhs, Vec<T> &bufferSizeOfB, Singleton<T> &bufferSing, double tolerance, Handle &hand) const {

    bufferSizeOfB.fill(1, hand);

    bufferSizeOfB.mult(rhs, bufferSing, &hand);

    T result = bufferSing.get(hand);

    return std::abs(result) < tolerance;
}

template<typename T>
__global__ void contains(T* result, DeviceData1d<T> vec1, DeviceData1d<T> vec2, DeviceData1d<T> vec3, T key, T tolerance) {
    auto id = idx();
    if (id < vec1.cols &&  vec1[id] - key <= tolerance && key - vec1[id] <= tolerance) *result = true;
    else if (id < vec2.cols &&  vec2[id] - key <= tolerance && key - vec2[id] <= tolerance) *result = true;
    else if (id < vec3.cols &&  vec3[id] - key <= tolerance && key - vec3[id] <= tolerance) *result = true;
}

template<typename T>
EigenDecompSolver<T>::EigenDecompSolver(const Eigen<T>& eMatsAndVecs, SimpleArray<T> &sizeOfB, bool allNeumann, T helmholtzShift) :
    dim(
        eMatsAndVecs.vecs.y._rows,
        eMatsAndVecs.vecs.x._rows,
        eMatsAndVecs.vecs.z._rows
    ),
    lapEigen(eMatsAndVecs),
    sizeOfB(sizeOfB),
    isSingular(allNeumann && helmholtzShift == 0),
    helmholtzShift(helmholtzShift) {
}

template<typename T>
template<typename BoundaryConfigT>
EigenDecompSolver<T>::EigenDecompSolver(const BoundaryConfigT& boundary, Handle* hands, Event* events, SimpleArray<T> sizeOfB, T helmholtzShift) :
    EigenDecompSolver(
        Eigen<T>::make(boundary, hands, events),
        sizeOfB,
        boundary.allNeumann(),
        helmholtzShift
    ){
}

template<typename T>
template<typename BoundaryConfigT>
EigenDecompSolver<T>::EigenDecompSolver(const BoundaryConfigT& boundary, Handle* hands, Event* events, T helmholtzShift) :
    EigenDecompSolver(
        boundary,
        hands,
        events,
        SimpleArray<T>::create(boundary.dim().size(), hands[0]),
        helmholtzShift
    ) {}

template<typename T>
SquareMat<T> EigenDecompSolver<T>::inverseL(Handle &hand) const {
    auto id = SquareMat<T>::create(this->dim.size());
    id.setToIdentity(hand);
    auto result = SquareMat<T>::create(id._rows);
    for (size_t i = 0; i < result._rows; ++i) {
        auto src = id.col(i);
        auto dst = result.col(i);
        solve(dst, src, hand);
    }
    return result;
}

template class EigenDecompSolver<double>;
template class EigenDecompSolver<float>;

// 2. Define a macro that specifically instantiates the two templated constructors
#define INSTANTIATE_EIGEN_DECOMP_SOLVER_CONSTRUCTORS(Real, SegX, SegY, SegZ) \
template EigenDecompSolver<Real>::EigenDecompSolver(                     \
const BoundaryConfig<Real, SegX, SegY, SegZ>& boundary,              \
Handle* hands,                                                       \
Event* events,                                                       \
SimpleArray<Real> sizeOfB,                                           \
Real helmholtzShift                                                  \
);                                                                       \
template EigenDecompSolver<Real>::EigenDecompSolver(                     \
const BoundaryConfig<Real, SegX, SegY, SegZ>& boundary,              \
Handle* hands,                                                       \
Event* events,                                                       \
Real helmholtzShift                                                  \
);

// 3. Invoke the 27-way permutation macro for both double and float
APPLY_TO_ALL_SEGMENT_COMBOS(double, INSTANTIATE_EIGEN_DECOMP_SOLVER_CONSTRUCTORS)
APPLY_TO_ALL_SEGMENT_COMBOS(float, INSTANTIATE_EIGEN_DECOMP_SOLVER_CONSTRUCTORS)