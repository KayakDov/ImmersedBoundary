
#ifndef CUDABANDED_NORM_H
#define CUDABANDED_NORM_H

#include "deviceArrays/headers/Vec.h"
#include <cmath>

/**
 * @brief How the length of a deviation vector is measured when working
 *        out how fast it grew.
 *
 * The default is the Euclidean norm over every entry. A pseudo-norm,
 * which looks at only some entries -- one coordinate, say -- can be
 * substituted. It only decides how growth is measured: Gram-Schmidt and
 * the reset of the perturbed points stay Euclidean, so a pseudo-norm
 * gives a perturbation a length of its own under it, which
 * LyapunovExponent remembers and divides by.
 *
 * A pseudo-norm used this way must not measure any deviation as zero.
 */
template<typename Real>
class Norm {
public:
    virtual ~Norm() = default;

    /**
     * @brief Writes the length of @p src under this norm to @p result,
     *        on @p handle's stream.
     */
    virtual void measure(
        const Vec<Real>& src,
        Singleton<Real> result,
        Handle& handle
    ) const {
        src.norm(result, handle);
    };
};

template<typename Real>
const Norm<Real>& euclideanNorm() {
    static const Norm<Real> instance;
    return instance;
}

template<typename Real>
__global__ void abs(Real* src, Real* dst) {
    if (size_t id = idx(); id == 0) *dst = *src < 0 ? -*src : *src;
}

template<typename Real>
class Norm1d : public Norm<Real> {
public:
    const size_t dim;

    Norm1d(const size_t dim) : dim(dim) {}

    void measure(
        const Vec<Real>& src,
        Singleton<Real> result,
        Handle& handle
    ) const override {
        abs<<<1, 1, 0, handle>>>(src.get(dim).data(), result.data());

    }
};

/**
 * @brief The Euclidean norm over a window of rows, first through
 *        first + count - 1.
 *
 * The many-row counterpart of Norm1d. The vector is assumed contiguous,
 * as a column of a Mat is.
 */
template<typename Real>
class NormRows : public Norm<Real> {
public:
    const size_t first;
    const size_t count;

    NormRows(const size_t first, const size_t count) : first(first), count(count) {}

    void measure(
        const Vec<Real>& src,
        Singleton<Real> result,
        Handle& handle
    ) const override {
        src.vec(first, 1, count).norm(result, handle);
    }
};

#endif //CUDABANDED_NORM_H
