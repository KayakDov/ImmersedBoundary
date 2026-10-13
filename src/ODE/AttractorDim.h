
#ifndef CUDABANDED_ATTRACTORDIM_H
#define CUDABANDED_ATTRACTORDIM_H

/**
 * @file AttractorDimension.h
 * @brief Computes the Kaplan-Yorke (Lyapunov) dimension of an attractor.
 */

#ifndef ATTRACTOR_DIMENSION_H
#define ATTRACTOR_DIMENSION_H

#include "LyapunovExponent.h"
#include "Norm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

/**
 * @brief Computes the Kaplan-Yorke (Lyapunov) dimension of an attractor from
 *        its Lyapunov exponent spectrum.
 *
 * The Kaplan-Yorke dimension $D_{KY}$ estimates the fractal dimension of an
 * attractor using the ordered spectrum of Lyapunov exponents
 * $\lambda_0 \ge \lambda_1 \ge \dots \ge \lambda_{n-1}$:
 *
 * \f[
 *     D_{KY} = k + \frac{\sum_{i=0}^{k-1} \lambda_i}{|\lambda_k|}
 * \f]
 *
 * where $k$ is the maximum number of exponents whose sum remains non-negative
 * ($\sum_{i=0}^{k-1} \lambda_i \ge 0$).
 *
 * If $\lambda_0 < 0$, the dimension is 0 (indicating a stable fixed point).
 * If the sum of all $n$ exponents remains non-negative, the computed dimension
 * saturates at $n$.
 *
 * @tparam Real Floating-point type.
 */
template<typename Real>
class AttractorDim {
public:
    /**
     * @brief Computes the Kaplan-Yorke dimension directly from a spectrum of exponents.
     *
     * Sorts the exponents in descending order before determining the expansion
     * balance point.
     *
     * @param exponents Vector of Lyapunov exponents.
     * @return Calculated Kaplan-Yorke dimension.
     */
    static Real kaplanYorke(std::vector<Real> exponents) {
        if (exponents.empty()) {
            return static_cast<Real>(0);
        }

        std::sort(exponents.begin(), exponents.end(), std::greater<Real>());

        if (exponents[0] < static_cast<Real>(0)) {
            return static_cast<Real>(0);
        }

        Real cumulativeSum = static_cast<Real>(0);
        Real previousSum = static_cast<Real>(0);

        for (size_t k = 0; k < exponents.size(); ++k) {
            previousSum = cumulativeSum;
            cumulativeSum += exponents[k];

            if (cumulativeSum < static_cast<Real>(0)) {
                return static_cast<Real>(k) + (previousSum / std::abs(exponents[k]));
            }
        }

        return static_cast<Real>(exponents.size());
    }

    /**
     * @param lyapunov Reference to the LyapunovExponent solver instance used
     *                 to compute the spectrum. Stored by reference, never copied;
     *                 must outlive this object.
     */
    explicit AttractorDim(const LyapunovExponent<Real>& lyapunov)
        : lyapunov_(lyapunov) {}

    /**
     * @brief Computes the full Lyapunov spectrum and calculates the Kaplan-Yorke dimension.
     *
     * @param initialPoint Baseline state vector.
     * @param numIntervals Number of integration intervals.
     * @param stepsPerInterval Integration steps per interval.
     * @param epsilon Target perturbation magnitude.
     * @param dt Integration step size.
     * @param warmupSteps Trajectory steps taken prior to measurement.
     * @param norm How growth is measured, Euclidean by default; see
     *             LyapunovExponent.
     * @return Calculated Kaplan-Yorke dimension.
     */
    Real compute(
        const Vec<Real>& initialPoint,
        size_t numIntervals,
        size_t stepsPerInterval,
        Real epsilon,
        Real dt = static_cast<Real>(1),
        size_t warmupSteps = 0,
        const Norm<Real>& norm = euclideanNorm<Real>()
    ) const {
        std::vector<Real> exponents = lyapunov_.compute(
            initialPoint,
            numIntervals,
            stepsPerInterval,
            epsilon,
            dt,
            warmupSteps,
            norm
        );

        return kaplanYorke(exponents);
    }

private:
    const LyapunovExponent<Real>& lyapunov_;
};

#endif // ATTRACTOR_DIMENSION_H

#endif //CUDABANDED_ATTRACTORDIM_H
