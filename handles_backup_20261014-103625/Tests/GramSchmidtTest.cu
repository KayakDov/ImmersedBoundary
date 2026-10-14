/**
 * @file GramSchmidtTest.cu
 * @brief Checks Mat::gramSchmidt, spread over several handles, against a CPU reference.
 *
 * Isolates the multi-handle Gram-Schmidt from the Lyapunov code. If these
 * pass, the fault is in how LyapunovExponent uses it; if they fail, the
 * printed column and value say where.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "deviceArrays/headers/Mat.h"
#include "deviceArrays/headers/Vec.h"
#include "deviceArrays/headers/handle.h"
#include "solvers/Handles.h"

namespace {

constexpr size_t kRows = 3;
constexpr size_t kCols = 3;

/**
 * @brief Column-major modified Gram-Schmidt on the host.
 * @param a      kRows x kCols input, column-major.
 * @param q      Receives the orthonormal columns.
 * @param lengths Receives each column's length just before it was normalized.
 */
void hostGramSchmidt(const std::vector<double>& a, std::vector<double>& q, std::vector<double>& lengths) {
    q = a;
    lengths.assign(kCols, 0.0);
    for (size_t col = 0; col < kCols; ++col) {
        for (size_t previous = 0; previous < col; ++previous) {
            double dot = 0.0;
            for (size_t row = 0; row < kRows; ++row)
                dot += q[previous * kRows + row] * q[col * kRows + row];
            for (size_t row = 0; row < kRows; ++row)
                q[col * kRows + row] -= dot * q[previous * kRows + row];
        }
        double sum = 0.0;
        for (size_t row = 0; row < kRows; ++row)
            sum += q[col * kRows + row] * q[col * kRows + row];
        lengths[col] = std::sqrt(sum);
        for (size_t row = 0; row < kRows; ++row)
            q[col * kRows + row] /= lengths[col];
    }
}

/**
 * @brief Runs the device Gram-Schmidt on a and compares it with the host one.
 * @param a        The input matrix, column-major.
 * @param inPlace  Whether dst is the source matrix.
 */
void checkAgainstHost(const std::vector<double>& a, bool inPlace) {
    Handles hands(kCols);

    Mat<double> src = Mat<double>::create(kRows, kCols, hands.get(0));
    Mat<double> other = Mat<double>::create(kRows, kCols, hands.get(0));
    Vec<double> lengths = Vec<double>::create(kCols, hands.get(0));
    Vec<double> inverseLengths = Vec<double>::create(kCols, hands.get(0));

    src.set(a.data(), kRows, hands.get(0));
    Mat<double>& dst = inPlace ? src : other;

    src.gramSchmidt(dst, lengths, inverseLengths, hands);

    hands.synch();

    std::vector<double> q(kRows * kCols), r(kCols), inverse(kCols);
    dst.get(q.data(), kRows, hands.get(0));
    lengths.get(r.data(), 1, hands.get(0));
    inverseLengths.get(inverse.data(), 1, hands.get(0));
    hands.get(0).synch();

    std::vector<double> expectedQ, expectedR;
    hostGramSchmidt(a, expectedQ, expectedR);

    for (size_t col = 0; col < kCols; ++col) {
        EXPECT_NEAR(r[col], expectedR[col], 1e-10 * expectedR[col]) << "length of column " << col;
        EXPECT_NEAR(inverse[col], 1.0 / expectedR[col], 1e-10 / expectedR[col]) << "inverse length of column " << col;
        for (size_t row = 0; row < kRows; ++row)
            EXPECT_NEAR(q[col * kRows + row], expectedQ[col * kRows + row], 1e-10)
                << "Q(" << row << ", " << col << ")";
    }
}

const std::vector<double> kGeneral = {
    1.0, 2.0, 3.0,
    4.0, 5.0, 7.0,
    0.5, -1.0, 2.0
};

const std::vector<double> kNearlyParallel = {
    1.0, 1.0, 1.0,
    1.0, 1.0 + 1e-6, 1.0,
    1.0, 1.0, 1.0 + 2e-6
};

}

TEST(GramSchmidt, OutOfPlaceMatchesHost) {
    checkAgainstHost(kGeneral, false);
}

TEST(GramSchmidt, InPlaceMatchesHost) {
    checkAgainstHost(kGeneral, true);
}

TEST(GramSchmidt, NearlyParallelColumnsMatchHost) {
    checkAgainstHost(kNearlyParallel, true);
}

TEST(GramSchmidt, RepeatedCallsOnTheSameWorkspaceMatchHost) {
    Handles hands(kCols);

    Mat<double> matrix = Mat<double>::create(kRows, kCols, hands.get(0));
    Vec<double> lengths = Vec<double>::create(kCols, hands.get(0));
    Vec<double> inverseLengths = Vec<double>::create(kCols, hands.get(0));

    for (int repeat = 0; repeat < 20; ++repeat) {
        std::vector<double> a = (repeat % 2 == 0) ? kGeneral : kNearlyParallel;
        a[0] += 0.01 * repeat;

        matrix.set(a.data(), kRows, hands.get(0));
        matrix.gramSchmidt(matrix, lengths, inverseLengths, hands);

        hands.synch();

        std::vector<double> q(kRows * kCols), r(kCols);
        matrix.get(q.data(), kRows, hands.get(0));
        lengths.get(r.data(), 1, hands.get(0));
        hands.get(0).synch();

        std::vector<double> expectedQ, expectedR;
        hostGramSchmidt(a, expectedQ, expectedR);

        for (size_t col = 0; col < kCols; ++col) {
            ASSERT_NEAR(r[col], expectedR[col], 1e-8) << "repeat " << repeat << ", length of column " << col;
            for (size_t row = 0; row < kRows; ++row)
                ASSERT_NEAR(q[col * kRows + row], expectedQ[col * kRows + row], 1e-8)
                    << "repeat " << repeat << ", Q(" << row << ", " << col << ")";
        }
    }
}
