/**
 * @file CfdTestHelpers.h
 * @brief Shared GoogleTest helpers for the cfdSolver verification suite.
 */

#ifndef CFDSOLVER_TESTS_CFDTESTHELPERS_H
#define CFDSOLVER_TESTS_CFDTESTHELPERS_H

#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace cfdtest {

/**
 * @brief The relative/absolute tolerance used throughout this project's
 * verification work: within 1e-12 of the larger operand's magnitude.
 * @param a First value.
 * @param b Second value.
 * @return True if a and b are close enough to count as matching.
 */
inline bool IsClose(double a, double b) {
    double diff = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    return diff <= 1e-12 * scale;
}

/**
 * @brief Aggregates many individual cell-level comparisons (often in the
 * hundreds of thousands, across a full grid-size/boundary-condition
 * sweep) into a single GoogleTest failure, rather than one EXPECT_* call
 * per cell. Keeps output readable on failure -- a handful of example
 * mismatches plus a total count -- while still failing the test if any
 * single comparison is wrong.
 */
class MismatchCollector {
public:
    /// @param label Identifies which quantity this collector is checking,
    /// used in the failure message if any mismatch is found.
    explicit MismatchCollector(std::string label) : label_(std::move(label)) {}

    /// @brief Records one comparison. Call once per cell in a sweep.
    void Check(double reference, double actual) {
        ++totalChecks_;
        if (IsClose(reference, actual)) return;
        ++totalMismatches_;
        if (examples_.size() < kMaxExamples) {
            std::ostringstream oss;
            oss << "  mismatch #" << totalMismatches_ << ": reference=" << reference << " actual=" << actual;
            examples_.push_back(oss.str());
        }
    }

    /**
     * @brief Reports the aggregated result as one GoogleTest expectation.
     * Call exactly once, after every Check() call for this collector.
     */
    void Finish() const {
        std::ostringstream detail;
        for (const auto& line : examples_) detail << line << "\n";
        if (totalMismatches_ > (int)kMaxExamples)
            detail << "  ... and " << (totalMismatches_ - (int)kMaxExamples) << " more\n";
        EXPECT_EQ(totalMismatches_, 0)
            << label_ << ": " << totalMismatches_ << " of " << totalChecks_ << " checks mismatched\n"
            << detail.str();
    }

private:
    static constexpr size_t kMaxExamples = 5;
    std::string label_;
    int totalChecks_ = 0;
    int totalMismatches_ = 0;
    std::vector<std::string> examples_;
};

/**
 * @brief A ghost-padded 3D array: 1-indexed over [0, N+1] with one ghost
 * layer on every face. Shared by every host-only test that checks a
 * production finite-difference operator against a hand-written,
 * ghost-padded reference (test_operators_host.cpp,
 * test_electromagnetic_force.cpp).
 */
struct PaddedField {
    int nx, ny, nz;
    std::vector<double> data;
    PaddedField(int x, int y, int z) : nx(x), ny(y), nz(z), data((size_t)(x + 1) * (y + 1) * (z + 1), 0.0) {}
    double& at(int i, int j, int k) { return data[(size_t)i + (size_t)(nx + 1) * ((size_t)j + (size_t)(ny + 1) * (size_t)k)]; }
    double at(int i, int j, int k) const { return data[(size_t)i + (size_t)(nx + 1) * ((size_t)j + (size_t)(ny + 1) * (size_t)k)]; }
};

/// @brief Zeroes a PaddedField's velocity ghosts on every face (velocity is always Dirichlet-zero).
inline void FillVelGhostsZero(PaddedField& v) {
    for (int j = 0; j <= v.ny; ++j) for (int k = 0; k <= v.nz; ++k) { v.at(0, j, k) = 0.0; v.at(v.nx, j, k) = 0.0; }
    for (int i = 0; i <= v.nx; ++i) for (int k = 0; k <= v.nz; ++k) { v.at(i, 0, k) = 0.0; v.at(i, v.ny, k) = 0.0; }
    for (int i = 0; i <= v.nx; ++i) for (int j = 0; j <= v.ny; ++j) { v.at(i, j, 0) = 0.0; v.at(i, j, v.nz) = 0.0; }
}

/// @brief Flattens a 1-indexed (i,j,k) triple into an unpadded array offset.
inline size_t Uidx(int i, int j, int k, int nx, int ny) { return (size_t)(i - 1) + (size_t)nx * ((size_t)(j - 1) + (size_t)ny * (size_t)(k - 1)); }

/// @brief The grid sizes exercised by every test that sweeps grid size in this suite.
inline const std::vector<int>& TestGridSizes() {
    static const std::vector<int> sizes = {5, 8, 13};
    return sizes;
}

} // namespace cfdtest

#endif //CFDSOLVER_TESTS_CFDTESTHELPERS_H
