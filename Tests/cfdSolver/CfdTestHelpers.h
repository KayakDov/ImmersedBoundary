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

} // namespace cfdtest

#endif //CFDSOLVER_TESTS_CFDTESTHELPERS_H
