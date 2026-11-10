/**
 * @file test_elementwise_ops.cpp
 * @brief GoogleTest direct verification of ElementwiseOps' five per-cell
 * static methods against their documented one-line formulas.
 *
 * Each method is a single mutating expression on FieldView::at(); the
 * reference below applies the same expression independently, through a
 * second FieldView over a separately-allocated copy, so this still checks
 * the real method rather than re-deriving its own copy of the arithmetic
 * inline with no comparison at all.
 */
#include <gtest/gtest.h>
#include "physics/ElementwiseOps.cuh"
#include "CfdTestHelpers.h"
#include <random>
#include <vector>

namespace {

std::mt19937 rng(20261108);
std::uniform_real_distribution<double> dist(-1.0, 1.0);
double R() { return dist(rng); }

/// @brief The grid sizes exercised by every test in this file.
const std::vector<int>& TestGridSizes() {
    static const std::vector<int> sizes = {3, 5, 8};
    return sizes;
}

/// @brief A random field of the given extent, as a flat FieldView-ready
/// buffer.
std::vector<double> RandomField(int nx, int ny, int nz) {
    std::vector<double> v((size_t)nx * (size_t)ny * (size_t)nz);
    for (auto& x : v) x = R();
    return v;
}

} // namespace

TEST(ElementwiseOpsTest, SubtractTimeDerivative) {
    cfdtest::MismatchCollector mismatches("subtractTimeDerivative");
    for (int n : TestGridSizes()) {
        const int nx = n, ny = n + 1, nz = n + 2;
        auto rhsRef = RandomField(nx, ny, nz);
        auto rhsImpl = rhsRef;
        auto current = RandomField(nx, ny, nz), old = RandomField(nx, ny, nz);
        const double coefficient = 0.1 + std::abs(R());
        FieldView<double> rhsRefView(rhsRef.data(), nx, ny, nz), rhsImplView(rhsImpl.data(), nx, ny, nz);
        FieldView<double> currentView(current.data(), nx, ny, nz), oldView(old.data(), nx, ny, nz);
        for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz; ++k) {
            rhsRefView.at(i, j, k) -= coefficient * (4.0 * currentView.at(i, j, k) - oldView.at(i, j, k));
            ElementwiseOps::subtractTimeDerivative(i, j, k, rhsImplView, currentView, oldView, coefficient);
        }
        for (size_t idx = 0; idx < rhsRef.size(); ++idx) mismatches.Check(rhsRef[idx], rhsImpl[idx]);
    }
    mismatches.Finish();
}

TEST(ElementwiseOpsTest, Scale) {
    cfdtest::MismatchCollector mismatches("scale");
    for (int n : TestGridSizes()) {
        const int nx = n, ny = n + 1, nz = n + 2;
        auto fieldRef = RandomField(nx, ny, nz);
        auto fieldImpl = fieldRef;
        const double factor = 0.1 + std::abs(R());
        FieldView<double> refView(fieldRef.data(), nx, ny, nz), implView(fieldImpl.data(), nx, ny, nz);
        for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz; ++k) {
            refView.at(i, j, k) *= factor;
            ElementwiseOps::scale(i, j, k, implView, factor);
        }
        for (size_t idx = 0; idx < fieldRef.size(); ++idx) mismatches.Check(fieldRef[idx], fieldImpl[idx]);
    }
    mismatches.Finish();
}

TEST(ElementwiseOpsTest, AddInto) {
    cfdtest::MismatchCollector mismatches("addInto");
    for (int n : TestGridSizes()) {
        const int nx = n, ny = n + 1, nz = n + 2;
        auto targetRef = RandomField(nx, ny, nz);
        auto targetImpl = targetRef;
        auto source = RandomField(nx, ny, nz);
        FieldView<double> refView(targetRef.data(), nx, ny, nz), implView(targetImpl.data(), nx, ny, nz);
        FieldView<double> sourceView(source.data(), nx, ny, nz);
        for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz; ++k) {
            refView.at(i, j, k) += sourceView.at(i, j, k);
            ElementwiseOps::addInto(i, j, k, implView, sourceView);
        }
        for (size_t idx = 0; idx < targetRef.size(); ++idx) mismatches.Check(targetRef[idx], targetImpl[idx]);
    }
    mismatches.Finish();
}

TEST(ElementwiseOpsTest, CorrectVelocity) {
    cfdtest::MismatchCollector mismatches("correctVelocity");
    for (int n : TestGridSizes()) {
        const int nx = n, ny = n + 1, nz = n + 2;
        auto velocityRef = RandomField(nx, ny, nz);
        auto velocityImpl = velocityRef;
        auto gradient = RandomField(nx, ny, nz);
        const double coefficient = 0.1 + std::abs(R());
        FieldView<double> refView(velocityRef.data(), nx, ny, nz), implView(velocityImpl.data(), nx, ny, nz);
        FieldView<double> gradientView(gradient.data(), nx, ny, nz);
        for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz; ++k) {
            refView.at(i, j, k) -= gradientView.at(i, j, k) * coefficient;
            ElementwiseOps::correctVelocity(i, j, k, implView, gradientView, coefficient);
        }
        for (size_t idx = 0; idx < velocityRef.size(); ++idx) mismatches.Check(velocityRef[idx], velocityImpl[idx]);
    }
    mismatches.Finish();
}

TEST(ElementwiseOpsTest, AccumulateScaled) {
    cfdtest::MismatchCollector mismatches("accumulateScaled");
    for (int n : TestGridSizes()) {
        const int nx = n, ny = n + 1, nz = n + 2;
        auto targetRef = RandomField(nx, ny, nz);
        auto targetImpl = targetRef;
        auto source = RandomField(nx, ny, nz);
        const double factor = 0.1 + std::abs(R());
        FieldView<double> refView(targetRef.data(), nx, ny, nz), implView(targetImpl.data(), nx, ny, nz);
        FieldView<double> sourceView(source.data(), nx, ny, nz);
        for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) for (int k = 1; k <= nz; ++k) {
            refView.at(i, j, k) += sourceView.at(i, j, k) * factor;
            ElementwiseOps::accumulateScaled(i, j, k, implView, sourceView, factor);
        }
        for (size_t idx = 0; idx < targetRef.size(); ++idx) mismatches.Check(targetRef[idx], targetImpl[idx]);
    }
    mismatches.Finish();
}
