/**
 * @file GPUScalarIntegrityEnvironment.cpp
 * @brief Global GoogleTest environment that checks GPUScalar<T>'s cached
 *        constants (0, 1, -1, 2, -2, for Real = float and double) still hold
 *        their correct values after the run -- catching not just the
 *        ImmersedEq::setRHSFPrime bug that prompted this, but any future
 *        code that makes the same mistake: passing a GPUScalar<Real>::get(...)
 *        result into a parameter that gets written through (such as
 *        Vec::subtract's `buffer` argument).
 *
 * GPUScalar<Real>::get(i, hand) does not construct a fresh Singleton<Real>:
 * it returns a reference to one process-wide, per-device Singleton, cached
 * forever in GPUScalar<Real>::instances (a static map -- Singleton.cu). Its
 * backing GPU memory is a *window* (GpuPointer::window()) into one 5-element
 * allocation, aliased -- "exactly like shared_ptr's own aliasing
 * constructor" (GpuPointer.h) -- by every copy anyone makes of it. Code that
 * writes through one of these constants corrupts it for the rest of the
 * process; that is exactly what setRHSFPrime's
 * RHSF.subtract(..., GPUScalar<Real>::get(1, hand), &hand) call does.
 *
 * SetUp() records the five constants, for both Real types, before any test
 * runs; TearDown() checks them after. Under ctest's gtest_discover_tests
 * (DISCOVERY_MODE PRE_TEST, see CMakeLists.txt), each discovered test runs
 * as its own process, so in practice this wraps exactly one test at a time
 * and pins a failure on it precisely. Run as the raw run_unit_tests binary
 * (every test, one process), it still catches the corruption, just
 * attributed to the end of the whole run rather than one test -- still
 * strictly more than today, where nothing checks this at all.
 *
 * Needs no change to any existing test file or to main(): registering a
 * GoogleTest Environment from a file-scope static initializer, as done at
 * the bottom of this file, is the documented way to add one without owning
 * main() (see AddGlobalTestEnvironment's own doc comment in gtest.h).
 */

#include <gtest/gtest.h>

#include "deviceArrays/headers/Singleton.h"
#include "deviceArrays/headers/handle.h"

namespace {

// The only integer values GPUScalar<Real>::get(i, hand) supports -- see its
// switch in Singleton.cu.
constexpr int kGpuScalarIndices[5] = {0, 1, -1, 2, -2};

template <typename Real>
struct GoldenFive {
    Real values[5] = {};
};

template <typename Real>
GoldenFive<Real> readFive(Handle& hand) {
    GoldenFive<Real> out;
    for (int slot = 0; slot < 5; ++slot)
        GPUScalar<Real>::get(kGpuScalarIndices[slot], hand).get(&out.values[slot], hand);
    hand.synch();
    return out;
}

template <typename Real>
void expectFiveUnchanged(const GoldenFive<Real>& golden, Handle& hand, const char* typeName) {
    GoldenFive<Real> actual = readFive<Real>(hand);
    for (int slot = 0; slot < 5; ++slot) {
        EXPECT_NEAR(actual.values[slot], golden.values[slot], 1e-9)
            << "GPUScalar<" << typeName << ">::get(" << kGpuScalarIndices[slot] << ", hand) "
            << "changed during this run: was " << golden.values[slot] << ", is now "
            << actual.values[slot] << ". Something wrote through a cached GPUScalar constant -- "
            << "check for a Vec::subtract (or similar) call passing GPUScalar<" << typeName
            << ">::get(...) as a mutable buffer/scratch argument.";
    }
}

/// Wraps the whole run (or, under ctest's one-process-per-test discovery,
/// each individual test) with a before/after check of every cached
/// GPUScalar constant.
class GPUScalarIntegrityEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        Handle hand;
        goldenDouble_ = readFive<double>(hand);
        goldenFloat_ = readFive<float>(hand);
    }

    void TearDown() override {
        Handle hand;
        expectFiveUnchanged(goldenDouble_, hand, "double");
        expectFiveUnchanged(goldenFloat_, hand, "float");
    }

private:
    GoldenFive<double> goldenDouble_;
    GoldenFive<float> goldenFloat_;
};

// Registering here rather than in main() means this environment is active
// no matter which file's main() ends up linked into run_unit_tests -- see
// AddGlobalTestEnvironment's doc comment in gtest.h for why this is the
// supported way to do that.
::testing::Environment* const g_gpuScalarIntegrityEnv =
    ::testing::AddGlobalTestEnvironment(new GPUScalarIntegrityEnvironment);

}  // namespace
