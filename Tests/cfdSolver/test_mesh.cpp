/**
 * @file test_mesh.cpp
 * @brief GoogleTest direct verification of Mesh's tanh-stretched grid
 * generation (computeAxisSpacing(), declared in Mesh.h -- see its own doc
 * comment for why it lives there rather than file-local to Mesh.cu),
 * checked against an independently-written reference transcription of
 * MeshStretch.f90's per-axis block.
 *
 * Mesh<Real>::generate() itself (the GPU-upload wrapper around this
 * function) is out of reach of a host-only test: GpuIndex's constructor
 * calls cudaSetDevice(), which needs a real CUDA device. This file closes
 * the part of the gap that doesn't need one: the actual node-position,
 * cell-spacing, and control-volume-spacing arithmetic that generate()
 * calls once per axis.
 */
#include <gtest/gtest.h>
#include "setup/Mesh.h"
#include "CfdTestHelpers.h"
#include <cmath>
#include <vector>

namespace {

/// @brief Independent transcription of MeshStretch.f90's per-axis block --
/// the same formula computeAxisSpacing() implements -- so this test checks
/// the real function rather than comparing it to itself.
template<typename Real>
AxisSpacing<Real> ReferenceAxisSpacing(int n, Real scale, Real stretchFactor) {
    const int n1 = n + 1;
    const int n2 = n1 + 1;
    std::vector<Real> node(n1 + 1), nodeHalf(n2 + 1);

    const Real h = Real(1) / Real(n1);
    node[0] = Real(0);
    for (int i = 1; i <= n1; ++i) node[i] = node[i - 1] + h;
    const Real tanhHalfStretch = std::tanh(Real(0.5) * stretchFactor);
    for (int i = 0; i <= n1; ++i)
        node[i] = scale * (Real(0.5) + Real(0.5) * std::tanh(stretchFactor * (node[i] - Real(0.5))) / tanhHalfStretch);

    nodeHalf[0] = Real(0);
    nodeHalf[n2] = scale;
    for (int i = 1; i <= n1; ++i) nodeHalf[i] = Real(0.5) * (node[i] + node[i - 1]);

    AxisSpacing<Real> result;
    result.hFull.resize(n + 1);
    for (int i = 0; i <= n; ++i) result.hFull[i] = node[i + 1] - node[i];
    result.hHalf.resize(n1 + 1);
    for (int i = 0; i <= n1; ++i) result.hHalf[i] = nodeHalf[i + 1] - nodeHalf[i];
    result.node = node;
    result.nodeHalf = nodeHalf;
    return result;
}

/// @brief Checks every element of all four arrays computeAxisSpacing()
/// returns, for one (n, scale, stretchFactor) combination.
void CheckAxis(int n, double scale, double stretchFactor, cfdtest::MismatchCollector& mismatches) {
    const AxisSpacing<double> ref = ReferenceAxisSpacing<double>(n, scale, stretchFactor);
    const AxisSpacing<double> impl = computeAxisSpacing<double>(n, scale, stretchFactor);
    ASSERT_EQ(ref.hFull.size(), impl.hFull.size());
    ASSERT_EQ(ref.hHalf.size(), impl.hHalf.size());
    ASSERT_EQ(ref.node.size(), impl.node.size());
    ASSERT_EQ(ref.nodeHalf.size(), impl.nodeHalf.size());
    for (size_t i = 0; i < ref.hFull.size(); ++i) mismatches.Check(ref.hFull[i], impl.hFull[i]);
    for (size_t i = 0; i < ref.hHalf.size(); ++i) mismatches.Check(ref.hHalf[i], impl.hHalf[i]);
    for (size_t i = 0; i < ref.node.size(); ++i) mismatches.Check(ref.node[i], impl.node[i]);
    for (size_t i = 0; i < ref.nodeHalf.size(); ++i) mismatches.Check(ref.nodeHalf[i], impl.nodeHalf[i]);
}

} // namespace

TEST(MeshTest, ComputeAxisSpacingAcrossGridSizesAndStretchFactors) {
    cfdtest::MismatchCollector mismatches("computeAxisSpacing");
    const std::vector<int> sizes = {4, 7, 12};
    /* 10.0 is the only value this project's golden-comparison harness ever
       actually uses (GoldenSupport.h's renderConvDat() hard-errors on any
       other mesh.stretching_factor), but the formula itself is general, so
       this also sweeps two other stretch factors and a near-uniform one. */
    const std::vector<double> stretchFactors = {1.0, 3.0, 10.0};
    /* aspectRatio-like, widthRatio-like, and z's typically-1.0-like scales. */
    const std::vector<double> scales = {1.0, 2.5, 0.4};
    for (int n : sizes)
        for (double stretch : stretchFactors)
            for (double scale : scales)
                CheckAxis(n, scale, stretch, mismatches);
    mismatches.Finish();
}

TEST(MeshTest, EndpointsAreExactDomainBoundaries) {
    /* node[0]==0 and node[n+1]==scale (and likewise for nodeHalf) on every
       axis: the tanh remap is built so the two domain walls land exactly
       on 0 and scale regardless of stretch factor, since
       tanh(+-stretchFactor*0.5)/tanhHalfStretch == +-1 there by
       construction. A meaningful property beyond the reference-vs-impl
       diff above, since both compute the same expression at the boundary. */
    for (int n : {3, 9}) {
        for (double scale : {1.0, 2.5}) {
            const AxisSpacing<double> impl = computeAxisSpacing<double>(n, scale, 5.0);
            EXPECT_TRUE(cfdtest::IsClose(impl.node.front(), 0.0));
            EXPECT_TRUE(cfdtest::IsClose(impl.node.back(), scale));
            EXPECT_TRUE(cfdtest::IsClose(impl.nodeHalf.front(), 0.0));
            EXPECT_TRUE(cfdtest::IsClose(impl.nodeHalf.back(), scale));
        }
    }
}
