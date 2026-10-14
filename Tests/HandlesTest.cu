/**
 * @file HandlesTest.cu
 * @brief Tests for Handles: construction, bounds, and that each waiting
 *        method lets work on one handle see the finished work of another.
 *
 * Ordering can only be shown to hold, not to fail, by a test on one
 * machine: a missing wait may still happen to pass. The large vectors make
 * the producing fill long enough that a missing wait would usually show.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "deviceArrays/headers/Vec.h"
#include "deviceArrays/headers/Handles.h"
#include "deviceArrays/headers/Singleton.h"

namespace {

constexpr size_t kLength = size_t{1} << 24;

/**
 * @brief Whether the first and last entries of a device vector hold a value.
 */
bool endsAre(const Vec<double>& vec, double value, Handle& hand) {
    std::vector<double> host(kLength);
    vec.get(host.data(), 1, hand);
    hand.synch();
    return host.front() == value && host.back() == value;
}

}

TEST(Handles, RejectsZeroHandles) {
    EXPECT_THROW(Handles(0), std::invalid_argument);
}

TEST(Handles, ReportsItsSizeAndChecksIndices) {
    Handles hands(3);
    EXPECT_EQ(hands.size(), size_t{3});
    EXPECT_NO_THROW(hands[2]);
    EXPECT_THROW(hands[3], std::out_of_range);
    EXPECT_THROW(hands.waitFor(0, 3), std::out_of_range);
    EXPECT_THROW(hands.waitFor(std::vector<size_t>{0, 5}, std::vector<size_t>{1}), std::out_of_range);
    EXPECT_THROW(hands.waitForAll(3), std::out_of_range);
    EXPECT_THROW(hands.allWaitFor(3), std::out_of_range);
}

TEST(Handles, WaitingForYourselfDoesNothing) {
    Handles hands(2);
    EXPECT_NO_THROW(hands.waitFor(1, 1));
    EXPECT_NO_THROW(hands.waitFor(std::vector<size_t>{0, 1}, std::vector<size_t>{0, 1}));
    hands.synch();
}

TEST(Handles, OneWaitsForOne) {
    Handles hands(2);
    auto source = Vec<double>::create(kLength, hands[0]);
    auto copy = Vec<double>::create(kLength, hands[1]);

    source.fill(7.0, hands[0]);
    hands.waitFor(0, 1);
    copy.set(source, hands[1]);

    EXPECT_TRUE(endsAre(copy, 7.0, hands[1]));
}

TEST(Handles, OneWaitsForAll) {
    Handles hands(3);
    auto first = Vec<double>::create(kLength, hands[0]);
    auto second = Vec<double>::create(kLength, hands[1]);
    auto sum = Vec<double>::create(kLength, hands[2]);

    first.fill(3.0, hands[0]);
    second.fill(4.0, hands[1]);
    sum.fill(0.0, hands[2]);
    hands.waitForAll(2);
    sum.add(first, &GPUScalar<double>::get(1, hands[2]), &hands[2]);
    sum.add(second, &GPUScalar<double>::get(1, hands[2]), &hands[2]);

    EXPECT_TRUE(endsAre(sum, 7.0, hands[2]));
}

TEST(Handles, AllWaitForOne) {
    Handles hands(3);
    auto source = Vec<double>::create(kLength, hands[0]);
    auto firstCopy = Vec<double>::create(kLength, hands[1]);
    auto secondCopy = Vec<double>::create(kLength, hands[2]);

    source.fill(5.0, hands[0]);
    hands.allWaitFor(0);
    firstCopy.set(source, hands[1]);
    secondCopy.set(source, hands[2]);

    EXPECT_TRUE(endsAre(firstCopy, 5.0, hands[1]));
    EXPECT_TRUE(endsAre(secondCopy, 5.0, hands[2]));
}

TEST(Handles, GroupWaitsForGroup) {
    Handles hands(4);
    auto first = Vec<double>::create(kLength, hands[0]);
    auto second = Vec<double>::create(kLength, hands[1]);
    auto firstCopy = Vec<double>::create(kLength, hands[2]);
    auto secondCopy = Vec<double>::create(kLength, hands[3]);

    first.fill(1.0, hands[0]);
    second.fill(2.0, hands[1]);
    hands.waitFor({0, 1}, {2, 3});
    firstCopy.set(first, hands[2]);
    secondCopy.set(second, hands[3]);

    EXPECT_TRUE(endsAre(firstCopy, 1.0, hands[2]));
    EXPECT_TRUE(endsAre(secondCopy, 2.0, hands[3]));
}

TEST(Handles, MarkIsWaitedForLater) {
    Handles hands(2);
    auto source = Vec<double>::create(kLength, hands[0]);
    auto unrelated = Vec<double>::create(kLength, hands[0]);
    auto copy = Vec<double>::create(kLength, hands[1]);

    source.fill(7.0, hands[0]);
    hands.markCompletion(0);
    unrelated.fill(1.0, hands[0]);
    unrelated.fill(2.0, hands[0]);
    hands.waitForMark(0, 1);
    copy.set(source, hands[1]);

    EXPECT_TRUE(endsAre(copy, 7.0, hands[1]));
}

TEST(Handles, MarksInDifferentSlotsAreIndependent) {
    Handles hands(2);
    auto first = Vec<double>::create(kLength, hands[0]);
    auto second = Vec<double>::create(kLength, hands[0]);
    auto firstCopy = Vec<double>::create(kLength, hands[1]);
    auto secondCopy = Vec<double>::create(kLength, hands[1]);

    first.fill(1.0, hands[0]);
    hands.markCompletion(0, 0);
    second.fill(2.0, hands[0]);
    hands.markCompletion(0, 1);

    hands.waitForMark(0, 1, 0);
    firstCopy.set(first, hands[1]);
    hands.waitForMark(0, 1, 1);
    secondCopy.set(second, hands[1]);

    EXPECT_TRUE(endsAre(firstCopy, 1.0, hands[1]));
    EXPECT_TRUE(endsAre(secondCopy, 2.0, hands[1]));
}

TEST(Handles, WaitForMarkNeverMarkedOrOnYourselfDoesNothing) {
    Handles hands(2);
    EXPECT_NO_THROW(hands.waitForMark(0, 1));
    EXPECT_NO_THROW(hands.waitForMark(0, 1, 5));
    hands.markCompletion(1);
    EXPECT_NO_THROW(hands.waitForMark(1, 1));
    EXPECT_THROW(hands.markCompletion(2), std::out_of_range);
    EXPECT_THROW(hands.waitForMark(0, 2), std::out_of_range);
    hands.synch();
}
