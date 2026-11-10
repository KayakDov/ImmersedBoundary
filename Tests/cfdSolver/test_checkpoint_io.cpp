/**
 * @file test_checkpoint_io.cpp
 * @brief GoogleTest direct verification of CheckpointIO's on-disk binary
 * round trip (writeBuffers()/readBuffers(), declared and defined in
 * CheckpointIO.h -- see its own doc comment for why they live there
 * rather than file-local to CheckpointIO.cu).
 *
 * CheckpointIO::write()/read() themselves are out of reach of a host-only
 * test: constructing a TimeDependentField, GpuField, or Handle always
 * touches a real CUDA device (see SolvedField's and Handle's
 * constructors), none of which exist in this sandbox. write()/read() are
 * now thin delegators to writeBuffers()/readBuffers() (see
 * CheckpointIO.cu), so testing those two functions directly exercises the
 * exact byte-level format production code writes and reads, not a
 * parallel reimplementation of it.
 */
#include <gtest/gtest.h>
#include "setup/CheckpointIO.h"
#include "CfdTestHelpers.h"
#include "GoldenSupport.h"
#include <filesystem>
#include <string>
#include <vector>

namespace {

/// @brief One grid's worth of synthetic data for all 9 arrays
/// CheckpointIO saves, plus an elapsed time, sized exactly per
/// CheckpointIO's documented extents.
struct SyntheticCheckpoint {
    int nx, ny, nz;
    std::vector<double> temperatureCurrent, vxCurrent, vyCurrent, vzCurrent, pressureCurrent;
    std::vector<double> temperatureOld, vxOld, vyOld, vzOld;
    double elapsedTime;
};

/// @brief Fills `data` with values distinguishable both by array identity
/// (via `tag`) and by position (via the running index), so a swapped
/// array, a wrong offset, or a truncated read fails rather than passing
/// by coincidence.
void FillPattern(std::vector<double>& data, double tag) {
    for (size_t i = 0; i < data.size(); ++i) data[i] = tag + 1e-3 * static_cast<double>(i);
}

/// @brief Builds one synthetic checkpoint sized per CheckpointIO's
/// documented extents: temperatureSize=(nx+1)(ny+1)(nz+1),
/// vxSize=nx(ny+1)(nz+1), vySize=(nx+1)ny(nz+1), vzSize=(nx+1)(ny+1)nz,
/// pressureSize=temperatureSize.
SyntheticCheckpoint MakeSynthetic(int nx, int ny, int nz) {
    SyntheticCheckpoint s;
    s.nx = nx; s.ny = ny; s.nz = nz;
    const size_t temperatureSize = (size_t)(nx + 1) * (ny + 1) * (nz + 1);
    const size_t vxSize = (size_t)nx * (ny + 1) * (nz + 1);
    const size_t vySize = (size_t)(nx + 1) * ny * (nz + 1);
    const size_t vzSize = (size_t)(nx + 1) * (ny + 1) * nz;

    s.temperatureCurrent.resize(temperatureSize);
    s.vxCurrent.resize(vxSize);
    s.vyCurrent.resize(vySize);
    s.vzCurrent.resize(vzSize);
    s.pressureCurrent.resize(temperatureSize);
    s.temperatureOld.resize(temperatureSize);
    s.vxOld.resize(vxSize);
    s.vyOld.resize(vySize);
    s.vzOld.resize(vzSize);

    FillPattern(s.temperatureCurrent, 1.0);
    FillPattern(s.vxCurrent, 2.0);
    FillPattern(s.vyCurrent, 3.0);
    FillPattern(s.vzCurrent, 4.0);
    FillPattern(s.pressureCurrent, 5.0);
    FillPattern(s.temperatureOld, 6.0);
    FillPattern(s.vxOld, 7.0);
    FillPattern(s.vyOld, 8.0);
    FillPattern(s.vzOld, 9.0);
    s.elapsedTime = 123.456;
    return s;
}

/// @brief Writes `s` via the real CheckpointIO::writeBuffers(), reads it
/// back via the real CheckpointIO::readBuffers() into fresh vectors, and
/// checks every element of all 9 arrays plus the elapsed time against
/// what was written.
void CheckRoundTrip(const SyntheticCheckpoint& s, const std::string& path) {
    CheckpointIO::writeBuffers<double>(path, s.nx, s.ny, s.nz,
        s.temperatureCurrent, s.vxCurrent, s.vyCurrent, s.vzCurrent, s.pressureCurrent,
        s.temperatureOld, s.vxOld, s.vyOld, s.vzOld, s.elapsedTime);

    std::vector<double> temperatureCurrent, vxCurrent, vyCurrent, vzCurrent, pressureCurrent;
    std::vector<double> temperatureOld, vxOld, vyOld, vzOld;
    const double elapsedTime = CheckpointIO::readBuffers<double>(path, s.nx, s.ny, s.nz,
        temperatureCurrent, vxCurrent, vyCurrent, vzCurrent, pressureCurrent,
        temperatureOld, vxOld, vyOld, vzOld);

    ASSERT_EQ(s.temperatureCurrent.size(), temperatureCurrent.size());
    ASSERT_EQ(s.vxCurrent.size(), vxCurrent.size());
    ASSERT_EQ(s.vyCurrent.size(), vyCurrent.size());
    ASSERT_EQ(s.vzCurrent.size(), vzCurrent.size());
    ASSERT_EQ(s.pressureCurrent.size(), pressureCurrent.size());
    ASSERT_EQ(s.temperatureOld.size(), temperatureOld.size());
    ASSERT_EQ(s.vxOld.size(), vxOld.size());
    ASSERT_EQ(s.vyOld.size(), vyOld.size());
    ASSERT_EQ(s.vzOld.size(), vzOld.size());

    cfdtest::MismatchCollector mismatches("CheckpointIO round trip");
    for (size_t i = 0; i < temperatureCurrent.size(); ++i) mismatches.Check(s.temperatureCurrent[i], temperatureCurrent[i]);
    for (size_t i = 0; i < vxCurrent.size(); ++i) mismatches.Check(s.vxCurrent[i], vxCurrent[i]);
    for (size_t i = 0; i < vyCurrent.size(); ++i) mismatches.Check(s.vyCurrent[i], vyCurrent[i]);
    for (size_t i = 0; i < vzCurrent.size(); ++i) mismatches.Check(s.vzCurrent[i], vzCurrent[i]);
    for (size_t i = 0; i < pressureCurrent.size(); ++i) mismatches.Check(s.pressureCurrent[i], pressureCurrent[i]);
    for (size_t i = 0; i < temperatureOld.size(); ++i) mismatches.Check(s.temperatureOld[i], temperatureOld[i]);
    for (size_t i = 0; i < vxOld.size(); ++i) mismatches.Check(s.vxOld[i], vxOld[i]);
    for (size_t i = 0; i < vyOld.size(); ++i) mismatches.Check(s.vyOld[i], vyOld[i]);
    for (size_t i = 0; i < vzOld.size(); ++i) mismatches.Check(s.vzOld[i], vzOld[i]);
    mismatches.Finish();

    EXPECT_TRUE(cfdtest::IsClose(elapsedTime, s.elapsedTime));
}

} // namespace

TEST(CheckpointIOTest, RoundTripAcrossGridSizes) {
    const auto dir = cfd_golden::uniqueRunDirectory("checkpoint_io");
    int caseIndex = 0;
    /* Deliberately distinct nx/ny/nz per case (never nx==ny==nz) so a
       dimension mixed up with another in an offset calculation would show
       up as a mismatch rather than passing by coincidence. */
    for (int nx : {3, 5}) {
        for (int ny : {4, 6}) {
            for (int nz : {2, 7}) {
                const SyntheticCheckpoint s = MakeSynthetic(nx, ny, nz);
                CheckRoundTrip(s, (dir / ("case_" + std::to_string(caseIndex++) + ".bin")).string());
            }
        }
    }
}

TEST(CheckpointIOTest, ReadThrowsWhenGridExtentsDoNotMatch) {
    const auto dir = cfd_golden::uniqueRunDirectory("checkpoint_io");
    const SyntheticCheckpoint s = MakeSynthetic(4, 5, 3);
    const std::string path = (dir / "mismatched_extents.bin").string();
    CheckpointIO::writeBuffers<double>(path, s.nx, s.ny, s.nz,
        s.temperatureCurrent, s.vxCurrent, s.vyCurrent, s.vzCurrent, s.pressureCurrent,
        s.temperatureOld, s.vxOld, s.vyOld, s.vzOld, s.elapsedTime);

    std::vector<double> t1, t2, t3, t4, t5, t6, t7, t8, t9;
    EXPECT_THROW(CheckpointIO::readBuffers<double>(path, s.nx + 1, s.ny, s.nz, t1, t2, t3, t4, t5, t6, t7, t8, t9),
                 std::runtime_error);
}

TEST(CheckpointIOTest, ReadThrowsWhenHeaderMagicIsInvalid) {
    const auto dir = cfd_golden::uniqueRunDirectory("checkpoint_io");
    const std::string path = (dir / "bad_magic.bin").string();
    {
        std::ofstream out(path, std::ios::binary);
        const CheckpointHeader badHeader{0xDEADBEEFu, 4, 5, 3};
        out.write(reinterpret_cast<const char*>(&badHeader), sizeof(badHeader));
    }
    std::vector<double> t1, t2, t3, t4, t5, t6, t7, t8, t9;
    EXPECT_THROW(CheckpointIO::readBuffers<double>(path, 4, 5, 3, t1, t2, t3, t4, t5, t6, t7, t8, t9),
                 std::runtime_error);
}

TEST(CheckpointIOTest, ReadThrowsWhenFileIsTruncated) {
    const auto dir = cfd_golden::uniqueRunDirectory("checkpoint_io");
    const SyntheticCheckpoint s = MakeSynthetic(4, 5, 3);
    const std::string path = (dir / "truncated.bin").string();
    CheckpointIO::writeBuffers<double>(path, s.nx, s.ny, s.nz,
        s.temperatureCurrent, s.vxCurrent, s.vyCurrent, s.vzCurrent, s.pressureCurrent,
        s.temperatureOld, s.vxOld, s.vyOld, s.vzOld, s.elapsedTime);

    /* Chop the file down to just past the header -- well inside
       temperatureCurrent's 120 elements for this grid -- so readArray()
       must hit end-of-file partway through the first array and throw. */
    std::filesystem::resize_file(path, sizeof(CheckpointHeader) + 10);

    std::vector<double> t1, t2, t3, t4, t5, t6, t7, t8, t9;
    EXPECT_THROW(CheckpointIO::readBuffers<double>(path, s.nx, s.ny, s.nz, t1, t2, t3, t4, t5, t6, t7, t8, t9),
                 std::runtime_error);
}

TEST(CheckpointIOTest, ReadThrowsWhenFileDoesNotExist) {
    const auto dir = cfd_golden::uniqueRunDirectory("checkpoint_io");
    const std::string path = (dir / "does_not_exist.bin").string();
    std::vector<double> t1, t2, t3, t4, t5, t6, t7, t8, t9;
    EXPECT_THROW(CheckpointIO::readBuffers<double>(path, 4, 5, 3, t1, t2, t3, t4, t5, t6, t7, t8, t9),
                 std::runtime_error);
}

TEST(CheckpointIOTest, WriteThrowsWhenDirectoryDoesNotExist) {
    const auto dir = cfd_golden::uniqueRunDirectory("checkpoint_io");
    const std::string path = (dir / "missing_subdirectory" / "checkpoint.bin").string();
    const SyntheticCheckpoint s = MakeSynthetic(2, 2, 2);
    EXPECT_THROW(CheckpointIO::writeBuffers<double>(path, s.nx, s.ny, s.nz,
        s.temperatureCurrent, s.vxCurrent, s.vyCurrent, s.vzCurrent, s.pressureCurrent,
        s.temperatureOld, s.vxOld, s.vyOld, s.vzOld, s.elapsedTime),
        std::runtime_error);
}
