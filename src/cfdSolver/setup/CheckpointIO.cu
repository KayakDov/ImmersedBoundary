/**
 * @file CheckpointIO.cu
 * @ingroup cfd_solver
 * @brief Implements CheckpointIO::write() and CheckpointIO::read().
 *
 * @warning Not compile-verified: see GpuSolverSetup.cu's file-level
 * warning for why. This file's actual device interaction is limited to
 * the GpuField::downloadTo()/uploadFrom() calls it makes, both already
 * implemented and documented elsewhere; the file I/O itself is ordinary
 * C++ and was additionally checked by hand against init_3D.f90/
 * outp_3D.f90's read/write order.
 */

#include "CheckpointIO.h"
#include <fstream>
#include <stdexcept>

namespace {

constexpr uint32_t kCheckpointMagic = 0x43464443; // "CFDC"

struct CheckpointHeader {
    uint32_t magic;
    int32_t nx, ny, nz;
};

template<typename Real>
void writeArray(std::ofstream& out, const std::vector<Real>& data) {
    out.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)(data.size() * sizeof(Real)));
}

template<typename Real>
std::vector<Real> readArray(std::ifstream& in, size_t count, const std::string& path, const char* label) {
    std::vector<Real> data(count);
    in.read(reinterpret_cast<char*>(data.data()), (std::streamsize)(count * sizeof(Real)));
    if (!in)
        throw std::runtime_error("CheckpointIO: failed reading '" + std::string(label) + "' from " + path + " -- file truncated or corrupt");
    return data;
}

} // namespace

template<typename Real>
void CheckpointIO::write(const std::string& path, int nx, int ny, int nz,
                          const TimeDependentField<Real>& temperature, const TimeDependentField<Real>& vx,
                          const TimeDependentField<Real>& vy, const TimeDependentField<Real>& vz,
                          const GpuField<Real>& pressure, Real elapsedTime, Handle& hand) {
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("CheckpointIO: could not open " + path + " for writing");

    CheckpointHeader header{kCheckpointMagic, nx, ny, nz};
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    writeArray(out, temperature.downloadCurrent(hand));
    writeArray(out, vx.downloadCurrent(hand));
    writeArray(out, vy.downloadCurrent(hand));
    writeArray(out, vz.downloadCurrent(hand));
    writeArray(out, pressure.downloadTo(hand));

    writeArray(out, temperature.downloadOld(hand));
    writeArray(out, vx.downloadOld(hand));
    writeArray(out, vy.downloadOld(hand));
    writeArray(out, vz.downloadOld(hand));

    out.write(reinterpret_cast<const char*>(&elapsedTime), sizeof(elapsedTime));

    if (!out)
        throw std::runtime_error("CheckpointIO: write to " + path + " failed partway through");
}

template<typename Real>
Real CheckpointIO::read(const std::string& path, int expectedNx, int expectedNy, int expectedNz,
                         TimeDependentField<Real>& temperature, TimeDependentField<Real>& vx,
                         TimeDependentField<Real>& vy, TimeDependentField<Real>& vz,
                         GpuField<Real>& pressure, Handle& hand) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("CheckpointIO: could not open " + path + " for reading");

    CheckpointHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || header.magic != kCheckpointMagic)
        throw std::runtime_error("CheckpointIO: " + path + " is not a valid checkpoint file");
    if (header.nx != expectedNx || header.ny != expectedNy || header.nz != expectedNz)
        throw std::runtime_error("CheckpointIO: " + path + " was written for a " +
            std::to_string(header.nx) + "x" + std::to_string(header.ny) + "x" + std::to_string(header.nz) +
            " grid, but this simulation is configured for " +
            std::to_string(expectedNx) + "x" + std::to_string(expectedNy) + "x" + std::to_string(expectedNz));

    int nx1 = expectedNx + 1, ny1 = expectedNy + 1, nz1 = expectedNz + 1;
    size_t temperatureSize = (size_t)nx1 * ny1 * nz1;
    size_t vxSize = (size_t)expectedNx * ny1 * nz1;
    size_t vySize = (size_t)nx1 * expectedNy * nz1;
    size_t vzSize = (size_t)nx1 * ny1 * expectedNz;

    temperature.uploadCurrent(readArray<Real>(in, temperatureSize, path, "Temperature (current)"), hand);
    vx.uploadCurrent(readArray<Real>(in, vxSize, path, "Vx (current)"), hand);
    vy.uploadCurrent(readArray<Real>(in, vySize, path, "Vy (current)"), hand);
    vz.uploadCurrent(readArray<Real>(in, vzSize, path, "Vz (current)"), hand);
    pressure.uploadFrom(readArray<Real>(in, temperatureSize, path, "Pressure"), hand);

    temperature.uploadOld(readArray<Real>(in, temperatureSize, path, "Temperature (old)"), hand);
    vx.uploadOld(readArray<Real>(in, vxSize, path, "Vx (old)"), hand);
    vy.uploadOld(readArray<Real>(in, vySize, path, "Vy (old)"), hand);
    vz.uploadOld(readArray<Real>(in, vzSize, path, "Vz (old)"), hand);

    Real elapsedTime = Real(0);
    in.read(reinterpret_cast<char*>(&elapsedTime), sizeof(elapsedTime));
    if (!in)
        throw std::runtime_error("CheckpointIO: failed reading elapsed time from " + path);

    return elapsedTime;
}

template void CheckpointIO::write<double>(const std::string&, int, int, int, const TimeDependentField<double>&,
    const TimeDependentField<double>&, const TimeDependentField<double>&, const TimeDependentField<double>&,
    const GpuField<double>&, double, Handle&);
template double CheckpointIO::read<double>(const std::string&, int, int, int, TimeDependentField<double>&,
    TimeDependentField<double>&, TimeDependentField<double>&, TimeDependentField<double>&, GpuField<double>&, Handle&);

template void CheckpointIO::write<float>(const std::string&, int, int, int, const TimeDependentField<float>&,
    const TimeDependentField<float>&, const TimeDependentField<float>&, const TimeDependentField<float>&,
    const GpuField<float>&, float, Handle&);
template float CheckpointIO::read<float>(const std::string&, int, int, int, TimeDependentField<float>&,
    TimeDependentField<float>&, TimeDependentField<float>&, TimeDependentField<float>&, GpuField<float>&, Handle&);
