/**
 * @file CheckpointIO.h
 * @ingroup cfd_solver
 * @brief CheckpointIO: restart file read/write.
 *
 * Direct port of init_3D.f90's restart branch and outp_3D.f90's Outp.
 * Saves exactly what the original saved: Temperature, Vx, Vy, Vz (each
 * current and one-step-old), Pressure (current only, no old/new triad),
 * and the elapsed simulation time. Potential is not saved -- it never was
 * in the original either, since it is fully recomputed from the current
 * velocity field at the start of every timestep and is never an
 * independent degree of freedom across a restart.
 *
 * This is a new, self-describing binary format, NOT a reader for
 * Fortran's own .ddd files: the original format stores every ghost cell
 * (Fortran's `Dimension(0:Nx1,...)` arrays, written in full); this
 * project's fields store only true interior data, since there is no
 * padding left to write. A checkpoint written by this program can only be
 * read by this program.
 */

#ifndef CFDSOLVER_CHECKPOINTIO_H
#define CFDSOLVER_CHECKPOINTIO_H

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include "../fields/TimeDependentField.h"

constexpr uint32_t kCheckpointMagic = 0x43464443; ///< ASCII "CFDC".

/**
 * @ingroup cfd_solver
 * @brief A checkpoint file's fixed-size leading header: a magic number
 * plus the grid extents it was written for.
 */
struct CheckpointHeader {
    uint32_t magic;
    int32_t nx, ny, nz;
};

/// @brief Writes one array's raw bytes, with no header or length prefix
/// of its own -- readArray()'s `count` is the only thing that bounds it
/// back out again.
template<typename Real>
void writeArray(std::ofstream& out, const std::vector<Real>& data) {
    out.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)(data.size() * sizeof(Real)));
}

/// @brief Reads one array's raw bytes back, given the element count
/// writeArray() was called with for it.
template<typename Real>
std::vector<Real> readArray(std::ifstream& in, size_t count, const std::string& path, const char* label) {
    std::vector<Real> data(count);
    in.read(reinterpret_cast<char*>(data.data()), (std::streamsize)(count * sizeof(Real)));
    if (!in)
        throw std::runtime_error("CheckpointIO: failed reading '" + std::string(label) + "' from " + path + " -- file truncated or corrupt");
    return data;
}

/**
 * @ingroup cfd_solver
 * @brief Reads and writes the fields a simulation needs to resume from.
 * @tparam Real Floating-point type.
 */
class CheckpointIO {
public:
    /**
     * @brief Writes a checkpoint.
     * @tparam Real Floating-point type.
     * @param path Destination file path.
     * @param nx The simulation's configured Nx, stored in the header for
     * later validation by read().
     * @param ny The simulation's configured Ny, stored likewise.
     * @param nz The simulation's configured Nz, stored likewise.
     * @param temperature Temperature's current and old values.
     * @param vx Vx's current and old values.
     * @param vy Vy's current and old values.
     * @param vz Vz's current and old values.
     * @param pressure Current pressure.
     * @param elapsedTime Simulation time elapsed so far.
     * @param hand Handle to perform the device downloads on.
     */
    template<typename Real>
    static void write(const std::string& path, int nx, int ny, int nz,
                       const TimeDependentField<Real>& temperature, const TimeDependentField<Real>& vx,
                       const TimeDependentField<Real>& vy, const TimeDependentField<Real>& vz,
                       const GpuField<Real>& pressure, Real elapsedTime, Handle& hand);

    /**
     * @brief Reads a checkpoint, uploading its contents.
     * @tparam Real Floating-point type.
     * @param path Source file path.
     * @param expectedNx The simulation's configured Nx, checked against
     * the checkpoint's own header before any data is read.
     * @param expectedNy The simulation's configured Ny, checked likewise.
     * @param expectedNz The simulation's configured Nz, checked likewise.
     * @param temperature Temperature's current and old values, overwritten.
     * @param vx Vx's current and old values, overwritten.
     * @param vy Vy's current and old values, overwritten.
     * @param vz Vz's current and old values, overwritten.
     * @param pressure Current pressure, overwritten.
     * @param hand Handle to perform the device uploads on.
     * @return The elapsed simulation time stored in the checkpoint.
     * @throws std::runtime_error if the file is missing, malformed, or
     * its grid extents do not match expectedNx/expectedNy/expectedNz.
     */
    template<typename Real>
    static Real read(const std::string& path, int expectedNx, int expectedNy, int expectedNz,
                      TimeDependentField<Real>& temperature, TimeDependentField<Real>& vx,
                      TimeDependentField<Real>& vy, TimeDependentField<Real>& vz,
                      GpuField<Real>& pressure, Handle& hand);

    /**
     * @brief Writes a checkpoint from already-downloaded host buffers,
     * with no device interaction. write() downloads each field then calls
     * this; exposed separately, and defined here rather than in
     * CheckpointIO.cu, so test code can verify the on-disk round trip
     * directly without needing a GPU to reach it through write()/read() --
     * constructing a TimeDependentField or GpuField always allocates
     * device memory (see SolvedField's constructor), so there is no
     * host-only path through the field-level API.
     * @tparam Real Floating-point type.
     * @param path Destination file path.
     * @param nx The simulation's configured Nx, stored in the header for
     * later validation by readBuffers().
     * @param ny The simulation's configured Ny, stored likewise.
     * @param nz The simulation's configured Nz, stored likewise.
     * @param temperatureCurrent Temperature's current-timestep value.
     * @param vxCurrent Vx's current-timestep value.
     * @param vyCurrent Vy's current-timestep value.
     * @param vzCurrent Vz's current-timestep value.
     * @param pressureCurrent Current pressure -- same (Nx+1)(Ny+1)(Nz+1)
     * extent as temperatureCurrent (see readBuffers()).
     * @param temperatureOld Temperature's one-step-back value.
     * @param vxOld Vx's one-step-back value.
     * @param vyOld Vy's one-step-back value.
     * @param vzOld Vz's one-step-back value.
     * @param elapsedTime Simulation time elapsed so far.
     */
    template<typename Real>
    static void writeBuffers(const std::string& path, int nx, int ny, int nz,
                              const std::vector<Real>& temperatureCurrent, const std::vector<Real>& vxCurrent,
                              const std::vector<Real>& vyCurrent, const std::vector<Real>& vzCurrent,
                              const std::vector<Real>& pressureCurrent, const std::vector<Real>& temperatureOld,
                              const std::vector<Real>& vxOld, const std::vector<Real>& vyOld,
                              const std::vector<Real>& vzOld, Real elapsedTime) {
        std::ofstream out(path, std::ios::binary);
        if (!out)
            throw std::runtime_error("CheckpointIO: could not open " + path + " for writing");

        CheckpointHeader header{kCheckpointMagic, nx, ny, nz};
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        writeArray(out, temperatureCurrent);
        writeArray(out, vxCurrent);
        writeArray(out, vyCurrent);
        writeArray(out, vzCurrent);
        writeArray(out, pressureCurrent);

        writeArray(out, temperatureOld);
        writeArray(out, vxOld);
        writeArray(out, vyOld);
        writeArray(out, vzOld);

        out.write(reinterpret_cast<const char*>(&elapsedTime), sizeof(elapsedTime));

        if (!out)
            throw std::runtime_error("CheckpointIO: write to " + path + " failed partway through");
    }

    /**
     * @brief Reads a checkpoint into host buffers, with no device
     * interaction. read() calls this, then uploads each result; see
     * writeBuffers() for why this is exposed and defined here.
     * @tparam Real Floating-point type.
     * @param path Source file path.
     * @param expectedNx The simulation's configured Nx, checked against
     * the checkpoint's own header before any data is read.
     * @param expectedNy The simulation's configured Ny, checked likewise.
     * @param expectedNz The simulation's configured Nz, checked likewise.
     * @param temperatureCurrent Overwritten with Temperature's
     * current-timestep value.
     * @param vxCurrent Overwritten with Vx's current-timestep value.
     * @param vyCurrent Overwritten with Vy's current-timestep value.
     * @param vzCurrent Overwritten with Vz's current-timestep value.
     * @param pressureCurrent Overwritten with the current pressure; read
     * with the same (Nx+1)(Ny+1)(Nz+1) extent as temperatureCurrent, since
     * that is the extent write()/writeBuffers() actually downloaded/wrote
     * it with.
     * @param temperatureOld Overwritten with Temperature's one-step-back
     * value.
     * @param vxOld Overwritten with Vx's one-step-back value.
     * @param vyOld Overwritten with Vy's one-step-back value.
     * @param vzOld Overwritten with Vz's one-step-back value.
     * @return The elapsed simulation time stored in the checkpoint.
     * @throws std::runtime_error if the file is missing, malformed, or
     * its grid extents do not match expectedNx/expectedNy/expectedNz.
     */
    template<typename Real>
    static Real readBuffers(const std::string& path, int expectedNx, int expectedNy, int expectedNz,
                             std::vector<Real>& temperatureCurrent, std::vector<Real>& vxCurrent,
                             std::vector<Real>& vyCurrent, std::vector<Real>& vzCurrent,
                             std::vector<Real>& pressureCurrent, std::vector<Real>& temperatureOld,
                             std::vector<Real>& vxOld, std::vector<Real>& vyOld, std::vector<Real>& vzOld) {
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

        temperatureCurrent = readArray<Real>(in, temperatureSize, path, "Temperature (current)");
        vxCurrent = readArray<Real>(in, vxSize, path, "Vx (current)");
        vyCurrent = readArray<Real>(in, vySize, path, "Vy (current)");
        vzCurrent = readArray<Real>(in, vzSize, path, "Vz (current)");
        pressureCurrent = readArray<Real>(in, temperatureSize, path, "Pressure");

        temperatureOld = readArray<Real>(in, temperatureSize, path, "Temperature (old)");
        vxOld = readArray<Real>(in, vxSize, path, "Vx (old)");
        vyOld = readArray<Real>(in, vySize, path, "Vy (old)");
        vzOld = readArray<Real>(in, vzSize, path, "Vz (old)");

        Real elapsedTime = Real(0);
        in.read(reinterpret_cast<char*>(&elapsedTime), sizeof(elapsedTime));
        if (!in)
            throw std::runtime_error("CheckpointIO: failed reading elapsed time from " + path);

        return elapsedTime;
    }
};

#endif //CFDSOLVER_CHECKPOINTIO_H
