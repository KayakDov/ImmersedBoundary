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

#include <string>
#include "../fields/TimeDependentField.h"

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
};

#endif //CFDSOLVER_CHECKPOINTIO_H
