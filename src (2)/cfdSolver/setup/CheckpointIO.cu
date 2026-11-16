/**
 * @file CheckpointIO.cu
 * @ingroup cfd_solver
 * @brief Implements CheckpointIO::write() and CheckpointIO::read().
 *
 * Compile-verified with nvcc (zero warnings); not run, since no physical
 * GPU is available in the environment this was checked in. This file's
 * actual device interaction is limited to the GpuField/TimeDependentField
 * download/upload calls it makes, both implemented and documented
 * elsewhere; the on-disk format itself (writeBuffers()/readBuffers(),
 * declared and defined in CheckpointIO.h so test_checkpoint_io.cpp can
 * round-trip it without a GPU) is ordinary C++ checked by hand against
 * init_3D.f90/outp_3D.f90's read/write order, and now by that direct
 * test too.
 */

#include "CheckpointIO.h"

template<typename Real>
void CheckpointIO::write(const std::string& path, int nx, int ny, int nz,
                          const TimeDependentField<Real>& temperature, const TimeDependentField<Real>& vx,
                          const TimeDependentField<Real>& vy, const TimeDependentField<Real>& vz,
                          const GpuField<Real>& pressure, Real elapsedTime, Handle& hand) {
    writeBuffers<Real>(path, nx, ny, nz,
        temperature.downloadCurrent(hand), vx.downloadCurrent(hand), vy.downloadCurrent(hand), vz.downloadCurrent(hand),
        pressure.downloadTo(hand),
        temperature.downloadOld(hand), vx.downloadOld(hand), vy.downloadOld(hand), vz.downloadOld(hand),
        elapsedTime);
}

template<typename Real>
Real CheckpointIO::read(const std::string& path, int expectedNx, int expectedNy, int expectedNz,
                         TimeDependentField<Real>& temperature, TimeDependentField<Real>& vx,
                         TimeDependentField<Real>& vy, TimeDependentField<Real>& vz,
                         GpuField<Real>& pressure, Handle& hand) {
    std::vector<Real> temperatureCurrent, vxCurrent, vyCurrent, vzCurrent, pressureCurrent;
    std::vector<Real> temperatureOld, vxOld, vyOld, vzOld;
    Real elapsedTime = readBuffers<Real>(path, expectedNx, expectedNy, expectedNz,
        temperatureCurrent, vxCurrent, vyCurrent, vzCurrent, pressureCurrent,
        temperatureOld, vxOld, vyOld, vzOld);

    temperature.uploadCurrent(temperatureCurrent, hand);
    vx.uploadCurrent(vxCurrent, hand);
    vy.uploadCurrent(vyCurrent, hand);
    vz.uploadCurrent(vzCurrent, hand);
    pressure.uploadFrom(pressureCurrent, hand);

    temperature.uploadOld(temperatureOld, hand);
    vx.uploadOld(vxOld, hand);
    vy.uploadOld(vyOld, hand);
    vz.uploadOld(vzOld, hand);

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
