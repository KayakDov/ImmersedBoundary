#!/usr/bin/env python3
"""
Chunk 4: convergence residuals (rnsX/rnsY/rnsZ) -> XYZ<Real>.

Patches CfdSimulation.h (3 declaration edits) and CfdSimulation.cu
(10 edits: residuals() signature+body, runOneTimeStep() signature+call,
the run()-loop call site, the printIterationReport() call, the
worstResidual line, and printIterationReport()'s own signature+2 body
lines).

Two-pass: every old_string is first checked to occur EXACTLY ONCE in its
file. Nothing is written until all checks for both files pass. Re-running
on an already-patched tree is refused (old_strings won't be found, count
will be 0, not 1) rather than corrupting anything.

Usage:
    python3 patch_cfdsimulation_residual_xyz.py /path/to/src
(where /path/to/src/cfdSolver/CfdSimulation.h and .cu exist)
"""
import sys
from pathlib import Path

H_EDITS = [
    (
        "    void residuals(Real& rnsX, Real& rnsY, Real& rnsZ, Real& rdp);\n"
        "    void shiftAllTimeLevels();\n"
        "    void runOneTimeStep(Real& rnsX, Real& rnsY, Real& rnsZ, Real& rTmpr, Real& rdp);\n",
        "    void residuals(XYZ<Real>& residual, Real& rdp);\n"
        "    void shiftAllTimeLevels();\n"
        "    void runOneTimeStep(XYZ<Real>& residual, Real& rTmpr, Real& rdp);\n",
    ),
    (
        "    void printIterationReport(Real timeNow, Real targetTime, Real rnsX, Real rnsY, Real rnsZ, Real rTmpr, Real rdp) const;\n",
        "    void printIterationReport(Real timeNow, Real targetTime, XYZ<Real> residual, Real rTmpr, Real rdp) const;\n",
    ),
]

CU_EDITS = [
    (
        "void CfdSimulation<Real>::residuals(Real& rnsX, Real& rnsY, Real& rnsZ, Real& rdp) {\n",
        "void CfdSimulation<Real>::residuals(XYZ<Real>& residual, Real& rdp) {\n",
    ),
    (
        "    rnsX = maxRelativeDifference(vx_.solved().view(), vx_.currentView(), vx_.solved().handle());\n"
        "    rnsY = maxRelativeDifference(vy_.solved().view(), vy_.currentView(), vy_.solved().handle());\n"
        "    rnsZ = maxRelativeDifference(vz_.solved().view(), vz_.currentView(), vz_.solved().handle());\n",
        "    residual.x = maxRelativeDifference(vx_.solved().view(), vx_.currentView(), vx_.solved().handle());\n"
        "    residual.y = maxRelativeDifference(vy_.solved().view(), vy_.currentView(), vy_.solved().handle());\n"
        "    residual.z = maxRelativeDifference(vz_.solved().view(), vz_.currentView(), vz_.solved().handle());\n",
    ),
    (
        "void CfdSimulation<Real>::runOneTimeStep(Real& rnsX, Real& rnsY, Real& rnsZ, Real& rTmpr, Real& rdp) {\n",
        "void CfdSimulation<Real>::runOneTimeStep(XYZ<Real>& residual, Real& rTmpr, Real& rdp) {\n",
    ),
    (
        "    residuals(rnsX, rnsY, rnsZ, rdp);\n",
        "    residuals(residual, rdp);\n",
    ),
    (
        "        Real rnsX, rnsY, rnsZ, rTmpr, rdp;\n"
        "        runOneTimeStep(rnsX, rnsY, rnsZ, rTmpr, rdp);\n",
        "        XYZ<Real> residual((Real)0, (Real)0, (Real)0);\n"
        "        Real rTmpr, rdp;\n"
        "        runOneTimeStep(residual, rTmpr, rdp);\n",
    ),
    (
        "            printIterationReport(elapsedTime_, targetTime, rnsX, rnsY, rnsZ, rTmpr, rdp);\n",
        "            printIterationReport(elapsedTime_, targetTime, residual, rTmpr, rdp);\n",
    ),
    (
        "        Real worstResidual = std::max(std::max(rnsX, rnsY), std::max(rnsZ, rTmpr));\n",
        "        Real worstResidual = std::max(std::max(residual.x, residual.y), std::max(residual.z, rTmpr));\n",
    ),
    (
        "void CfdSimulation<Real>::printIterationReport(Real timeNow, Real targetTime, Real rnsX, Real rnsY, Real rnsZ, Real rTmpr, Real rdp) const {\n",
        "void CfdSimulation<Real>::printIterationReport(Real timeNow, Real targetTime, XYZ<Real> residual, Real rTmpr, Real rdp) const {\n",
    ),
    (
        '    std::printf("  RNSx=%11.4g  RNSy=%11.4g  RNSz=%11.4g  RTmpr=%11.4g  RDP=%11.4g\\n",\n'
        "        (double)rnsX, (double)rnsY, (double)rnsZ, (double)rTmpr, (double)rdp);\n",
        '    std::printf("  RNSx=%11.4g  RNSy=%11.4g  RNSz=%11.4g  RTmpr=%11.4g  RDP=%11.4g\\n",\n'
        "        (double)residual.x, (double)residual.y, (double)residual.z, (double)rTmpr, (double)rdp);\n",
    ),
    (
        '        std::fprintf(convOutFile_.get(), "  RNSx=%11.4g  RNSy=%11.4g  RNSz=%11.4g  RTmpr=%11.4g  RDP=%11.4g\\n",\n'
        "            (double)rnsX, (double)rnsY, (double)rnsZ, (double)rTmpr, (double)rdp);\n",
        '        std::fprintf(convOutFile_.get(), "  RNSx=%11.4g  RNSy=%11.4g  RNSz=%11.4g  RTmpr=%11.4g  RDP=%11.4g\\n",\n'
        "            (double)residual.x, (double)residual.y, (double)residual.z, (double)rTmpr, (double)rdp);\n",
    ),
]


def apply_edits(path: Path, edits):
    text = path.read_text()
    for i, (old, new) in enumerate(edits):
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"REFUSING to patch {path}: edit #{i} expected exactly 1 "
                f"occurrence, found {count}.\n--- old_string ---\n{old}"
            )
    for old, new in edits:
        text = text.replace(old, new, 1)
    return text


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} /path/to/src")
    src_root = Path(sys.argv[1])
    h_path = src_root / "cfdSolver" / "CfdSimulation.h"
    cu_path = src_root / "cfdSolver" / "CfdSimulation.cu"
    for p in (h_path, cu_path):
        if not p.is_file():
            raise SystemExit(f"not found: {p}")

    # Pass 1: verify everything, write nothing yet.
    new_h = apply_edits(h_path, H_EDITS)
    new_cu = apply_edits(cu_path, CU_EDITS)

    # Pass 2: all checks passed for both files -- now write.
    h_path.write_text(new_h)
    cu_path.write_text(new_cu)
    print(f"Patched {h_path} ({len(H_EDITS)} edits)")
    print(f"Patched {cu_path} ({len(CU_EDITS)} edits)")


if __name__ == "__main__":
    main()
