#!/usr/bin/env python3
"""
Patches CfdSimulation.cu's launchPotential() for the Hartmann=1000 fix.

Why a patch script instead of a complete drop-in file: unlike
GpuSolverSetup.cu and ElectromagneticForce.cuh, the full text of
CfdSimulation.cu was never read in this investigation -- only small
excerpts -- so reconstructing it whole would risk silently dropping
unrelated code. This script only touches the one line that needs to
change, and only if it finds the exact text it expects exactly once.

Revision note: the first version of this script matched only on
"LaunchConfig3D cfg3d(rhs.nx(), rhs.ny(), rhs.nz());" -- that turned out
to be a generic pattern reused by every field's launch function (found 4
times, not 1), so its safety guard correctly refused to touch the file.
This version anchors the match to the line that follows it in
launchPotential() specifically -- the launch of potentialRightHandSideKernel,
which only this one function calls -- so it can no longer match any other
field's launch code.

What's changing and why: GpuSolverSetup.cu now builds Potential's
solver/buffer with roles permuted (nx()<-physical Z, ny()<-physical X,
nz()<-physical Y) to match Fortran's init_eigen_decomp_d forwarding rule.
potentialRightHandSideKernel (in ElectromagneticForce.cuh) was updated to
match: it still sweeps physical x,y,z via blockIdx.x/y/z, but now bounds
that sweep against (rhs.ny(), rhs.nz(), rhs.nx()) -- the slots physical
x,y,z now actually occupy -- instead of (rhs.nx(), rhs.ny(), rhs.nz()).
launchPotential()'s LaunchConfig3D must request threads using that same
(rhs.ny(), rhs.nz(), rhs.nx()) ordering, or it will size the launch grid
using Potential's permuted extents in the wrong slots -- under-covering
one physical axis (dropping real cells) and over-covering another
(wasted, immediately-returning threads) rather than matching what the
kernel's own bounds check now expects.

Usage: python3 patch_cfdsimulation.py [path-to-CfdSimulation.cu]
Defaults to src/cfdSolver/CfdSimulation.cu if no path is given.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/cfdSolver/CfdSimulation.cu"

with open(path, "r") as f:
    text = f.read()

old = (
    "    LaunchConfig3D cfg3d(rhs.nx(), rhs.ny(), rhs.nz());\n"
    "    potentialRightHandSideKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>("
)
new = (
    "    // FIX (Hartmann=1000 investigation): Potential's buffer is now role-\n"
    "    // permuted (nx()<-physical Z, ny()<-physical X, nz()<-physical Y) --\n"
    "    // see GpuSolverSetup.cu and ElectromagneticForce.cuh. Request threads\n"
    "    // in that same (ny(),nz(),nx()) order so the launch grid matches the\n"
    "    // kernel's own (now-permuted) bounds check exactly.\n"
    "    LaunchConfig3D cfg3d(rhs.ny(), rhs.nz(), rhs.nx());\n"
    "    potentialRightHandSideKernel<Real><<<cfg3d.blocks, cfg3d.threads, 0, hand>>>("
)

count = text.count(old)
if count == 1:
    with open(path, "w") as f:
        f.write(text.replace(old, new))
    print(f"{path}: launchPotential() patched OK (1 occurrence replaced).")
elif count == 0:
    print(f"{path}: expected text not found. NOT modified. "
          f"The line may have already been patched, or the file doesn't "
          f"match what this investigation has seen -- check manually.")
    sys.exit(1)
else:
    print(f"{path}: expected exactly 1 match, found {count}. NOT modified -- "
          f"ambiguous, needs a manual look.")
    sys.exit(1)
