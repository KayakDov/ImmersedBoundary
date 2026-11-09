#!/usr/bin/env python3
"""
Chunk 1 of the C++ refactor: centralizes the copy-pasted CUDA thread-index
triple (blockIdx.*+blockDim.*+threadIdx.*+1, repeated identically 21 times)
into one threadGridIndex() helper in a new physics/ThreadIndex.cuh.

Run with: REPO=<path to repo root> python3 patch_thread_index.py

Two-pass: first reads and verifies every file (exact occurrence counts),
and only writes anything if every check across every file passes. Writes
nothing on any failure.
"""
import os
import sys

REPO = os.environ.get("REPO")
if not REPO:
    print("ERROR: set REPO=<path to repo root> first", file=sys.stderr)
    sys.exit(1)

PHYSICS_DIR = os.path.join(REPO, "src", "cfdSolver", "physics")

OLD_BLOCK = (
    "    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;\n"
    "    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;\n"
    "    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;"
)
NEW_BLOCK = (
    "    XYZ<int> idx = threadGridIndex();\n"
    "    int i = idx.x, j = idx.y, k = idx.z;"
)

FILES = {
    "Operators.cuh": ('#include "../fields/BoundaryConditions.cuh"', 9),
    "Diagnostics.cuh": ('#include "Operators.cuh"', 3),
    "ElectromagneticForce.cuh": ('#include "Operators.cuh"', 4),
    "ElementwiseOps.cuh": ('#include "../fields/BoundaryConditions.cuh"', 5),
}

THREAD_INDEX_CUH = '''/**
 * @file ThreadIndex.cuh
 * @ingroup cfd_solver
 * @brief threadGridIndex(): this thread's 1-based (i,j,k) grid index.
 *
 * Every full-volume kernel in this project launches over a 3D grid and
 * needs its thread's (i,j,k) position from CUDA's own
 * blockIdx/blockDim/threadIdx built-ins. Before this file existed, that
 * computation --
 *   int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
 *   int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
 *   int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
 * -- was copy-pasted identically into every such kernel (21 times, across
 * Operators.cuh, Diagnostics.cuh, ElectromagneticForce.cuh, and
 * ElementwiseOps.cuh). It carries no per-axis physical meaning of its own
 * -- unlike Spacing's hx12/hy12/hz12, which this project deliberately
 * keeps as three separate, Fortran-matching pointers (see Operators.cuh's
 * header comment) -- so there is no cross-checking benefit to leaving it
 * unrolled, and every copy was a fresh chance for a pasted-but-not-updated
 * axis: the same shape of mistake as ConvMain_3D_Q2D.f90's
 * `VMz = VMx_Az` typo. Centralized here once instead.
 *
 * Deliberately NOT used by nusseltKernel/nusseltMiddleKernel in
 * Diagnostics.cuh: those two launch over a degenerate (1-wide) axis and
 * rely on reading threadIdx.x/.y/.z directly to guard against the
 * 8x/64x-inflation bug documented on those kernels -- see Diagnostics.cuh.
 */

#ifndef CFDSOLVER_THREADINDEX_CUH
#define CFDSOLVER_THREADINDEX_CUH

#ifdef __CUDACC__

#include "math/XYZ.cuh"

/**
 * @ingroup cfd_solver
 * @brief This thread's 1-based (i,j,k) grid index: .x/.y/.z hold the same
 * i/j/k every full-volume kernel in this project indexes its
 * FieldView/Spacing arguments with.
 * @return The thread's 1-based index, one component per axis.
 */
__device__ __forceinline__ XYZ<int> threadGridIndex() {
    return XYZ<int>(
        (int)(blockIdx.x * blockDim.x + threadIdx.x) + 1,
        (int)(blockIdx.y * blockDim.y + threadIdx.y) + 1,
        (int)(blockIdx.z * blockDim.z + threadIdx.z) + 1
    );
}

#endif //__CUDACC__

#endif //CFDSOLVER_THREADINDEX_CUH
'''

def main():
    new_file_path = os.path.join(PHYSICS_DIR, "ThreadIndex.cuh")
    if os.path.exists(new_file_path):
        print(f"ERROR: {new_file_path} already exists -- already patched?", file=sys.stderr)
        sys.exit(1)

    planned = {}
    for name, (anchor, expected_count) in FILES.items():
        path = os.path.join(PHYSICS_DIR, name)
        if not os.path.isfile(path):
            print(f"ERROR: not found: {path}", file=sys.stderr)
            sys.exit(1)

        with open(path, "r") as f:
            text = f.read()

        actual_count = text.count(OLD_BLOCK)
        if actual_count != expected_count:
            print(f"ERROR: {name}: expected {expected_count} occurrences of the old "
                  f"index block, found {actual_count}. Aborting before writing anything.",
                  file=sys.stderr)
            sys.exit(1)
        if anchor not in text:
            print(f"ERROR: {name}: anchor include not found: {anchor!r}", file=sys.stderr)
            sys.exit(1)
        if text.count(anchor) != 1:
            print(f"ERROR: {name}: anchor include appears {text.count(anchor)} times, "
                  f"expected exactly 1", file=sys.stderr)
            sys.exit(1)
        if '#include "ThreadIndex.cuh"' in text:
            print(f"ERROR: {name}: already includes ThreadIndex.cuh -- already patched?",
                  file=sys.stderr)
            sys.exit(1)

        new_text = text.replace(anchor, anchor + '\n#include "ThreadIndex.cuh"', 1)
        new_text = new_text.replace(OLD_BLOCK, NEW_BLOCK)

        if new_text.count(OLD_BLOCK) != 0:
            print(f"ERROR: {name}: old block still present after replacement", file=sys.stderr)
            sys.exit(1)
        if new_text.count("threadGridIndex()") != expected_count:
            print(f"ERROR: {name}: expected {expected_count} threadGridIndex() calls after "
                  f"patch, found {new_text.count('threadGridIndex()')}", file=sys.stderr)
            sys.exit(1)
        if new_text.count("{") != text.count("{") or new_text.count("}") != text.count("}"):
            print(f"ERROR: {name}: brace count changed by the patch", file=sys.stderr)
            sys.exit(1)

        planned[path] = new_text

    with open(new_file_path, "w") as f:
        f.write(THREAD_INDEX_CUH)
    print(f"OK: wrote {new_file_path}")

    for path, new_text in planned.items():
        with open(path, "w") as f:
            f.write(new_text)
        print(f"OK: patched {path}")

    print("All files patched successfully. 21 kernel index sites converted across 4 files.")

if __name__ == "__main__":
    main()
