#!/usr/bin/env python3
"""
Adds __host__ __device__ to XYZ<T>'s constructor in src/math/XYZ.cuh.

Purely additive annotation: XYZ<T>'s operator[], fill(), and toArray() are
already __host__ __device__; only the constructor itself was left
unannotated (defaults to __host__-only under nvcc), which is why
threadGridIndex() (physics/ThreadIndex.cuh, cfdSolver refactor chunk 1)
fails to compile -- it's a pure __device__ function trying to call a
__host__-only constructor. Every existing call site constructs XYZ<T> on
the host (GpuSolverSetup.cu) and is unaffected; this only *adds* a context
the constructor is callable from.

Run with: REPO=<path to repo root> python3 patch_xyz_ctor.py
Verifies the exact line appears exactly once before touching anything.
"""
import os
import sys

REPO = os.environ.get("REPO")
if not REPO:
    print("ERROR: set REPO=<path to repo root> first", file=sys.stderr)
    sys.exit(1)

path = os.path.join(REPO, "src", "math", "XYZ.cuh")

OLD_LINE = "    XYZ(const T& x, const T& y, const T& z): x(x), y(y), z(z) {}"
NEW_LINE = "    __host__ __device__ XYZ(const T& x, const T& y, const T& z): x(x), y(y), z(z) {}"

if not os.path.isfile(path):
    print(f"ERROR: not found: {path}", file=sys.stderr)
    sys.exit(1)

with open(path, "r") as f:
    text = f.read()

count = text.count(OLD_LINE)
if count != 1:
    print(f"ERROR: expected exactly 1 occurrence of the constructor line, found {count}. "
          f"Not touching the file.", file=sys.stderr)
    sys.exit(1)
if NEW_LINE in text:
    print("ERROR: annotated form already present -- already patched?", file=sys.stderr)
    sys.exit(1)

before_braces = (text.count("{"), text.count("}"), text.count("("), text.count(")"))
new_text = text.replace(OLD_LINE, NEW_LINE, 1)
after_braces = (new_text.count("{"), new_text.count("}"), new_text.count("("), new_text.count(")"))

if new_text.count(NEW_LINE) != 1:
    print("ERROR: new line not present exactly once after replacement", file=sys.stderr)
    sys.exit(1)
if new_text.count(OLD_LINE) != 0:
    print("ERROR: old (unannotated) line still present after replacement", file=sys.stderr)
    sys.exit(1)
if before_braces != after_braces:
    print(f"ERROR: brace/paren counts changed unexpectedly: before={before_braces} "
          f"after={after_braces}", file=sys.stderr)
    sys.exit(1)

with open(path, "w") as f:
    f.write(new_text)
print(f"OK: patched {path}")
print("XYZ<T>'s constructor is now __host__ __device__.")
