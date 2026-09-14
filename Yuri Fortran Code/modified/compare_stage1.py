#!/usr/bin/env python3
"""
Stage 1 comparison: diffs the Dprs (pressure correction) and F_tag (force
correction) dumps produced by orig/ and modified/ after the very first
pressure-force solve of time step 1, then both STOP.

Usage:
    python3 compare_stage1.py <orig_run_dir> <modified_run_dir> Nx1 Ny1 Nz1 <TotalUnknownsP>

Nx1, Ny1, Nz1, and TotalUnknownsP must match the case's actual grid/body
sizes (read them out of your input file) -- they're needed to know how many
doubles to expect in each file; the dumps themselves are just a flat stream
of IEEE-754 doubles with no header.
"""
import sys
import numpy as np

def load(path, n_expected):
    arr = np.fromfile(path, dtype=np.float64)
    if arr.size != n_expected:
        raise ValueError(f"{path}: expected {n_expected} doubles, got {arr.size} "
                          f"-- check Nx1/Ny1/Nz1/TotalUnknownsP match this case")
    return arr

def report(name, a, b):
    diff = a - b
    absdiff = np.abs(diff)
    denom = np.maximum(np.abs(a), 1e-300)
    reldiff = absdiff / denom
    print(f"--- {name} ---")
    print(f"  max |a-b|          = {absdiff.max():.6e}")
    print(f"  max |a-b|/|a|      = {reldiff.max():.6e}  (near meaningless where a~0)")
    print(f"  mean |a-b|         = {absdiff.mean():.6e}")
    worst = np.argmax(absdiff)
    print(f"  worst element: index {worst}, orig={a[worst]:.10e}, modified={b[worst]:.10e}")
    n_nan_a, n_nan_b = np.isnan(a).sum(), np.isnan(b).sum()
    n_inf_a, n_inf_b = np.isinf(a).sum(), np.isinf(b).sum()
    if n_nan_a or n_nan_b or n_inf_a or n_inf_b:
        print(f"  !! NaN: orig={n_nan_a} modified={n_nan_b}   Inf: orig={n_inf_a} modified={n_inf_b}")
    print()

def main():
    if len(sys.argv) != 7:
        print(__doc__)
        sys.exit(1)
    orig_dir, mod_dir, Nx1, Ny1, Nz1, total_unknowns_p = sys.argv[1:]
    Nx1, Ny1, Nz1, total_unknowns_p = int(Nx1), int(Ny1), int(Nz1), int(total_unknowns_p)

    n_p = Nx1 * Ny1 * Nz1
    n_f = 3 * total_unknowns_p

    dprs_orig = load(f"{orig_dir}/dprs_dump.bin", n_p)
    dprs_mod  = load(f"{mod_dir}/dprs_dump.bin",  n_p)
    ftag_orig = load(f"{orig_dir}/ftag_dump.bin", n_f)
    ftag_mod  = load(f"{mod_dir}/ftag_dump.bin",  n_f)

    report("Dprs (pressure correction)", dprs_orig, dprs_mod)
    report("F_tag (force correction)",   ftag_orig, ftag_mod)

    print("Rule of thumb: max |a-b| should be within a couple orders of "
          "magnitude of the solver's own tolerance (Eps in the input file) "
          "-- much larger than that means a real discrepancy, not just "
          "floating-point/iteration-order noise between the two solvers.")

if __name__ == "__main__":
    main()
