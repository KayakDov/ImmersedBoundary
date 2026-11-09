#!/usr/bin/env python3
"""
Parameter-sweep runner for the CFDSolver Fortran-vs-C++ parity check.

What this does, and does NOT do
--------------------------------
For each test case below, this script:
  1. writes a Conv.dat variant (for the Fortran binary) and an
     example_config.json variant (for the C++ binary), changing only the
     specific values that case varies;
  2. runs the REAL, unmodified Fortran binary and the REAL, unmodified C++
     binary, each exactly as you'd run them by hand, saving each run's full
     stdout/stderr to its own log file;
  3. appends the last 20 lines of each log (the final report block, same
     "tail -n 20" we already used by hand) to one running SUMMARY.txt,
     tagged with the case name, exit code, and wall time.

It does NOT parse numbers, apply tolerances, or print any pass/fail verdict
anywhere. All of that stays manual -- you and I read SUMMARY.txt together,
the same way we read the single baseline comparison. This script only
automates the mechanical part (editing configs, building the per-case
directories, running, collecting raw output) that you'd otherwise do by
hand ten times over.

Conv.dat line layout (1-indexed), confirmed directly from
ConvMain_3D_Q2D.f90's READ statements -- one value per line, in this exact
order, nothing skipped:
    1 AspRa   2 WidRa   3 Gr   4 Bi   5 Prandtl   6 Hartmann
    7 Nx      8 Ny      9 Nz   10 Eps  11 EpsCnv   12 ItMax
    13 Htime  14 Time   15 Iprint  16 Icheck  17 Iexcl  18 Istat
    19-21 EVD_BCx/y/z   22-24 EVD_Pot_X/Y/Z   25 Ncpus
    26 I_Fourier  27 N_Fourier  (28+ Omega(i) only if N_Fourier > 0)
This script only ever overwrites lines 3, 5, 6, 7, 8, 9 (Gr, Prandtl,
Hartmann, Nx, Ny, Nz). Every other line is copied byte-for-byte from your
real, already-verified Conv.dat, so nothing about boundary-condition codes,
convergence settings, Fourier state, etc. can drift.

Usage
-----
    python3 run_sweep.py

Expect roughly 15-20 minutes total, dominated by the one full-resolution
(100^3) Fortran run (~7 minutes, same as the run you already did by hand).
"""

import json
import os
import subprocess
import time

# ---- Paths: adjust these if anything differs on your machine --------------
HOME              = "/home/usr"
FORTRAN_DIR       = f"{HOME}/Documents/TelAvivU/Alex Code/CFDSolverModified"
FORTRAN_BIN       = f"{FORTRAN_DIR}/cmake-build-debug/CFDSolver"
FORTRAN_BASE_CONV = f"{FORTRAN_DIR}/Conv.dat"                       # your live, known-good Conv.dat -- read only, never modified
CPP_BIN           = f"{HOME}/Documents/TelAvivU/cmake-build-debug/cfdSolver"
CPP_BASE_JSON     = f"{HOME}/Documents/TelAvivU/src/cfdSolver/example_config.json"  # read only, never modified
SWEEP_ROOT        = f"{HOME}/Documents/TelAvivU/sweep_runs"         # everything this script writes lives under here
# -----------------------------------------------------------------------------

LINE = {"Gr": 3, "Prandtl": 5, "Hartmann": 6, "Nx": 7, "Ny": 8, "Nz": 9}

BASE_GRASHOF  = 1.85185185e7
BASE_PRANDTL  = 0.054
BASE_HARTMANN = 1000.0

# (case name, grid Nx=Ny=Nz, Gr override, Prandtl override, Hartmann override)
# None means "leave at the real baseline value from Conv.dat / example_config.json"
CASES = [
    # --- grid-resolution sweep, physics held at baseline ---
    ("grid_030",      30, None,         None,         None),
    ("grid_050",      50, None,         None,         None),
    ("grid_100",     100, None,         None,         None),  # re-run of the original baseline, for a clean apples-to-apples point generated the same way as the rest of this sweep
    # --- physical-parameter sweep, grid held at a faster 40^3 ---
    ("phys_baseline",  40, None,              None,              None),             # shared reference point for the three sweeps below
    ("gr_low",         40, BASE_GRASHOF / 2,  None,              None),
    ("gr_high",        40, BASE_GRASHOF * 2,  None,              None),
    ("pr_low",         40, None,              BASE_PRANDTL / 2,  None),
    ("pr_high",        40, None,              BASE_PRANDTL * 2,  None),
    ("ha_low",         40, None,              None,              BASE_HARTMANN / 2),
    ("ha_high",        40, None,              None,              BASE_HARTMANN * 2),
]


def write_conv_dat(path, grid, gr, pr, ha):
    with open(FORTRAN_BASE_CONV) as f:
        lines = f.readlines()

    def set_line(key, value):
        lines[LINE[key] - 1] = f"{value}\n"

    set_line("Nx", grid)
    set_line("Ny", grid)
    set_line("Nz", grid)
    if gr is not None:
        set_line("Gr", gr)
    if pr is not None:
        set_line("Prandtl", pr)
    if ha is not None:
        set_line("Hartmann", ha)

    with open(path, "w") as f:
        f.writelines(lines)


def write_json_config(path, grid, gr, pr, ha, out_dir):
    with open(CPP_BASE_JSON) as f:
        cfg = json.load(f)

    cfg["grid"]["Nx"] = cfg["grid"]["Ny"] = cfg["grid"]["Nz"] = grid
    if gr is not None:
        cfg["physics"]["grashof"] = gr
    if pr is not None:
        cfg["physics"]["prandtl"] = pr
    if ha is not None:
        cfg["physics"]["hartmann"] = ha
    cfg["output"]["directory"] = out_dir

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def run(cmd, cwd, log_path):
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode, time.time() - start


def append_tail(summary_path, label, log_path, n=20):
    with open(log_path) as f:
        lines = f.readlines()
    with open(summary_path, "a") as out:
        out.write(f"\n{'=' * 70}\n{label}\n{'=' * 70}\n")
        out.writelines(lines[-n:] if lines else ["(empty log)\n"])


def main():
    cpp_cfg_dir = os.path.join(SWEEP_ROOT, "cpp_configs")
    fortran_dir = os.path.join(SWEEP_ROOT, "fortran_runs")
    log_dir = os.path.join(SWEEP_ROOT, "logs")
    cpp_out_dir = os.path.join(SWEEP_ROOT, "cpp_outputs")
    for d in (cpp_cfg_dir, fortran_dir, log_dir, cpp_out_dir):
        os.makedirs(d, exist_ok=True)

    summary_path = os.path.join(SWEEP_ROOT, "SUMMARY.txt")
    open(summary_path, "w").close()  # start fresh each run

    overall_start = time.time()

    for name, grid, gr, pr, ha in CASES:
        print(f"\n--- {name}  (grid={grid}^3, "
              f"Gr={gr if gr is not None else BASE_GRASHOF:.6g}, "
              f"Pr={pr if pr is not None else BASE_PRANDTL:.6g}, "
              f"Ha={ha if ha is not None else BASE_HARTMANN:.6g}) ---")

        # ---- Fortran ----
        case_dir = os.path.join(fortran_dir, name)
        os.makedirs(case_dir, exist_ok=True)
        write_conv_dat(os.path.join(case_dir, "Conv.dat"), grid, gr, pr, ha)
        flog = os.path.join(log_dir, f"fortran_{name}.log")
        print("  Fortran: running...", flush=True)
        rc, secs = run([FORTRAN_BIN], case_dir, flog)
        print(f"  Fortran: exit={rc}  ({secs:.1f}s)")
        append_tail(summary_path, f"FORTRAN  {name}  exit={rc}  {secs:.1f}s", flog)

        # ---- C++ ----
        case_out = os.path.join(cpp_out_dir, name)
        cfg_path = os.path.join(cpp_cfg_dir, f"{name}.json")
        write_json_config(cfg_path, grid, gr, pr, ha, case_out)
        clog = os.path.join(log_dir, f"cpp_{name}.log")
        print("  C++: running...", flush=True)
        rc, secs = run([CPP_BIN, cfg_path], None, clog)
        print(f"  C++: exit={rc}  ({secs:.1f}s)")
        append_tail(summary_path, f"C++      {name}  exit={rc}  {secs:.1f}s", clog)

    print(f"\nAll {len(CASES)} cases done in {time.time() - overall_start:.0f}s total.")
    print(f"Combined final-report tails: {summary_path}")
    print(f"Full per-run logs:           {log_dir}")


if __name__ == "__main__":
    main()
