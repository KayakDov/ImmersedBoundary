# CHANGES.md -- GPU migration of the eigendecomposition solves

## What changed, in one sentence

All six of the implicit linear solves per time step -- temperature, the
three velocity components, pressure, and the electric potential -- were
moved from CPU eigendecomposition (`EVDLapTmpr`, `EVDLapVx/y/z`, `EVDLapP`,
`EVD_Fi`, `EVD_Thomas`/`EVDmethod`) to Dov's CudaBandedLib GPU
eigendecomposition solver, via a new module, `AlexCudaCompatibility`.
**No array in the codebase was reindexed or restructured.** 14 of the 18
files in `src/` are byte-identical to the original. Only 5 files differ,
and every difference in them is one of: a `Use` statement bringing the new
module into scope, a CPU solver call replaced by its GPU equivalent, or a
small scratch buffer holding a scaled copy of a right-hand side (never a
reindexed one). See "Files touched" below for the exact, complete list of
differences.

## Why no reindexing was needed

CudaBandedLib's solver is isotropic: it has no concept of "X", "Y", or "Z",
only `dim1`/`dim2`/`dim3`, defined purely by which one changes fastest in
the flattened array -- no physical axis is attached to any of them. Alex's
arrays already declare X first (fastest, since Fortran is column-major), so
`dim1 = X, dim2 = Y, dim3 = Z` throughout this bridge, matching the
original array declarations exactly. No array anywhere in the codebase
needed to change shape, declaration order, or loop index order to make this
integration possible.

(An earlier iteration of this integration, made against an older,
non-isotropic version of the library that required a fixed physical
flattening order, did reindex every array in the codebase to `(Y,Z,X)`.
That reindexing is not present in, and was not needed for, this version.)

## Why six solvers, two different equation types

Alex's 3-level implicit time integrator turns the temperature and velocity
updates into HELMHOLTZ solves, `(L - shift*I) x = b`, where the shift comes
from the `-(Ckor/Htime)` term of the time discretization (`Ckor = 1.5`, the
3-level BDF coefficient). Pressure and potential have no time derivative in
their equations and remain pure POISSON solves (`shift = 0`). The GPU
operator is built un-scaled; the equivalent scaling the CPU solvers folded
into their own `T_left/T_center/...`, `Vx_left/...` coefficient arrays is
applied here as a plain elementwise scale of the right-hand side (`* GrPr`
for temperature, `/ DGr` for the velocities) immediately before each GPU
solve call, in `time_step_Q2D.f90`. See the per-solver comments in
`AlexCudaCompatibility.f90` for the exact derivation.

## Why two different axis discretizations (FLUX_LAPL / VARIABLE_DELTA_LAPL)

Alex's mesh is staggered per field, not just per axis: each velocity
component is node-centred along its own direction (the faces of its control
volume) and cell-centred along the other two. Pressure and temperature are
cell-centred on every axis. CudaBandedLib exposes this as a per-axis
`eigen::LaplOperatorT` choice, matched here exactly to which deltas each of
Alex's `EVDLap*`/`EVD_Fi` routines uses for that axis (verified directly
against their source, not inferred):

- `FLUX_LAPL` -- conservative finite-volume Laplacian for cell-centred axes
  (deltas `HPx/HPy/HPz`, widths reconstructed from `Hx12/Hy12/Hz12`);
  reproduces Alex's exact finite-volume operator, which is REQUIRED for
  pressure: the projection step needs this operator to equal `div(grad)`
  discretely, or residual divergence accumulates and the simulation
  diverges.
- `VARIABLE_DELTA_LAPL` -- pointwise 3-point Laplacian for node-centred axes
  (deltas `Hx12/Hy12/Hz12`); matches Alex's node-centred operators exactly
  on any mesh.

Per-solver axis assignment (all confirmed against the exact stencil formulas
in the corresponding `EVDLap*`/`EVD_Fi` file):

| Solver | X axis | Y axis | Z axis |
|---|---|---|---|
| Temperature | FLUX | FLUX | FLUX |
| Vx | VARIABLE (own axis) | FLUX | FLUX |
| Vy | FLUX | VARIABLE (own axis) | FLUX |
| Vz | FLUX | FLUX | VARIABLE (own axis) |
| Pressure | FLUX | FLUX | FLUX |
| Potential | VARIABLE | FLUX | VARIABLE |

## A real bug found while building this: GPU solver construction order

The CPU solvers only use `Ckor` (needed for the Helmholtz shift) at SOLVE
time, inside `EVD_Thomas`, called every time step -- long after `Call Init`
has already run and set `Ckor = 1.5`. Their build-time position (the
`EVDLapTmpr`/`EVDLapVx/y/z`/`EVDLapP`/`EVD_Fi` calls, originally BEFORE
`Call Init` in `Solution_time`) never depended on `Ckor` being set yet.

The GPU solvers bake the Helmholtz shift in at CONSTRUCTION time
(`Initialize_GPU_Solvers`, see `AlexCudaCompatibility.f90`), which DOES need
`Ckor` already set. Building at the original call site -- before `Call Init`
-- silently captures `Ckor`'s uninitialized value, giving a shift of `0` for
every solver instead of the correct value. This was caught by a structural
test (all six solvers' shift values printed as `0` instead of the expected
~1e6-1e7) and fixed by moving the `Call Initialize_GPU_Solvers()` line to
immediately after `Call Init` in `Solution_time_Q2D.f90` -- the one place in
this migration where the new call site's position genuinely cannot match
the original CPU call's position, for the reason above (documented inline
at that call site as well).

## Pre-existing bugs in Alex's original code, found but NOT fixed

Two bugs were found in the true original source during this work, unrelated
to the GPU migration. Per the instruction to stay as close to the original
as possible, **both are left exactly as they were** -- noted here only so
Alex can decide whether to address them:

1. **`ConvMain_3D_Q2D.f90`**: `Tmpr = Tmpr_Av; VMx = VMx_Av; VMy = VMy_Av;
   VMz = VMx_Az` -- both `Tmpr_Av` and `VMx_Az` are undeclared identifiers;
   under implicit typing, Fortran silently treats them as uninitialized
   local scalars rather than raising a compile error. `Tmpr_Av` was
   evidently meant to be `Tmp_Av` and `VMx_Az` was evidently meant to be
   `VMz_Av`, matching the pattern of the surrounding lines and of the
   equivalent line in `Solution_time_Q2D.f90`.
2. **`modfv_3D_Q2D.f90` / `Solution_time_Q2D.f90`**: `Prs_Av` is declared
   `Allocatable` but never actually `Allocate`d anywhere in the codebase,
   yet `Solution_time_Q2D.f90` uses it (`Prs_Av = Prs_Av + Prs * Htime`,
   etc.) whenever `I_Fourier == 0`. Using an unallocated allocatable array
   is undefined behavior.

## Validation

The six-solver GPU-vs-CPU comparison harness used earlier in this project
(against the previously-reindexed version of this bridge) confirmed all six
solvers agree with Alex's original CPU solutions to within one to two
orders of magnitude of double-precision roundoff:

| Solver | Relative difference (GPU vs. CPU) |
|---|---|
| Temperature | 1.23e-15 |
| Vx | 1.37e-15 |
| Vy | 8.23e-16 |
| Vz | 9.55e-16 |
| Pressure (de-meaned*) | 1.29e-13 |
| Potential (de-meaned*) | 1.35e-15 |

*Pressure is an all-Neumann system and Potential may be, depending on
`EVD_Pot_X/Y/Z`; both are defined only up to an additive constant, so the
comparison subtracts each solution's mean before comparing.

The underlying operators and shift/scaling math in this version are
identical to what was validated above -- only relabeled onto native X/Y/Z
dim slots instead of the earlier Y/Z/X reindexed ones -- so the same
agreement is expected. This version was verified structurally (correct
dims, segment types, and shift values for all six solvers; full compile,
link, and run with zero bounds violations under `-fcheck=all`) but the
CPU-vs-GPU numeric comparison has not yet been rerun against this exact
file set. Rerunning it (the harness code from earlier in this project can
be dropped back in for one validation run and removed afterward) is the
recommended last step before treating this as fully confirmed.

## Files touched

Only these 5 files differ from Alex's original `src/`. Every other file is
byte-identical to the original.

- **`AlexCudaCompatibility.f90`** (new) -- the GPU bridge: builds all six
  eigendecompositions once (`Initialize_GPU_Solvers`), holding six handles.
- **`Solution_time_Q2D.f90`** -- one `Use` statement swapped
  (`Use EVD_Operators` -> `Use AlexCudaCompatibility, only :
  Initialize_GPU_Solvers`); the six `Call EVDLapTmpr` / `EVDLapVx/y/z` /
  `EVDLapP` / `EVD_Fi` calls replaced by one `Call Initialize_GPU_Solvers()`,
  moved to run immediately after `Call Init` (see the bug note above).
- **`time_step_Q2D.f90`** -- each `Call EVD_Thomas`/`EVDmethod` replaced by
  the corresponding `Call solve_eigen_decomp_d`; small scratch buffers
  (`GPU_RHS_T`, `GPU_RHS_Vx/Vy/Vz`) hold the GrPr/DGr-scaled right-hand side
  where that scaling is mathematically required (temperature, velocities);
  pressure passes `Dprs`/`FDRHP` directly with no scratch buffer, since its
  operator needs no scaling.
- **`EM_forcing.f90`** -- `Get_Potential`'s `Call EVDmethod` replaced by
  `Call solve_eigen_decomp_d`, passing `Potential`/`FDRHP` directly (no
  scratch buffer, no scaling -- Potential's operator needs none).
  `EM_force` (the electromagnetic force calculation) is untouched.
- **`ConvMain_3D_Q2D.f90`** -- two `Use` lines added
  (`Use AlexCudaCompatibility`, `Use eigenbcgsolver_eigen_mod, only :
  finalize_eigen_decomp_d`) and one teardown call
  (`Call finalize_eigen_decomp_d()`) added immediately before the program's
  existing `Stop`.

`wrapfEigenBCGSolver_eigen.f90` is generated by CudaBandedLib's own build
(via Shroud) and should not be hand-edited or copied here; see the note in
`AlexCudaCompatibility.f90`'s header if this file needs regenerating.

The ten `EVD_*` CPU solver files, `rg_eispack_R16.f`, and `Mesh_Intrp.f90`
are unused by this integration (nothing calls into them) and can be removed
from `src/` if desired, exactly as in the previous cleanup pass -- see the
prior `DELETE_THESE_FILES.md` for the file list and rationale, which still
applies unchanged.
