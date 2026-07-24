# Validation package: CPU-vs-GPU harness at real (100x100x100) resolution

This is the NO-REINDEXING version's validation harness -- it needs no
transpose/reordering step at all, since Alex's original arrays and the GPU
library both use the same native (X,Y,Z) order in this version.

## Apply

1. Copy all `.f90`/`.f` files in this package into `src/`, alongside your
   current `AlexCudaCompatibility.f90`, `time_step_Q2D.f90`, and
   `EM_forcing.f90` from the no-reindexing delivery (unchanged, not part of
   this package). This adds the ten `EVD_*` CPU solver files (one dormant,
   normally-harmless typo in `EVD_solver_DGEMM_3D_v2.f90` fixed for
   bounds-check safety -- doesn't change behavior for a cubic Nx=Ny=Nz grid
   like your current `Conv.dat`) and the new `ValidateSolvers.f90`.

   **`Solution_time_Q2D.f90` in this package REPLACES your current one** --
   it's your no-reindexing delivery's version with two "VALIDATION ONLY"
   lines added (a `Use ValidateGPU` and a `Call Validate_GPU_Solvers`,
   placed right after `Call Initialize_GPU_Solvers()`, i.e. after `Ckor` is
   guaranteed set). Remove those two lines (or just restore the delivery's
   original `Solution_time_Q2D.f90`) once you're done validating.

   **Do NOT add a separate `EVD_Modules.f90`** -- your true-original
   `modfv_3D_Q2D.f90` already defines the `EVD_Operators` /
   `Thomas_coefficients` modules these files need inline; adding a second
   copy causes duplicate-symbol link errors (caught and fixed during
   testing here).

2. Rebuild `CFDSolver` and run it against your real `Conv.dat`
   (100x100x100). This will take noticeably longer than a normal run: it
   builds all six CPU reference operators from scratch (Vgeev-decomposing
   dense matrices up to 101x101) before the harness prints its report and
   the normal time-stepping begins.

## What you'll see

Right after the six CPU-operator "max (inverse matrix)"/"max (eigenvalue
decomposition)" sanity-check lines (from Alex's own `EVDLapTmpr`/etc. --
these run regardless, they're part of building the CPU reference), a block
like:

```
 ================ CPU vs GPU solver validation (native X,Y,Z; N= 100) ================
   Temperature  max|cpu-gpu| = ...   relative = ...
   Vx           ...
   Vy           ...
   Vz           ...
   Pressure     ...  (de-meaned)
   Potential    ...  (de-meaned)
 ===============================================================
```

## How to read it

At the 3x2x4 debug grid, all six solvers agreed with the CPU reference to
1e-13 to 1e-16 relative -- machine precision. Given the observed physics
divergence at 100x100x100 (RDP growing geometrically instead of decaying),
at least one of these six numbers is expected to be much larger than that
at this resolution.

- **If Pressure's relative difference is large** (say, > 1e-6): this
  confirms the projection-compatibility hypothesis directly -- the
  FLUX_LAPL pressure operator disagrees with Alex's CPU operator at this
  scale, which would explain the geometric RDP growth exactly (residual
  divergence not fully removed each step, compounding).
- **If Temperature/Vx/Vy/Vz are large** instead (or as well): this points
  somewhere else -- possibly a genuine precision issue in the GPU
  eigendecomposition at this size that isn't specific to FLUX_LAPL, since
  Vx/Vy/Vz each also use FLUX_LAPL on two of their three axes.
- **If ALL SIX are still ~1e-13 or better**: the individual solvers are all
  correct at this resolution, and the bug is somewhere else entirely --
  most likely in how the six solves interact over many time steps (e.g. an
  accumulation effect, or something in `time_step_Q2D.f90`'s scratch-buffer
  reuse) rather than in any single solve. This would need a different
  investigation.

Paste the full six-line report back and I'll take it from there.
