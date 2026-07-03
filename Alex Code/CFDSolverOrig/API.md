# API.md

## Project Overview

This project is Alex's 3D Q2D CFD solver.

The numerical solver consists of two largely independent pieces:

1. **Physics**

   * Momentum
   * Pressure correction
   * Temperature
   * Boundary conditions
   * Time stepping

2. **Linear Solver**

   * Eigenvalue decomposition (EVD)
   * Thomas solver
   * DGEMM transforms

The long-term goal is to replace the entire EVD subsystem with calls to **CudaBandedLib** while leaving the physics unchanged.

---

# Programs

## ConvMain_3D_Q2D.f90

Main CFD program.

Responsibilities:

* Reads/initializes problem.
* Creates mesh.
* Runs time-stepping loop.
* Calls pressure, velocity and temperature solvers.
* Writes output.

Future:

* Replace EVD calls with CudaBandedLib.

---

## Mesh_Intrp.f90

Standalone utility.

Not part of the CFD solver.

Performs mesh interpolation / preprocessing.

Build as a separate executable.

---

# Global Modules

## modfv_3D_Q2D.f90

Defines nearly all shared data.

Contains:

* grid sizes
* solution arrays
* parameters
* global variables

Equivalent to the benchmark's `modfv_3D.f90`.

---

# Mesh

## MeshStretch.f90

Constructs computational mesh.

Produces:

* X
* Y
* Z

and spacing arrays used by finite differences.

---

## Mesh_Intrp.f90

Mesh interpolation utility.

Not used by the CFD executable.

---

# Time Integration

## time_step_Q2D.f90

One timestep of the CFD solver.

Coordinates:

* pressure
* velocity
* temperature

---

## Solution_time_Q2D.f90

Overall transient solution driver.

Controls simulation progression.

---

# Initialization

## init_3D.f90

Initializes fields.

Sets initial conditions.

---

# Boundary Conditions

## bounds_3D.f90

Applies boundary conditions.

---

## sublid_3D.f90

Lid-driven cavity boundary conditions.

---

# Pressure

## gradp_3D.f90

Pressure gradient.

---

## divvel_3D_alloc.f90

Velocity divergence.

Used in pressure correction.

---

# Velocity

## vgradf_3D.f90

Velocity convection terms.

---

## ekinem_3D.f90

Kinetic energy calculations.

---

# Output

## outp_3D.f90

Writes simulation output.

---

## check_3D.f90

Diagnostics.

---

## Yaverage.f90

Computes Y-averaged quantities.

---

## prmesh.f90

Mesh printing/output.

---

## pointw.f90

Point interpolation / output helper.

---

# Electromagnetics

## EM_forcing.f90

Electromagnetic forcing terms.

---

# Eigenvalue Solver (to be replaced)

## EVD_eigenvector_ag1_R16.f90

Eigenvector generation.

Replace with:

CudaBandedLib internal eigensolver.

---

## EVD_solver_DGEMM_3D_v2.f90

Dense eigenvector solver.

Replace with:

```
solve_eigen_decomp_d(..., thomas=.false.)
```

---

## EVD_Thomas3D_dgemm_OMP_v2_time.f90

Thomas-based solver.

Replace with:

```
solve_eigen_decomp_d(..., thomas=.true.)
```

---

## EVD_laptmpr_3D.f90

Temperature Laplacian construction.

Will disappear.

---

## EVD_lapP_3D.f90

Pressure Laplacian construction.

Will disappear.

---

## EVD_lapVx_3D.f90

Velocity-X Laplacian.

Will disappear.

---

## EVD_lapVy_3D.f90

Velocity-Y Laplacian.

Will disappear.

---

## EVD_lapVz_3D.f90

Velocity-Z Laplacian.

Will disappear.

---

## EVD_lap_Fi_3D.f90

Additional scalar Laplacian.

Will disappear.

---

## rg_eispack_R16.f

Legacy EISPACK eigensolver.

No longer needed after migration to CudaBandedLib.

---

# Migration Plan

1. Build original solver unchanged.
2. Verify numerical results.
3. Replace temperature solver.
4. Replace pressure solver.
5. Replace velocity solvers.
6. Remove all EVD_* files.
7. Remove rg_eispack_R16.f.
8. Final CFD solver depends only on CudaBandedLib.

