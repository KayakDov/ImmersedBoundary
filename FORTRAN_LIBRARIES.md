# The two Fortran interfaces

CudaBandedLib is a C++/CUDA library. Its Fortran side is two separate interfaces, generated with
[Shroud](https://shroud.readthedocs.io/) into `shroud_project/`. They exist because two research
supervisors, each with their own Fortran CFD solver, needed different features from the GPU library.
They share the core library (the GPU containers, handles and streams, and the eigen-decomposition
and BiCGSTAB solvers) but are independent of each other, and a Fortran program can use either one or both.

| | Eigen-decomposition interface | Immersed-boundary interface |
|---|---|---|
| Fortran module | `eigenbcgsolver_eigen_mod` (`shroud_project/wrapfEigenBCGSolver_eigen.f90`) | `eigenbcgsolver_imeq_mod` (`shroud_project/wrapfEigenBCGSolver_ImEq.f90`) |
| C++ entry points | `wrapper/FortranBindings.hpp` (`initEigenDecompSolver`, `runDecompSolver`, `synch`, `pinnedPtr`) | `wrapper/FortranBindings.hpp`, namespace `ImEq` (`initImmersedEq`, `solveImmersedEq`, `finalizeImmersedEq`) |
| Written for | Prof. Alexander Gelfgat's 3D Fortran CFD solver (Tel Aviv University) | Prof. Yuri Feldman's SchurSR Fortran solver (Ben-Gurion University) |
| What it solves | Separable Poisson and Helmholtz systems `L x = b` on a Cartesian grid | The coupled pressure-force system of the immersed-boundary method, `(L + 2 BᵀB) x = 2 Bᵀ f + p` |
| Method | Direct, by eigen-decomposition of the 1D operators (optionally Thomas along one axis) | Iterative (BiCGSTAB) with the eigen-decomposition solver as the inverse of `L` |
| Precision and indices | `double` and `float` | `double` and `float`, 32-bit and 64-bit sparse indices |

## Eigen-decomposition interface

The features this interface has because the solver it was written for needed them:

- Dirichlet or Neumann boundary conditions on each end of each axis.
- Uniform, variable-spacing, and flux-form Laplacian discretizations, chosen independently for each axis.
- An optional Helmholtz shift.
- An optional Thomas-algorithm path along one axis.
- A choice of which GPU each solver runs on.
- A zero-copy host buffer. The Fortran caller writes the right-hand side into one pinned buffer,
  calls the solve, and reads the solution back from the same buffer.

## Immersed-boundary interface

- Takes the boundary operators as sparse matrices (`B` in CSR form, `R` in CSC form) and solves for the
  pressure correction and the boundary-force correction.
- Reuses the same eigen-decomposition solver for the Laplacian part, so the two interfaces share that code.

## Layout

- `src/` is the C++/CUDA library and `src/wrapper/` the code the interfaces call.
- `shroud_project/` holds the generated Fortran modules and their C++ glue. They are compiled into the library by CMake.
- `Tests/` holds unit tests for the core library and for both interfaces.
