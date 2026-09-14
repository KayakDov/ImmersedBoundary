# User Guide: Eigen Decomposition and Immersed Boundary Method (IBM) CUDA Solver

This library provides solvers for the following systems:

(1) $$(L - \sigma I) x = b + bc$$

(2) $$(L + 2 B^T B)x = 2 B^T F + p + bc$$

and solves the following system for $p'$ and $F'$:

(3) $$(L + 2 B^T B)p' = 2 B^T ((\frac{3}{2 \Delta t})R^Tu^* - U^\Gamma) + (\frac{-3}{2 \Delta t})\nabla\cdot u^* + bc$$

$$F' = 2(Bp' + (\frac{3}{2 \Delta t})\nabla\cdot u^*)$$

Where $L$ is the laplacian and $bc$ is the right hand side modifier do to boundary conditions.
It uses CUDA-accelerated Eigen Decomposition to handle the Laplacian inversion ($L^{-1}$) and BiCGSTAB to solve the coupled system.
The standalone Direct Eigendecomposition fast poisson solvers for the discrete Poisson and Helmholtz equations have an optional Thomas variant which will offer faster performance with increased numerical error.

---

## 1. Building the Library
First, compile the C++ source.

### Prerequisites
* CMake (version 3.18+)
* CUDA Toolkit (nvcc)
* gfortran (or another compatible Fortran compiler)

### Build Steps
From the project root directory:
1. `mkdir build && cd build`
2. `cmake ..`
3. `make -j$(nproc)`

> Output: Look for `libCudaBandedLib.a` in the build folder. This is the static library you will link against from Fortran.

---

## 2. Fortran Implementation

### The "Persistent" Workflow

The Eigendecomposition solver supports multiple independent solver instances.

Each call to `init_eigen_decomp_*` creates a new solver and returns a solver handle. This handle uniquely identifies the precomputed eigendecomposition and must be supplied to subsequent calls to `solve_eigen_decomp_*`, `synch_*`, and `get_pinned_ptr_*`.

The RHS and the solution share a single pinned host buffer per solver, exposed by `get_pinned_ptr_*`. Fetch this pointer **once**, right after initializing each solver, and alias it via `ISO_C_BINDING`'s `C_F_POINTER` -- the address is stable for the solver's entire lifetime, so there is no need to re-fetch it on every timestep. Write your RHS directly into the aliased array before calling `solve_eigen_decomp_*`; after `synch_*` returns, read the solution back out of that same array. Neither `solve_eigen_decomp_*` nor `synch_*` takes a data argument -- both operate implicitly on whatever is currently sitting in the solver's pinned buffer.

Because solver calls launch work asynchronously on the GPU, you must call `synch_*` before reading the solution out of the pinned buffer, to ensure kernel completion.

Typical usage is:

1. Initialize one solver for each grid or boundary configuration.
2. Save the returned solver handle.
3. Call `get_pinned_ptr_*` once for that handle and alias the result with `C_F_POINTER`.
4. Each timestep: write the RHS into the aliased array, call `solve_eigen_decomp_*` to launch asynchronously, call `synch_*` to wait for completion, then read the solution from the same aliased array.
5. Call `finalize_eigen_decomp_*()` once before program termination to release all eigendecomposition resources.

Example:

```fortran
use, intrinsic :: iso_c_binding

integer(C_SIZE_T) :: pressureSolver
integer(C_SIZE_T) :: temperatureSolver

type(C_PTR) :: pressurePtrRaw, temperaturePtrRaw
real(C_DOUBLE), pointer :: pressureBuf(:), temperatureBuf(:)

! The final argument to init_eigen_decomp_d is gpuIndex (section 7) -- here
! the two solvers are placed on separate physical GPUs and will run
! concurrently rather than competing for the same device's resources.
pressureSolver = init_eigen_decomp_d(..., gpuIndex=0)
temperatureSolver = init_eigen_decomp_d(..., gpuIndex=1)

! Fetch and alias each solver's pinned buffer ONCE -- not once per
! timestep. dim1*dim2*dim3 is the same size passed to the corresponding
! init_eigen_decomp_d call.
pressurePtrRaw = get_pinned_ptr_d(pressureSolver)
call C_F_POINTER(pressurePtrRaw, pressureBuf, [dim1*dim2*dim3])

temperaturePtrRaw = get_pinned_ptr_d(temperatureSolver)
call C_F_POINTER(temperaturePtrRaw, temperatureBuf, [dim1*dim2*dim3])

! --- inside the timestep loop ---

! Write the RHS directly into the aliased pinned array, then launch.
! No separate RHS argument is passed -- solve_eigen_decomp_d operates on
! whatever is currently in the solver's pinned buffer.
pressureBuf = bp
temperatureBuf = bt

call solve_eigen_decomp_d(pressureSolver)
call solve_eigen_decomp_d(temperatureSolver)

! Wait for each handle's GPU work to finish, then read the solution back
! out of the same pinned array the RHS was written into.
call synch_d(pressureSolver)
call synch_d(temperatureSolver)

xp = pressureBuf
xt = temperatureBuf

! --- end timestep loop ---

call finalize_eigen_decomp_d()
```
---

## 3. Direct Eigendecomposition (Standalone)
For problems requiring a direct solution to $(L - \sigma I)x = b$, the library provides an optimized Eigendecomposition solver.

Each of the three logical dimensions independently selects one of four discretizations via `dim*SegType` (see section 9 for the full enum). Two of the four use variable (non-uniform) spacing and differ only in what $L$ means at each row:

* **`VariableDeltaLapl`** -- the pointwise 3-point Laplacian: exact for quadratics at each node, with the implicit assumption that control-volume faces sit at the midpoints between adjacent unknowns.
* **`FluxLapl`** -- the conservative finite-volume Laplacian: unknowns are cell averages at the centres of cells that tile the segment wall-to-wall, and each row is a flux difference divided by that cell's true width. Because it telescopes under summation, `FluxLapl` reproduces $\text{div}(\text{grad})$ discretely -- required whenever the operator must be compatible with a separate divergence operator built on the same mesh (e.g. pressure-projection methods). The two kinds coincide on a locally uniform mesh and differ at wall-adjacent rows and anywhere the mesh stretches; see section 9 for which deltas each one expects.

### Thomas Optimization
The solver includes an optimized "Thomas" variant for the 1D tridiagonal sub-problems. This can be toggled via the `thomas` logical flag during initialization.

---

## 4. Critical Rules for Fortran Programmers

### Indexing: The Zero-Base Trap
Fortran is 1-based, but the underlying CUDA kernels are 0-based.
* **The Rule:** When filling `rowOffsetsB` and `colIndsB` for the sparse matrix $B$, you must subtract 1 from your indices.
* **Example:** To point to the very first node in the grid, your Fortran code must store the value 0.

### Data Types
You must use `iso_c_binding` types to ensure Fortran memory layout matches the GPU:
* `real(C_DOUBLE)` -> Double precision (e.g., `init_*_d`)
* `real(C_FLOAT)`  -> Single precision (e.g., `init_*_s`)
* `integer(C_INT32_T)` -> 4-byte integer
* `integer(C_SIZE_T)` -> 8-byte integer

---

## 5. Resource Management
The solver uses a persistent state on the GPU. Failing to release this state before the Fortran program terminates will result in a `SIGABRT` or a CUDA driver error.

| Routine                         | Purpose                                  |
|:--------------------------------|:-----------------------------------------|
| `finalize_immersed_eq_*()`      | Cleans up the IBM/BiCGSTAB state.        |
| `finalize_eigen_decomp_*()`     | Cleans up the Direct Eigen state.        |

---

## 6. Argument Reference: Immersed Boundary Solver

> **GPU selection:** unlike the Direct Eigendecomposition solver (section 7), `init_immersed_eq_*` has no `gpuIndex` argument. This solver always runs on GPU 0.

### Initialization Routine (`init_immersed_eq_*`)
Allocates GPU memory and pre-computes the Laplacian Eigen Decomposition.

| Argument | Type | Description |
| :--- | :--- | :--- |
| `dim1Length`, `dim2Length`, `dim3Length` | integer(C_SIZE_T) | Number of grid points along the first, second, and third logical dimensions. The solver is isotropic and does not assign any physical meaning (such as X, Y, or Z) to these dimensions. |
| `dim1StartIsNeumann`, `dim1EndIsNeumann` | logical | Boundary-condition type at the beginning and end of the first logical dimension (`.true.` = Neumann, `.false.` = Dirichlet). |
| `dim2StartIsNeumann`, `dim2EndIsNeumann` | logical | Boundary-condition type for the second logical dimension. |
| `dim3StartIsNeumann`, `dim3EndIsNeumann` | logical | Boundary-condition type for the third logical dimension. |
| `dim1StartVal`, `dim1EndVal` | real | Boundary values associated with the first logical dimension. |
| `dim2StartVal`, `dim2EndVal` | real | Boundary values associated with the second logical dimension. |
| `dim3StartVal`, `dim3EndVal` | real | Boundary values associated with the third logical dimension. |
| `isStaggered` | logical | `.true.` if using a staggered grid discretization. |
| `forceSize` | integer(C_SIZE_T) | Size of the force vector. |
| `nnzMax` | integer(C_SIZE_T) | Maximum number of non-zeros permitted in sparse matrix $B$. |
| `p` | real array | Initial pressure vector. |
| `f` | real array | Initial immersed-boundary force vector. |
| `dim1Delta`, `dim2Delta`, `dim3Delta` | real array | Grid spacing arrays corresponding to each logical dimension. For non-uniform grids, each spacing array must contain `dimLength+1` values. For uniform grids, pass a single-element array and set the corresponding `dim*UniformDelta` flag to `.true.`. |
| `dt` | real | Time-step size. |
| `dim1UniformDelta`, `dim2UniformDelta`, `dim3UniformDelta` | logical | `.true.` if the corresponding spacing array contains a single uniform value. |
| `tol` | real | BiCGSTAB convergence tolerance. |
| `maxIterations` | integer(C_SIZE_T) | Maximum number of BiCGSTAB iterations. |

### Solve Routine (`solve_immersed_eq_*`)
Executes the iterative solver for a specific state of CSR matrix $B$ or CSC of $B^T$.

| Argument    | Type | Description |
|:------------| :--- |:------------------------------------------------|
| `result`    | real array | Output: Array overwritten by $x$. |
| `nnzB`      | integer(C_SIZE_T) | Current non-zero count in matrix $B$. |
| `rowOffsetsB`| integer array | Sparse row offsets (MUST BE 0-BASED). |
| `colIndsB`  | integer array | Sparse column indices (MUST BE 0-BASED). |
| `val`       | real array | Non-zero values for matrix $B$. |

### Solve Primes Routine (`solve_immersed_eq_primes_*`)
Executes the iterative solver for the coupled Pressure ($P'$) and Force ($F'$) system.

| Argument | Type | Description |
| :--- | :--- | :--- |
| `resultPPrime` | real array | Output: Array overwritten by $P'$. |
| `resultFPrime` | real array | Output: Array overwritten by $F'$. |
| `nnzB` | integer(C_SIZE_T) | Current non-zero count in matrix $B$. |
| `rowOffsetsB` | integer array | Sparse row offsets for $B$ (MUST BE 0-BASED). |
| `colIndsB` | integer array | Sparse column indices for $B$ (MUST BE 0-BASED). |
| `valuesB` | real array | Non-zero values for matrix $B$. |
| `nnzR` | integer(C_SIZE_T) | Current non-zero count in matrix $R$. |
| `colOffsetsR` | integer array | Sparse column offsets for $R$ (MUST BE 0-BASED). |
| `rowIndsR` | integer array | Sparse row indices for $R$ (MUST BE 0-BASED). |
| `valuesR` | real array | Non-zero values for matrix $R$. |
| `UGamma` | real array | Immersed boundary velocity vector $\Gamma$. |
| `uStar` | real array | Intermediate velocity field $u^*$. The first contiguous third should be the x component of each velocity vector, then y, then z component of the velocity vector. |

---

## 7. Argument Reference: Direct Eigen Solver

### Initialization Routine (`init_eigen_decomp_*`)

Creates a new eigendecomposition solver and returns a solver handle.

| Return Value | Type | Description |
| :--- | :--- | :--- |
| solverHandle | integer(C_SIZE_T) | Identifier used in subsequent solve calls. |

| Argument | Type | Description                                                                                                                                                                                                                                                                              |
| :--- | :--- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dim1Length`, `dim2Length`, `dim3Length` | integer(C_SIZE_T) | Number of grid points along the first, second, and third logical dimensions. The solver is isotropic and does not assign any physical meaning (such as X, Y, or Z) to these dimensions.                                                                                                  |
| `dim1Delta`, `dim2Delta`, `dim3Delta` | real array | Grid spacing array for each logical dimension; see section 9 for the exact size and content required by each `dim*SegType`.                                                                                                                                                              |
| `dim1SegType`, `dim2SegType`, `dim3SegType` | integer(C_INT) | Discretization kind for each logical dimension. One of `UNIFORM_NODE_CENTERED_LAPL`, `UNIFORM_STAGGERED_LAPL`, `VARIABLE_DELTA_LAPL`, or `FLUX_LAPL` (see section 9). Replaces the old `isStaggered`/`dim*UniformDelta` flags -- every axis states its own discretization independently. |
| `dim1StartIsNeumann`, `dim1EndIsNeumann` | logical | Boundary-condition type at the beginning and end of the first logical dimension (`.true.` = Neumann, `.false.` = Dirichlet).                                                                                                                                                             |
| `dim2StartIsNeumann`, `dim2EndIsNeumann` | logical | Boundary-condition type for the second logical dimension.                                                                                                                                                                                                                                |
| `dim3StartIsNeumann`, `dim3EndIsNeumann` | logical | Boundary-condition type for the third logical dimension.                                                                                                                                                                                                                                 |
| `dim1StartVal`, `dim1EndVal` | real | Boundary values associated with the first logical dimension.                                                                                                                                                                                                                             |
| `dim2StartVal`, `dim2EndVal` | real | Boundary values associated with the second logical dimension.                                                                                                                                                                                                                            |
| `dim3StartVal`, `dim3EndVal` | real | Boundary values associated with the third logical dimension.                                                                                                                                                                                                                             |
| `thomas` | logical | `.true.` to use the optimized Thomas variant for the direct eigendecomposition solver.                                                                                                                                                                                                   |
| `helmholtzShift` | real | Scalar shift $\sigma$. Set to `0.0` for the Poisson equation or a non-zero value to solve $(L - \sigma I)x = b$. If all boundary conditions are neumann, and \sigma > 0, be sure \sigma is not an eigenvalue of L.                                                                       |
| `gpuIndex` | integer(C_SIZE_T) | Which GPU this solver's data and computation live on. See [GPU Selection](#gpu-selection) below. |

### GPU Selection

`gpuIndex` is a 0-based index into whichever GPUs are visible to the process -- the same numbering `nvidia-smi -L` and `CUDA_VISIBLE_DEVICES` use. Pass `0` for the first (and on a single-GPU machine, only) device.

Each call to `init_eigen_decomp_*` is independent: different solver handles may be created with different `gpuIndex` values and run concurrently on separate physical GPUs. A given solver's `solve_eigen_decomp_*`/`synch_*` calls always operate on whichever GPU that solver was initialized with -- there is nothing further to pass at solve time.

**This applies only to the Direct Eigendecomposition solver (`init_eigen_decomp_*`).** The Immersed Boundary solver (`init_immersed_eq_*`, section 6) does not currently accept a `gpuIndex` and always runs on GPU 0, regardless of what else is passed to it.

### Pinned Buffer Accessor (`get_pinned_ptr_*`)
Returns the raw pinned host pointer a solver uses for both its RHS and its solution -- see section 2 for the intended call-once-and-alias usage pattern.

| Argument | Type | Description |
| :--- | :--- | :--- |
| solverHandle | integer(C_SIZE_T) | Handle returned by `init_eigen_decomp_*`. |

| Return Value | Type | Description |
| :--- | :--- | :--- |
| (unnamed) | `type(C_PTR)` | Pointer to `dim1*dim2*dim3` elements of pinned host memory. Alias it once via `C_F_POINTER`; the address does not change for the solver's lifetime. Write the RHS into it before `solve_eigen_decomp_*`; read the solution from it after `synch_*`. |

### Solve Routine (`solve_eigen_decomp_*`)
Launches the spectral solve on the GPU, using whatever is currently in the solver's pinned buffer as the RHS.

| Argument | Type | Description                                                                          |
| :--- | :--- |:-------------------------------------------------------------------------------------|
| solverHandle | integer(C_SIZE_T) | Handle returned by `init_eigen_decomp_*`. |

Takes no data argument: write the RHS into the pointer from `get_pinned_ptr_*` before calling this. Be sure it is in the column space of the laplacian. Returns immediately once the GPU work is enqueued, without waiting for it to finish.

### Synchronization Routine (`synch_*`)
Blocks host execution until GPU operations for the given solver handle are complete.

| Argument | Type | Description |
| :--- | :--- | :---|
| solverHandle | integer(C_SIZE_T) | Handle returned by `init_eigen_decomp_*`. |

Takes no data argument: after this returns, read the solution directly from the same pointer `get_pinned_ptr_*` returned.
---

## 8. Compiling & Linking

To create your executable, link the C++ library and the CUDA runtimes:

1. **Compile your Fortran source:**
   `gfortran -c main.f90`

2. **Link everything:**
   `gfortran main.o -L./build -lCudaBandedLib -lstdc++ -lcudart -o ibm_solver`

* `-lCudaBandedLib`: Your newly built library.
* `-lstdc++`: Required for C++ compatibility.
* `-lcudart`: The CUDA Runtime library.

---

## 9. API Reference: Method & Type Variations

The library uses a consistent naming convention to denote data types:
* `_d`: `real(C_DOUBLE)`
* `_s`: `real(C_FLOAT)`
* `_i32`: `integer(C_INT32_T)` sparse indices
* `_i64`: `integer(C_INT64_T)` sparse indices

### Module: `eigenbcgsolver_imeq_mod`
This module provides the coupled Immersed Boundary Method solvers.

| Routine | Precision | Index Type | Purpose |
| :--- | :--- | :--- | :--- |
| `init_immersed_eq_d_i32` | Double | 32-bit | Initialize IBM environment. |
| `init_immersed_eq_s_i32` | Single | 32-bit | Initialize IBM environment. |
| `init_immersed_eq_d_i64` | Double | 64-bit | Initialize IBM environment. |
| `init_immersed_eq_s_i64` | Single | 64-bit | Initialize IBM environment. |
| `solve_immersed_eq_d_i32` | Double | 32-bit | Solve for Grid Pressure ($x$). |
| `solve_immersed_eq_s_i32` | Single | 32-bit | Solve for Grid Pressure ($x$). |
| `solve_immersed_eq_d_i64` | Double | 64-bit | Solve for Grid Pressure ($x$). |
| `solve_immersed_eq_s_i64` | Single | 64-bit | Solve for Grid Pressure ($x$). |
| `solve_immersed_eq_primes_d_i32` | Double | 32-bit | Solve for coupled Pressure ($P'$) and Force ($F'$). |
| `solve_immersed_eq_primes_s_i32` | Single | 32-bit | Solve for coupled Pressure ($P'$) and Force ($F'$). |
| `solve_immersed_eq_primes_d_i64` | Double | 64-bit | Solve for coupled Pressure ($P'$) and Force ($F'$). |
| `solve_immersed_eq_primes_s_i64` | Single | 64-bit | Solve for coupled Pressure ($P'$) and Force ($F'$). |
| `finalize_immersed_eq_d_i32` | N/A | N/A | Free IBM GPU resources. |
| `finalize_immersed_eq_s_i32` | N/A | N/A | Free IBM GPU resources. |
| `finalize_immersed_eq_d_i64` | N/A | N/A | Free IBM GPU resources. |
| `finalize_immersed_eq_s_i64` | N/A | N/A | Free IBM GPU resources. |

---

### Module: `eigenbcgsolver_eigen_mod`
This module provides standalone direct Eigendecomposition solvers for the Poisson equation.

| Routine                   | Precision | Purpose |
|:--------------------------| :--- | :--- |
| `init_eigen_decomp_d`     | Double | Create a new eigendecomposition solver and return its solver handle. |
| `init_eigen_decomp_s`     | Single | Create a new eigendecomposition solver and return its solver handle. |
| `get_pinned_ptr_d`        | Double | Return the solver's pinned host buffer as a `type(C_PTR)`. Call once per handle and alias with `C_F_POINTER`; do not re-fetch every timestep. |
| `get_pinned_ptr_s`        | Single | Return the solver's pinned host buffer as a `type(C_PTR)`. Same usage as `get_pinned_ptr_d`. |
| `solve_eigen_decomp_d`    | Double | Launch a solve on an existing handle ($\nabla^2 x = b$ or $\nabla^2 x - \sigma x = b$), using whatever is currently in the solver's pinned buffer as `b`. Takes no data argument. Returns without waiting for the GPU. |
| `solve_eigen_decomp_s`    | Single | Launch a solve on an existing handle. Same behavior as `solve_eigen_decomp_d`. |
| `synch_d`                 | Double | Wait for a handle's launched solve to finish. Takes no data argument -- the solution is available afterward by reading the pointer `get_pinned_ptr_d` returned for this handle. |
| `synch_s`                 | Single | Wait for a handle's launched solve to finish. Same behavior as `synch_d`. |
| `finalize_eigen_decomp`   | N/A | Free Eigendecomposition GPU resources. |

### Segment Types (`dim*SegType`)

Each logical dimension's discretization is selected independently via a plain integer, matching `eigen::LaplOperatorT`:

| Constant | Value | Deltas required | Description |
| :--- | :--- | :--- | :--- |
| `UNIFORM_NODE_CENTERED_LAPL` | 0 | Single-element array (one uniform spacing) | Uniform spacing, unknowns at grid nodes. |
| `UNIFORM_STAGGERED_LAPL` | 1 | Single-element array (one uniform spacing) | Uniform spacing, unknowns at cell centres, walls half a spacing beyond the first/last unknown. |
| `VARIABLE_DELTA_LAPL` | 2 | `dimLength+1` values: free, independent centre-to-centre/wall-to-centre distances | Non-uniform spacing, pointwise 3-point Laplacian (faces implicitly at midpoints between unknowns). |
| `FLUX_LAPL` | 3 | `dimLength+1` values, but **constrained**: they must be the wall/centre distances of `dimLength` cells that tile the axis wall-to-wall (`d(0) = W(0)/2`, `d(i) = (W(i-1)+W(i))/2`, `d(n) = W(n-1)/2` for cell widths `W`) | Non-uniform spacing, conservative finite-volume Laplacian. The library reconstructs the cell widths from the deltas and will reject (`std::invalid_argument`) a delta array that isn't a valid wall-to-wall tiling -- free-form deltas that would be valid for `VARIABLE_DELTA_LAPL` are generally NOT valid for `FLUX_LAPL`. |

`FLUX_LAPL` and `VARIABLE_DELTA_LAPL` may be mixed freely across `dim1`/`dim2`/`dim3` in the same solver, and with either uniform kind, in any combination -- a single call can e.g. use `FLUX_LAPL` on `dim1`, `UNIFORM_STAGGERED_LAPL` on `dim2`, and `VARIABLE_DELTA_LAPL` on `dim3`.

### Global Configuration
Before calling any of the init methods, you may configure the global input format to define how your grid's flattened array is interpreted by the solver.

| Constant | Value | Description |
| :--- | :--- | :--- |
| `INPUT_FORMAT_XYZ` | 0 | Columns (X) change fastest, then rows (Y), then layers (Z). A form of row-major order. |
| `INPUT_FORMAT_YXZ` | 1 | Rows (Y) change fastest, then columns (X), then layers (Z). A form of column-major order. |
| `INPUT_FORMAT_YZX` | 2 | Rows (Y) change fastest, then layers (Z), then columns (X). This is the default and fastest format. |

| Routine | Purpose |
| :--- | :--- |
| `set_global_input_format` | Sets the grid interpretation format for subsequent solver initializations. |

Note, there is only one input format stored globally. The most recent value set is the global value for all subsequent function calls.

---

## 10. Flattened Indexing

When mapping multi-dimensional grids to a flattened 1D array, **`dim1` is the dimension whose flattened indices change fastest, then `dim2`, and `dim3`'s flattened indices change slowest.**

This is a column-major style indexing layout, where all elements along `dim1` are iterated over before advancing to the next index in `dim2`, and so on.
