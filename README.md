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

Each call to `init_eigen_decomp_*` creates a new solver and returns a solver handle. This handle uniquely identifies the precomputed eigendecomposition and must be supplied to subsequent calls to `solve_eigen_decomp_*`.

Typical usage is:

1. Initialize one solver for each grid or boundary configuration.
2. Save the returned solver handle.
3. Reuse that handle for as many solves as needed.
4. Call `finalize_eigen_decomp_*()` once before program termination to release all eigendecomposition resources.

Example:

```fortran
integer(C_SIZE_T) :: pressureSolver
integer(C_SIZE_T) :: temperatureSolver

pressureSolver = init_eigen_decomp_d(...)

temperatureSolver = init_eigen_decomp_d(...)

call solve_eigen_decomp_d(pressureSolver, xp, bp)

call solve_eigen_decomp_d(temperatureSolver, xt, bt)

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

> **Note:** `init_immersed_eq_*` (section 6) still uses the legacy global `isStaggered` / `dim*UniformDelta` flags and has not been migrated to `dim*SegType`. The two solvers currently have different initialization APIs; don't copy one's argument list to the other.

### Solve Routine (`solve_eigen_decomp_*`)
Performs the spectral solve on the GPU.

| Argument | Type | Description                                                                          |
| :--- | :--- |:-------------------------------------------------------------------------------------|
| solverHandle | integer(C_SIZE_T) | Handle returned by `init_eigen_decomp_*`. |
| x | real array | Output: The solved field.                                                            |
| b | real array | Input: The source term (RHS).  Be sure this is in the column space of the laplacian. |

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

| Routine | Precision | Purpose |
| :--- | :--- | :--- |
| `init_eigen_decomp_d` | Double | Create a new eigendecomposition solver and return its solver handle. |
| `init_eigen_decomp_s` | Single | Create a new eigendecomposition solver and return its solver handle. |
| `solve_eigen_decomp_d` | Double | Solve using an existing solver handle. ($\nabla^2 x = b$ or $\nabla^2 x - \sigma x = b$). |
| `solve_eigen_decomp_s` | Single | Solve using an existing solver handle. ($\nabla^2 x = b$ or $\nabla^2 x - \sigma x = b$). |
| `finalize_eigen_decomp_d` | N/A | Free Eigendecomposition GPU resources. |
| `finalize_eigen_decomp_s` | N/A | Free Eigendecomposition GPU resources. |

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