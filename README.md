# User Guide: Eigen Decomposition and Immersed Boundary Method (IBM) CUDA Solver

This library provides solvers for the following systems:

(1) $$L x = b + bc$$

(2) $$(L + 2 B^T B)x = 2 B^T F + p + bc$$

and solves the following system for $p'$ and $F'$:

(3) $$(L + 2 B^T B)p' = 2 B^T ((\frac{3}{2 \Delta t})R^Tu^* - U^\Gamma) + (\frac{-3}{2 \Delta t})\nabla\cdot u^* + bc$$

$$F' = 2(Bp' + (\frac{3}{2 \Delta t})\nabla\cdot u^*)$$

Where $L$ is the laplacian and $bc$ is the right hand side modifier do to boundary conditions.
It uses CUDA-accelerated Eigen Decomposition to handle the Laplacian inversion ($L^{-1}$) and BiCGSTAB to solve the coupled system. Additionally, the library exposes standalone Direct Eigendecomposition solvers for the discrete Poisson equation with an optional Thomas variant.

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
For problems requiring a direct solution to $L x = b$, the library provides an optimized Eigendecomposition solver.

### Thomas Optimization
The solver includes an optimized "Thomas" variant for the 1D tridiagonal sub-problems. This can be toggled via the `thomas` logical flag during initialization.

---

## 4. Critical Rules for Fortran Programmers

### Indexing: The Zero-Base Trap
Fortran is 1-based, but the underlying CUDA kernels are 0-based.
* **The Rule:** When filling `rowPtrs` and `colOffsets` for the sparse matrix $B$, you must subtract 1 from your indices.
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
| :--- | :--- |:-------------------------------------------|
| gridHeight, gridWidth, gridDepth | integer(C_SIZE_T) | Grid dimensions (Y, X, Z). |
| leftIsNeumann ... frontIsNeumann | logical | Boundary condition type flags (`.true.` = Neumann, `.false.` = Dirichlet). |
| leftVal ... backVal | real | Boundary condition values (derivative or constant). |
| isStaggered | logical | `.true.` if using a staggered grid discretization. |
| nnzMaxB | integer(C_SIZE_T) | Max non-zeros allowed in matrix $B$. |
| p | real array | Pressure vector (Size: H*W*D). |
| f | real array | Force vector (Size: heightB). |
| dx, dy, dz | real array | Physical grid spacing arrays (Size 1 if uniform, otherwise axis dimension + 1). |
| dt | real | Time step size. |
| uniformDeltaX, Y, Z | logical | `.true.` if the corresponding delta array is uniform (single element). |
| tolerance | real | Solver convergence threshold. |
| maxBCGIterations | integer(C_SIZE_T) | Max iterations for the BiCGSTAB solver. |

### Solve Routine (`solve_immersed_eq_*`)
Executes the iterative solver for a specific state of CSR matrix $B$ or CSC of $B^T$.

| Argument    | Type | Description |
|:------------| :--- |:------------------------------------------------|
| result      | real array | Output: Array overwritten by $x$. |
| nnzB        | integer(C_SIZE_T) | Current non-zero count in matrix $B$. |
| offsetsB    | integer array | Sparse row offsets (MUST BE 0-BASED). |
| indsB       | integer array | Sparse column indices (MUST BE 0-BASED). |
| valuesB     | real array | Non-zero values for matrix $B$. |

### Solve Primes Routine (`solve_immersed_eq_primes_*`)
Executes the iterative solver for the coupled Pressure ($P'$) and Force ($F'$) system.

| Argument | Type | Description |
| :--- | :--- | :--- |
| resultPPrime | real array | Output: Array overwritten by $P'$. |
| resultFPrime | real array | Output: Array overwritten by $F'$. |
| nnzB | integer(C_SIZE_T) | Current non-zero count in matrix $B$. |
| rowOffsetsB | integer array | Sparse row offsets for $B$ (MUST BE 0-BASED). |
| colIndsB | integer array | Sparse column indices for $B$ (MUST BE 0-BASED). |
| valuesB | real array | Non-zero values for matrix $B$. |
| nnzR | integer(C_SIZE_T) | Current non-zero count in matrix $R$. |
| colOffsetsR | integer array | Sparse column offsets for $R$ (MUST BE 0-BASED). |
| rowIndsR | integer array | Sparse row indices for $R$ (MUST BE 0-BASED). |
| valuesR | real array | Non-zero values for matrix $R$. |
| UGamma | real array | Immersed boundary velocity vector $\Gamma$. |
| uStar | real array | Intermediate velocity field $u^*$. |

---

## 7. Argument Reference: Direct Eigen Solver

### Initialization Routine (`init_eigen_decomp_*`)

Creates a new eigendecomposition solver and returns a solver handle.

| Return Value | Type | Description |
| :--- | :--- | :--- |
| solverHandle | integer(C_SIZE_T) | Identifier used in subsequent solve calls. |

| Argument | Type | Description |
| :--- | :--- | :--- |
| rows, cols, layers | integer(C_SIZE_T) | Grid dimensions. |
| dx, dy, dz | real array | Grid spacing arrays (Size 1 if uniform, otherwise axis dimension + 1). |
| uniformDeltaX, Y, Z | logical | `.true.` if the corresponding delta array is uniform (single element). |
| leftIsNeumann ... frontIsNeumann | logical | Boundary condition type flags (`.true.` = Neumann, `.false.` = Dirichlet). |
| leftVal ... backVal | real | Boundary condition values (derivative or constant). |
| isStaggered | logical | `.true.` if using a staggered grid discretization. |
| thomas | logical | `.true.` to use optimized Thomas algorithm. |

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
| `solve_eigen_decomp_d` | Double | Solve using an existing solver handle. ($\nabla^2 x = b$). |
| `solve_eigen_decomp_s` | Single | Solve using an existing solver handle. ($\nabla^2 x = b$). |
| `finalize_eigen_decomp_d` | N/A | Free Eigendecomposition GPU resources. |
| `finalize_eigen_decomp_s` | N/A | Free Eigendecomposition GPU resources. |

## 10. Flatened Indexing
When mapping from (row, col, layer) = (y, x, z) indices to a flatened indexing, 
y changes fastest, then z, then x changes slowest.  That is a column major indexing where all the first columns
in each later are would be itereatred over before all the second columns in each layer and so on.