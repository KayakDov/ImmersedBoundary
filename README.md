# User Guide: Immersed Boundary Method (IBM) CUDA Solver

This library provides a high-performance solver for the following system:

$$(L + 2 B^T B)x = B^T f + p$$

It uses CUDA-accelerated Eigen Decomposition to handle the Laplacian inversion ($L^{-1}$) and BiCGSTAB to solve the coupled system. Additionally, the library exposes standalone Direct Eigendecomposition solvers for the discrete Poisson equation.

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

> Output: Look for `libCudaBandedLib.a` in the build folder. This is the static library you will link against.

---

## 2. Fortran Implementation

### The "Persistent" Workflow
Unlike a standard function call, this library maintains a "state" on the GPU to maximize throughput.
1. **Initialize Once:** The GPU allocates memory and pre-calculates eigenvalues.
2. **Solve Many Times:** Call the solve routine inside your loops. You can update the boundary matrix ($B$) every time you call it without re-initializing.
3. **Finalize Once:** Release GPU resources before program termination.

### Implementation Examples
For a complete working example of both the Immersed Boundary and the Direct Eigen solvers, please refer to the source code and build scripts in the `/FortranTest` subfolder.

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
| height, width, depth | integer(C_SIZE_T) | Grid dimensions (Y, X, Z). |
| nnz | integer(C_SIZE_T) | Max non-zeros allowed in matrix B. |
| p | real array | Pressure vector (Size: H*W*D). |
| f | real array | Force vector. (Size heightB) |
| dx, dy, dz | real(C_DOUBLE) | Physical grid spacing. |
| tolerance | real(C_DOUBLE) | Solver convergence threshold. |
| maxIter | integer(C_SIZE_T) | Max iterations for the BiCGSTAB solver. |

### Solve Routine (`solve_immersed_eq_*`)
Executes the iterative solver for a specific state of CSR matrix $B$ or CSC of $B^T$.

| Argument    | Type | Description |
|:------------| :--- |:------------------------------------------------|
| result      | real array | Output: Array overwritten by $x$. |
| nnzB        | integer(C_SIZE_T) | Current non-zero count in matrix $B$. |
| offsetsB    | integer array | Sparse row offsets (MUST BE 0-BASED). |
| indsB       | integer array | Sparse column indices (MUST BE 0-BASED). |
| valuesB     | real array | Non-zero values for matrix $B$. |

---

## 7. Argument Reference: Direct Eigen Solver

### Initialization Routine (`init_eigen_decomp_*`)
Pre-calculates the spectral basis for the Laplacian on the given grid.

| Argument | Type | Description |
| :--- | :--- | :--- |
| rows, cols, layers | integer(C_SIZE_T) | Grid dimensions. |
| dx, dy, dz | real(C_DOUBLE) | Grid spacing. |
| thomas | logical | `.true.` to use optimized Thomas algorithm. |

### Solve Routine (`solve_eigen_decomp_*`)
Performs the spectral solve on the GPU.

| Argument | Type | Description |
| :--- | :--- | :--- |
| x | real array | Output: The solved field. |
| b | real array | Input: The source term (RHS). |

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
| `init_eigen_decomp_d` | Double | Initialize Eigendecomposition / spectral basis. |
| `init_eigen_decomp_s` | Single | Initialize Eigendecomposition / spectral basis. |
| `solve_eigen_decomp_d` | Double | Direct spectral solve ($\nabla^2 x = b$). |
| `solve_eigen_decomp_s` | Single | Direct spectral solve ($\nabla^2 x = b$). |
| `finalize_eigen_decomp_d` | N/A | Free Eigendecomposition GPU resources. |
| `finalize_eigen_decomp_s` | N/A | Free Eigendecomposition GPU resources. |