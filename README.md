# User Guide: Immersed Boundary Method (IBM) CUDA Solver

This library provides a high-performance solver for the following system:

(I + 2L^-1 B^T B)x = L^-1(B^T f + p)

It uses CUDA-accelerated Eigen Decomposition to handle the Laplacian inversion (L^-1) and BiCGSTAB to solve the coupled system.

---

## 1. Building the Library
First, compile the C++.

### Prerequisites
* CMake (version 3.18+)
* CUDA Toolkit (nvcc)
* gfortran (or another compatible Fortran compiler)

### Build Steps
From the project root directory:
1. mkdir build && cd build
2. cmake ..
3. make -j$(nproc)

> Output: Look for "libCudaBandedLib.a" in the build folder. This is the static library you will link against.

---

## 2. Fortran Implementation

### The "Persistent" Workflow
Unlike a standard function call, this solver maintains a "state" on the GPU to save time.
1. Initialize Once: The GPU allocates memory and pre-calculates eigenvalues.
2. Solve Many Times: Call the solve routine inside your loops. You can update the boundary matrix (B) every time you call it without re-initializing.

### Code Snippet
program test_solver
use iso_c_binding
use fortranbindings_mod
implicit none

    ! 1. Setup dimensions (e.g., 3x2x2 grid = 12 nodes)
    integer(C_SIZE_T) :: h=3, w=2, d=2
    real(C_DOUBLE)    :: p(12), f(2), result(12)
    
    ! 2. Initialize the GPU environment
    call init_immersed_eq_d_i32(h, w, d, 2_C_SIZE_T, p, f, & 1.0_d0, 1.0_d0, 1.0_d0, 1e-6_d0, 100_C_SIZE_T)

    ! 3. Execute Solver
    call solve_immersed_eq_d_i32(result, nnzB, rowPtrs, colOffsets, values, .true.)

    end program test_solver

---

## 3. Critical Rules for Fortran Programmers

### Indexing: The Zero-Base Trap
Fortran is 1-based, but CUDA is 0-based.
* The Rule: When filling rowPtrs and colOffsets for the sparse matrix B, you must subtract 1 from your indices.
* Example: To point to the very first node in the grid, your Fortran code must store the value 0.

### Data Types
You must use "iso_c_binding" types to ensure Fortran memory matches the GPU:
* real(C_DOUBLE) -> Double precision
* real(C_FLOAT)  -> Single precision
* integer(C_INT32_T) -> 4-byte integer

---
### 4. Resource Management (Crucial)
The solver uses a persistent state on the GPU. Failing to release this state before the Fortran program terminates will result in a `SIGABRT` or a CUDA driver error.

* **The Rule:** Always call the `finalize` routine corresponding to your data type before the end of your program.

| Routine                         | Purpose                   |
|:--------------------------------|:--------------------------|
| `finalize_immersed_eq_*()` | Cleans up.  No arguments. |


---

### 5. Code Example
```fortran
program test_solver
    use iso_c_binding
    use fortranbindings_mod
    implicit none

    ! ... (setup code) ...
    
    ! 1. Initialize 
    call init_immersed_eq_d_i32(...)

    ! 2. Solve Loop
    do i = 1, 100
        call solve_immersed_eq_d_i32(...)
    end do

    ! 3. Finalize - Do not skip this!
    print *, "Cleaning up GPU resources..."
    call finalize_immersed_eq_d_i32()
end program test_solver
```
## 6. Argument Reference

### Initialization Routine (`initImmersedEq_*`)
This routine allocates GPU memory and pre-computes the Eigen Decomposition.

| Argument | Type | Description                                |
| :--- | :--- |:-------------------------------------------|
| height | integer(C_SIZE_T) | Grid height (Y-dimension).                 |
| width | integer(C_SIZE_T) | Grid width (X-dimension).                  |
| depth | integer(C_SIZE_T) | Grid depth (Z-dimension).                  |
| nnz | integer(C_SIZE_T) | Max non-zeros allowed in matrix B.         |
| p | real array | Pressure vector (Size: H*W*D).             |
| f | real array | Force vector. (Size heightB)               |
| dx, dy, dz | real(C_DOUBLE) | Physical grid spacing.                     |
| tolerance | real(C_DOUBLE) | Solver convergence threshold (e.g., 1e-6). |
| maxIter | integer(C_SIZE_T) | Max iterations for the BiCGSTAB solver.    |

---

### Solve Routine (`solveImmersedEq_*`)
This routine executes the iterative solver for a specific state of CSR matrix B
or CSC matrix B^T.  If B^T is passed as a CSC, use colOffsetsBT instead of rowOffsets B
and rowIndsBT instead of colIndsB.

| Argument    | Type | Description                                     |
|:------------| :--- |:------------------------------------------------|
| result      | real array | Output: Array to overwritten by the value of x. |
| nnzB        | integer(C_SIZE_T) | Current non-zero count in matrix B.             |
| offsetsB    | integer array | Sparse row offsets (MUST BE 0-BASED).           |
| indsB       | integer array | Sparse column indices (MUST BE 0-BASED).        |
| valuesB     | real array | Non-zero values for matrix B for this step.     |
| multiStream | logical | .true. to run solver in parallel CUDA streams.  |

---

### Coupled Solve Routine (`solveImmersedEqPrimes_*`)
This routine executes the coupled solver, returning results for both the grid domain ($P$) and the boundary interface ($F$).

| Argument    | Type | Description |
|:------------| :--- |:------------------------------------------------|
| resultP     | real array | Output: Grid-based result (Size: H*W*D). |
| resultF     | real array | Output: Boundary-based result (e.g., Force). |
| nnzB        | integer(C_SIZE_T) | Non-zero count for matrix B. |
| offsetsB    | integer array | Row offsets for B (or NULL to reuse previous). |
| indsB       | integer array | Column indices for B (or NULL to reuse previous).|
| valuesB     | real array | Matrix values for B (or NULL to reuse previous). |
| nnzR        | integer(C_SIZE_T) | Non-zero count for matrix R. |
| offsetsR    | integer array | Sparse offsets for R. |
| indsR       | integer array | Sparse indices for R. |
| valuesR     | real array | Matrix values for R. |
| UGamma      | real array | Prescribed boundary velocity/boundary state. |
| uStar       | real array | Staggered intermediate velocity field ($u^*$). |
| multiStream | logical | .true. to run solver in parallel CUDA streams. |

> **Note on Matrix R:** Matrix R can be provided as CSC, or $R^T$ can be provided in CSR format.
>
> **Note on `uStar`:** The input velocity field $u^*$ must be defined on a staggered grid matching the solver's internal discretization.
>
> **Optimization (Reusing B and R):** If any of the arrays in B or R have not changed, pass `C_NULL_PTR` for `offsetsB`, `indsB`, or `valuesB` to skip redundant GPU memory transfers.

## 7. Compiling & Linking

To create your executable, you must link the C++ library and the CUDA runtimes:

1. Compile your Fortran source:
   gfortran -c main.f90

2. Link everything:
   gfortran main.o -L./build -lCudaBandedLib -lstdc++ -lcudart -o ibm_solver

* -lCudaBandedLib: Your newly built library.
* -lstdc++: Required for C++ compatibility.
* -lcudart: The CUDA Runtime library.

See /FortranTest for a tested example Fortran project.