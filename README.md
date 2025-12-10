# **Fortran Interface: Guide to the Poisson Eigen Decomposition Solver**

This file contains instructions for Fortran programmers on how to build the necessary C++/CUDA library call the Fortran bindings.

## **Building the C++/CUDA Solver Library (CMake)**

The Fortran bindings require that the core C++/CUDA solver library is built first using CMake.

From your project root directory, follow these steps:

1. **Navigate to the Build Directory**  

2. **Build the EigenDecomp Target**  The Fortran bindings rely on the `CudaBandedLib` library, which contains the core C++/CUDA solver logic as well as the Fortran wrapper.

   From the project root directory:

   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build . -j$(nproc)

3. This will build both the **static library** `libCudaBandedLib.a.

4. The library and executable are located in the build directory (e.g., `cmake-build-debug`).

## **Using the Fortran Module**

The interface is contained in the `fortranbindings_mod` module (from `wrapffortranbindings.f90`).

To use the solver in your Fortran program:

```
program my_fortran_app
    use fortranbindings_mod
    implicit none

    ! Declare and initialize your arrays
    ! Call the solver subroutines
end program my_fortran_app
```

### Available Subroutines

- The module exposes **four main subroutines**, two for BiCGSTAB solving and two for decomposition:

  | Subroutine               | Precision                 | Description                                                  |
  | ------------------------ | ------------------------- | ------------------------------------------------------------ |
  | `solve_bi_cgstab_float`  | single (`real(C_FLOAT)`)  | Calls the BiCGSTAB solver with single-precision arrays.      |
  | `solve_bi_cgstab_double` | double (`real(C_DOUBLE)`) | Calls the BiCGSTAB solver with double-precision arrays.      |
  | `solve_decomp_float`     | single (`real(C_FLOAT)`)  | Calls the 3D Poisson eigen decomposition solver with single-precision arrays. |
  | `solve_decomp_double`    | double (`real(C_DOUBLE)`) | Calls the 3D Poisson eigen decomposition solver with double-precision arrays. |

# **3D Poisson Solver: Eigen Decomposition Summary**

This solver uses an eigen decomposition approach to quickly solve the discrete 3D Poisson equation: L u \= f.

## **Steps**

1. **Initialization and Input**
   * **Inputs:** Boundary coefficients (frontBack, leftRight, topBottom) defining the discretized Laplacian matrix (L), and the Right-Hand Side vector (f).
2. **Spectral Decomposition**
   * The solver implicitly uses the pre-computed eigenvalues (lambda) and eigenmatrices (Phi) corresponding to the 1D discrete Laplacian for each grid dimension.
3. **Forward Transformation (3D FFT-like)**
   * The Right-Hand Side vector (f) is transformed into the eigen-space (frequency domain) by successive application of the transposed eigenmatrices across each dimension (Depth, Width, and Height).
   * **Operation:** The transformed vector (f\_tilde) is calculated as:  
     f\_tilde \= Phi\_Depth \* Phi\_Width \* Phi\_Height \* f

4. **Decoupled Solution**
   * The system is solved algebraically in the eigen-space where the matrix is diagonal, yielding the transformed solution (u\_tilde).
   * **Operation:** The calculation is an element-wise division:  
     u\_tilde\[i,j,k\] \= f\_tilde\[i,j,k\] / lambda\[i,j,k\]

5. **Inverse Transformation**
   * The final solution u is recovered by applying the inverse transformation to the decoupled solution (u\_tilde).
   * **Operation:** This is the reverse application of the eigenmatrices:  
     u \= Inverse(Phi\_Depth \* Phi\_Width \* Phi\_Height) \* u\_tilde

6. **Output**
   * The final solution vector x (which is u), mapped to the H x W x D grid.

### **Solver Subroutines and Data Types**

Two subroutines are available, one for single precision and one for double precision:

* **Single Precision:** eigenDecompSolver\_float (uses real(c\_float))
* **Double Precision:** eigenDecompSolver\_double (uses real(c\_double))

### **Argument Details**

The subroutines accept ten arguments. Note that all dimension and stride arguments are passed **by value** (value) using the C size type (integer(c\_size\_t)).

Note, all data is column major. Two-dimensional matrices have a distance of ld between the first elements of each column, with ld - height padding at the end of each column.  Three-dimensional data (row, column, depth) is stored stored with row changing fastest, then depth, then column.  So when flattened, the first height elements are the first column in the first layer, the next height elements are the first column in the second layer, and after all the first columns in all the layers we have the second columns in each layer in turn, and so on.
The boundary matrices frontBack, leftRight, and topBottom are stored as three-dimensional matrices, each with two layers.  The first layer is the front/left/top and the second layer back/right/bottom.
The first row of the front and back boundary matrices is up against the top.  The first column of the front and back matrices is up against the left boundary.
The first row of the left and right matrices is up against the top.  The first column of the left and right matrices is up against the back.
The first row the top and bottom matrices is up against the back boundary.  The first column up against the left boundary.

## 3. Data Layout and Argument Conventions

### BiCGSTAB Solver (`solve_bi_cgstab_*`)

Arguments:

| Argument Name     | Type              | Description                               |
| ----------------- | ----------------- | ----------------------------------------- |
| A                 | real array        | Matrix for the linear system (in/out).    |
| aLd               | integer(C_SIZE_T) | Leading dimension of `A`.                 |
| inds              | integer array     | Index array (in/out).                     |
| indsStride        | integer(C_SIZE_T) | Stride for `inds`.                        |
| numInds           | integer(C_SIZE_T) | Number of indices.                        |
| b                 | real array        | Right-hand side vector (in/out).          |
| bStride           | integer(C_SIZE_T) | Stride of `b`.                            |
| bSize             | integer(C_SIZE_T) | Size of `b`.                              |
| prealocatedSizeX7 | real array        | Scratch space of size N×7 (in/out).       |
| prealocatedLd     | integer(C_SIZE_T) | Leading dimension of `prealocatedSizeX7`. |
| maxIterations     | integer(C_SIZE_T) | Maximum number of BiCGSTAB iterations.    |
| tolerance         | real              | Convergence tolerance.                    |

### Decomposition Solver (`solve_decomp_*`)

Arguments:

| Argument Name                                         | Type              | Description                                                  |
| ----------------------------------------------------- | ----------------- | ------------------------------------------------------------ |
| frontBack, leftRight, topBottom                       | real array        | Boundary matrices of the 3D grid. Each has **two layers**: front/back, left/right, top/bottom. |
| fbLd, lrLd, tbLd                                      | integer(C_SIZE_T) | Leading dimensions for the corresponding boundary arrays.    |
| f                                                     | real array        | Right-hand side vector (in/out).                             |
| fStride                                               | integer(C_SIZE_T) | Stride of `f`.                                               |
| x                                                     | real array        | Output solution vector.                                      |
| xStride                                               | integer(C_SIZE_T) | Stride of `x`.                                               |
| height, width, depth                                  | integer(C_SIZE_T) | Dimensions of the 3D grid.                                   |
| rowsXRows, colsXCols, depthsXDepths, maxDimX3         | real arrays       | Precomputed eigenmatrices for the 3D decomposition.          |
| rowsXRowsLd, colsXColsLd, depthsXDepthsLd, maxDimX3Ld | integer(C_SIZE_T) | Leading dimensions of the corresponding eigenmatrices.       |

## **Linking Your Fortran Program**

The final step is linking your compiled Fortran code against the C++/CUDA library.

1. **Compile Fortran:**  
   gfortran \-c src/eigendecomp\_interface.f90  
   gfortran \-c your\_main\_program.f90

2. Link Everything Together:  
   You must link the Fortran objects with the C++ library output and required runtimes (stdc++ and cudart).  
   gfortran your\_main\_program.o eigendecomp\_interface.o \\  
   \-L\~/Documents/TelAvivU/cmake-build-debug \-lBiCGSTAB\_LIB \\  
   \-o final\_solver\_app \\  
   \-Wl,-rpath,\~/Documents/TelAvivU/cmake-build-debug \\  
   \-lstdc++ \-lcudart

    * **\-L**: Path to the required library.
    * **\-lBiCGSTAB\_LIB**: Links the dependency library (e.g., libBiCGSTAB\_LIB.a).
    * **\-lstdc++ \-lcudart**: Links C++ and CUDA runtime libraries.

This should result in a runnable executable.