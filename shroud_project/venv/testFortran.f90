program test_immersed_ib
    use fortranbindings_mod
    use iso_c_binding, only : C_DOUBLE, C_INT32_T, C_SIZE_T
    implicit none

    ! Parameters matching the C++ test
    integer(C_SIZE_T), parameter :: height = 3
    integer(C_SIZE_T), parameter :: width  = 2
    integer(C_SIZE_T), parameter :: depth  = 2
    integer(C_SIZE_T), parameter :: grid_size = height * width * depth ! 12
    integer(C_SIZE_T), parameter :: nnzMaxB = 1

    ! Data Arrays
    real(C_DOUBLE) :: f(2)
    real(C_DOUBLE) :: p(grid_size)
    real(C_DOUBLE) :: values(1)
    real(C_DOUBLE) :: result(grid_size)

    ! CSR Indices (Note: values must match C++ logic for the solver)
    integer(C_INT32_T) :: rowPointers(1)
    integer(C_INT32_T) :: colOffsets(grid_size + 1)

    integer :: i

    ! 1. Initialize data
    f = [1.0_C_DOUBLE, 2.0_C_DOUBLE]

    p = 0.0_C_DOUBLE
    p(1) = -2.0_C_DOUBLE
    p(grid_size) = 2.0_C_DOUBLE

    values(1) = 1.0_C_DOUBLE
    rowPointers(1) = 0

    colOffsets(1) = 0
    do i = 2, grid_size + 1
        colOffsets(i) = 1
    end do

    print *, "Initializing Immersed Boundary Solver..."

    ! 2. Call the wrapper (The generic interface handles the suffix)
    ! Arguments: height, width, depth, nnzMaxB, p, f, dx, dy, dz, tol, maxIter
    call init_immersed_eq(height, width, depth, nnzMaxB, p, f, &
            1.0_C_DOUBLE, 1.0_C_DOUBLE, 1.0_C_DOUBLE, &
            1e-6_C_DOUBLE, 3_C_SIZE_T)

    print *, "Solving system..."

    ! 3. Solve
    ! Arguments: result, nnzB, rowPointersB, colPointersB, valuesB, multiStream
    call solve_immersed_eq(result, nnzMaxB, rowPointers, colOffsets, values, .true.)

    ! 4. Output results
    print *, "Solution Result:"
    do i = 1, grid_size
        write(*, '(F8.4, " ")', advance='no') result(i)
    end do
    print *

end program test_immersed_ib