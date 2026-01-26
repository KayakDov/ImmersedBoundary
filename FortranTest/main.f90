program test_immersed_eq
    use iso_c_binding
    use fortranbindings_mod
    implicit none


    ! Parameters matching your C++ test
    integer(C_SIZE_T), parameter :: height = 3
    integer(C_SIZE_T), parameter :: width  = 2
    integer(C_SIZE_T), parameter :: depth  = 2
    integer(C_SIZE_T), parameter :: total_size = height * width * depth ! 12

    ! Data arrays
    real(C_DOUBLE) :: p(total_size)
    real(C_DOUBLE) :: f(2)
    real(C_DOUBLE) :: result(total_size)

    ! Sparse Matrix data (CSR format)
    integer(C_SIZE_T), parameter :: nnz = 1
    integer(C_INT32_T) :: rowPointers(nnz)
    integer(C_INT32_T) :: colOffsets(total_size + 1)
    real(C_DOUBLE)     :: values(nnz)

    integer(C_SIZE_T) :: i

    integer :: ierr
    ! ...
    print *, "Activating GPU..."
    ierr = cudaFree(0) ! Standard trick to initialize CUDA context

    ! 1. Initialize data matching your C++ vectors
    f = [1.0_C_DOUBLE, 2.0_C_DOUBLE]

    ! p = {-2, 0, ..., 0, 2}
    p = 0.0_C_DOUBLE
    p(1) = -2.0_C_DOUBLE
    p(total_size) = 2.0_C_DOUBLE

    ! 2. Initialize Sparse Matrix (matching C++ logic)
    ! Note: We use 0 and 1 explicitly because the C++ solver expects 0-based indices
    values(1) = 1.0_C_DOUBLE
    rowPointers(1) = 0

    colOffsets(1) = 0
    do i = 2, total_size + 1
        colOffsets(i) = 1
    end do

    ! 3. Call the Generated Wrapper
    ! Using the generic name for init
    print *, "Initializing Immersed Equation..."
    call init_immersed_eq_d_i32( &
            height, width, depth, &
            nnz, &
            p, f, &
            1.0_C_DOUBLE, 1.0_C_DOUBLE, 1.0_C_DOUBLE, &
            1e-6_C_DOUBLE, &
            3_C_SIZE_T &
            )

    ! 4. Solve
    ! Using the specific name as we removed generic for solve
    print *, "Solving..."
    call solve_immersed_eq_d_i32( &
            result, &
            nnz, &
            rowPointers, &
            colOffsets, &
            values, &
            .true. &               ! multiStream
            )

    ! 5. Print Results
    print *, "Result:"
    do i = 1, total_size
        write(*, '(F6.2, " ")', advance='no') result(i)
    end do
    print *

end program test_immersed_eq