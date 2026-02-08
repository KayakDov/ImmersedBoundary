program test_immersed_eq
    use iso_c_binding
    use fortranbindings_mod
    implicit none

    integer(C_SIZE_T), parameter :: height = 3, width = 2, depth = 1
    integer(C_SIZE_T), parameter :: total_size = height * width * depth
    real(C_DOUBLE) :: p(total_size), f(2), result(total_size)

    integer(C_SIZE_T), parameter :: nnzB = 1
    integer(C_INT32_T)           :: rowOffsetsB(total_size + 1)
    integer(C_INT32_T)           :: colIndsB(nnzB)
    real(C_DOUBLE)               :: valuesB(nnzB)
    integer(C_SIZE_T)            :: i

    ! 1. Data Setup
    f = [1.0_C_DOUBLE, 2.0_C_DOUBLE]
    p = 0.0_C_DOUBLE
    p(1) = 2.0_C_DOUBLE
    p(total_size) = -2.0_C_DOUBLE

    ! 2. Sparse Matrix B (CSR)
    valuesB(1) = 1.0_C_DOUBLE
    colIndsB(1) = 0
    rowOffsetsB(1) = 0
    do i = 2, total_size + 1
        rowOffsetsB(i) = 1
    end do

    ! 3. Call Initialization
    print *, "Initializing Immersed Equation..."
    call init_immersed_eq_d_i32( &
            height, width, depth, &
            nnzB, &
            p, f, &
            1.0_C_DOUBLE, 1.0_C_DOUBLE, 1.0_C_DOUBLE, & ! dx, dy, dz
            1.0_C_DOUBLE, &                             ! dt
            1.0e-6_C_DOUBLE, &                          ! tolerance (fixed typo)
            3_C_SIZE_T &                                ! maxIter
            )

    ! 4. Solve
    print *, "Solving..."
    call solve_immersed_eq_d_i32(result, nnzB, rowOffsetsB, colIndsB, valuesB)

    ! 5. Results
    print *, "Result:"
    print '(12F6.2)', result

    ! 6. The "Magic" Fix: Cleanup
    print *, "Finalizing"
    call finalize_immersed_eq_d_i32()

end program test_immersed_eq