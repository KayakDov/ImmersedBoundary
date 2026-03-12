program test_cudabanded_suite
    use iso_c_binding
    use eigenbcgsolver_imeq_mod
    use eigenbcgsolver_eigen_mod
    implicit none

    ! Common parameters
    real(C_DOUBLE) :: dx, dy, dz
    integer(C_SIZE_T) :: rows, cols, layers, n_cells
    integer(C_SIZE_T) :: i

    ! Immersed Eq specific
    real(C_DOUBLE), allocatable :: p_im(:), f_constraints(:), rhs_b(:)
    integer(C_INT32_T), allocatable :: rowOffsetsB(:), colIndsB(:)
    real(C_DOUBLE), allocatable :: valuesB(:)
    integer(C_SIZE_T) :: nnzB, nConstraints

    ! EigenDecomp specific
    real(C_DOUBLE), allocatable :: x_eig(:), b_eig(:)
    logical :: use_thomas

    ! Initialize grid dimensions
    rows = 3_C_SIZE_T
    cols = 4_C_SIZE_T
    layers = 2_C_SIZE_T
    n_cells = rows * cols * layers
    dx = 1.0_C_DOUBLE
    dy = 0.5_C_DOUBLE
    dz = 2.0_C_DOUBLE

    print *, "==========================================="
    print *, "STARTING TEST 1: IMMERSED BOUNDARY SOLVER"
    print *, "==========================================="

    nConstraints = 2_C_SIZE_T
    nnzB = 2_C_SIZE_T
    allocate(rowOffsetsB(nConstraints + 1), colIndsB(nnzB), valuesB(nnzB))
    allocate(f_constraints(nConstraints), p_im(n_cells), rhs_b(n_cells))

    rowOffsetsB = [0_C_INT32_T, 1_C_INT32_T, 2_C_INT32_T]
    colIndsB    = [0_C_INT32_T, 1_C_INT32_T]
    valuesB     = [1.0_C_DOUBLE, 1.0_C_DOUBLE]
    f_constraints = [1.0_C_DOUBLE, 2.0_C_DOUBLE]
    rhs_b = 0.0_C_DOUBLE
    rhs_b(1) = 10.0_C_DOUBLE ! Sample RHS value
    p_im = 0.0_C_DOUBLE

    call init_immersed_eq_d_i32(rows, cols, layers, nnzB, rhs_b, f_constraints, &
            dx, dy, dz, 1.0_C_DOUBLE, 1.0e-8_C_DOUBLE, 1000_C_SIZE_T)

    call solve_immersed_eq_d_i32(p_im, nnzB, rowOffsetsB, colIndsB, valuesB)

    print *, "Immersed Eq Result (first 3):", p_im(1:3)
    call finalize_immersed_eq_d_i32()
    deallocate(rowOffsetsB, colIndsB, valuesB, f_constraints, p_im, rhs_b)

    print *, ""
    print *, "==========================================="
    print *, "STARTING TEST 2: THOMAS EIGEN DECOMPOSITION"
    print *, "==========================================="

    allocate(x_eig(n_cells), b_eig(n_cells))
    b_eig = 1.0_C_DOUBLE ! Simple test: Solve Lap(x) = 1
    x_eig = 0.0_C_DOUBLE
    use_thomas = .true.

    call init_eigen_decomp_d(rows, cols, layers, dx, dy, dz, use_thomas)
    call solve_eigen_decomp_d(x_eig, b_eig)

    print *, "Eigen Decomp Result (first 3):", x_eig(1:3)
    call finalize_eigen_decomp_d()
    deallocate(x_eig, b_eig)

    print *, "==========================================="
    print *, "ALL TESTS COMPLETED"
    print *, "==========================================="

end program test_cudabanded_suite