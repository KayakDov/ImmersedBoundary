program test_cudabanded_suite
    use iso_c_binding
    use eigenbcgsolver_imeq_mod
    use eigenbcgsolver_eigen_mod
    implicit none

    !------------------------------------------------------------------
    ! Grid parameters
    !------------------------------------------------------------------
    integer(C_SIZE_T) :: rows, cols, layers, n_cells

    real(C_DOUBLE), allocatable :: dx(:), dy(:), dz(:)
    real(C_DOUBLE) :: dt

    logical :: uniformDeltaX
    logical :: uniformDeltaY
    logical :: uniformDeltaZ

    !------------------------------------------------------------------
    ! Boundary conditions
    !------------------------------------------------------------------
    logical :: leftIsNeumann, rightIsNeumann
    logical :: topIsNeumann, bottomIsNeumann
    logical :: backIsNeumann, frontIsNeumann

    real(C_DOUBLE) :: leftVal, rightVal
    real(C_DOUBLE) :: topVal, bottomVal
    real(C_DOUBLE) :: frontVal, backVal

    logical :: isStaggered

    !------------------------------------------------------------------
    ! Immersed solver
    !------------------------------------------------------------------
    real(C_DOUBLE), allocatable :: p_im(:)
    real(C_DOUBLE), allocatable :: f_constraints(:)
    real(C_DOUBLE), allocatable :: rhs_b(:)

    integer(C_INT32_T), allocatable :: rowOffsetsB(:)
    integer(C_INT32_T), allocatable :: colIndsB(:)
    real(C_DOUBLE), allocatable :: valuesB(:)

    integer(C_SIZE_T) :: nnzB
    integer(C_SIZE_T) :: nConstraints

    !------------------------------------------------------------------
    ! Eigen solver
    !------------------------------------------------------------------
    real(C_DOUBLE), allocatable :: x_eig(:)
    real(C_DOUBLE), allocatable :: b_eig(:)

    logical :: use_thomas

    !------------------------------------------------------------------
    ! Grid
    !------------------------------------------------------------------
    rows   = 3_C_SIZE_T
    cols   = 4_C_SIZE_T
    layers = 2_C_SIZE_T

    n_cells = rows * cols * layers

    allocate(dx(rows))
    allocate(dy(cols))
    allocate(dz(layers))

    dx = 1.0_C_DOUBLE
    dy = 0.5_C_DOUBLE
    dz = 2.0_C_DOUBLE

    dt = 1.0_C_DOUBLE

    uniformDeltaX = .true.
    uniformDeltaY = .true.
    uniformDeltaZ = .true.

    !------------------------------------------------------------------
    ! Boundary conditions
    !------------------------------------------------------------------
    leftIsNeumann   = .false.
    rightIsNeumann  = .false.
    topIsNeumann    = .false.
    bottomIsNeumann = .false.
    backIsNeumann   = .false.
    frontIsNeumann  = .false.

    leftVal   = 0.0_C_DOUBLE
    rightVal  = 0.0_C_DOUBLE
    topVal    = 0.0_C_DOUBLE
    bottomVal = 0.0_C_DOUBLE
    frontVal  = 0.0_C_DOUBLE
    backVal   = 0.0_C_DOUBLE

    isStaggered = .false.

    print *, "==========================================="
    print *, "STARTING TEST 1: IMMERSED BOUNDARY SOLVER"
    print *, "==========================================="

    nConstraints = 2_C_SIZE_T
    nnzB = 2_C_SIZE_T

    allocate(rowOffsetsB(nConstraints+1))
    allocate(colIndsB(nnzB))
    allocate(valuesB(nnzB))

    allocate(f_constraints(nConstraints))
    allocate(p_im(n_cells))
    allocate(rhs_b(n_cells))

    rowOffsetsB = [0_C_INT32_T,1_C_INT32_T,2_C_INT32_T]
    colIndsB    = [0_C_INT32_T,1_C_INT32_T]
    valuesB     = [1.0_C_DOUBLE,1.0_C_DOUBLE]

    f_constraints = [1.0_C_DOUBLE,2.0_C_DOUBLE]

    rhs_b = 0.0_C_DOUBLE
    rhs_b(1) = 10.0_C_DOUBLE

    p_im = 0.0_C_DOUBLE

    call init_immersed_eq_d_i32( &
            rows, cols, layers, &
            leftIsNeumann, rightIsNeumann, topIsNeumann, &
            bottomIsNeumann, backIsNeumann, frontIsNeumann, &
            leftVal, rightVal, topVal, bottomVal, frontVal, backVal, &
            isStaggered, &
            nnzB, &
            p_im, &
            f_constraints, &
            dx, dy, dz, &
            dt, &
            uniformDeltaX, uniformDeltaY, uniformDeltaZ, &
            1.0e-8_C_DOUBLE, &
            1000_C_SIZE_T)

    call solve_immersed_eq_d_i32( &
            p_im, &
            nnzB, &
            rowOffsetsB, &
            colIndsB, &
            valuesB)

    print *, "Immersed Eq Result (first 3): ", p_im(1:3)

    call finalize_immersed_eq_d_i32()

    deallocate(rowOffsetsB,colIndsB,valuesB)
    deallocate(f_constraints,p_im,rhs_b)

    print *
    print *, "==========================================="
    print *, "STARTING TEST 2: THOMAS EIGEN DECOMPOSITION"
    print *, "==========================================="

    allocate(x_eig(n_cells))
    allocate(b_eig(n_cells))

    x_eig = 0.0_C_DOUBLE
    b_eig = 1.0_C_DOUBLE

    use_thomas = .true.

    call init_eigen_decomp_d( &
            rows, cols, layers, &
            dx, dy, dz, &
            uniformDeltaX, &
            uniformDeltaY, &
            uniformDeltaZ, &
            leftIsNeumann, &
            rightIsNeumann, &
            topIsNeumann, &
            bottomIsNeumann, &
            backIsNeumann, &
            frontIsNeumann, &
            leftVal, &
            rightVal, &
            topVal, &
            bottomVal, &
            frontVal, &
            backVal, &
            isStaggered, &
            use_thomas)

    call solve_eigen_decomp_d(x_eig,b_eig)

    print *, "Eigen Decomp Result (first 3): ", x_eig(1:3)

    call finalize_eigen_decomp_d()

    deallocate(x_eig,b_eig)
    deallocate(dx,dy,dz)

    print *, "==========================================="
    print *, "ALL TESTS COMPLETED"
    print *, "==========================================="

end program test_cudabanded_suite