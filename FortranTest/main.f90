program fortran_app
    use iso_c_binding
    use eigenbcgsolver_imeq_mod
    implicit none

    ! Dimensions
    integer(C_SIZE_T) :: gridHeight, gridWidth, gridDepth
    real(C_DOUBLE) :: deltaX, deltaY, deltaZ, dt, tolerance
    integer(C_SIZE_T) :: maxBCGIterations

    ! Field arrays
    real(C_DOUBLE), allocatable :: p(:), f(:)

    ! CSR sparse matrix arrays for solver
    integer(C_SIZE_T) :: nnzB
    integer(C_INT32_T), allocatable :: rowOffsetsB(:), colIndsB(:)
    real(C_DOUBLE), allocatable :: valuesB(:)

    ! Example: initialize problem sizes
    gridHeight = 10_C_SIZE_T
    gridWidth  = 10_C_SIZE_T
    gridDepth  = 10_C_SIZE_T
    deltaX = 0.1_C_DOUBLE
    deltaY = 0.1_C_DOUBLE
    deltaZ = 0.1_C_DOUBLE
    dt     = 0.01_C_DOUBLE
    tolerance = 1.0e-6_C_DOUBLE
    maxBCGIterations = 100_C_SIZE_T

    ! Allocate fields
    allocate(p(gridHeight*gridWidth*gridDepth))
    allocate(f(gridHeight*gridWidth*gridDepth))

    ! Fill p and f with initial data
    p = 0.0_C_DOUBLE
    f = 1.0_C_DOUBLE

    ! Initialize the immersed equation solver
    call init_immersed_eq_d_i32(gridHeight, gridWidth, gridDepth, &
            gridHeight*gridWidth*gridDepth*5_C_SIZE_T, p, f, &
            deltaX, deltaY, deltaZ, dt, tolerance, maxBCGIterations)

    ! --- Prepare dummy CSR matrix (replace with your real data) ---
    nnzB = 0_C_SIZE_T
    allocate(rowOffsetsB(1))
    allocate(colIndsB(1))
    allocate(valuesB(1))
    ! TODO: fill nnzB, rowOffsetsB, colIndsB, valuesB with your sparse matrix

    ! Solve the immersed equation
    call solve_immersed_eq_d_i32(p, nnzB, rowOffsetsB, colIndsB, valuesB)

    ! Finalize solver
    call finalize_immersed_eq_d_i32()

    ! Print first few values of solution
    print *, "First 10 entries of solution p:"
    print *, p(1:min(10,size(p)))

    ! Deallocate arrays
    deallocate(p, f, rowOffsetsB, colIndsB, valuesB)

end program fortran_app