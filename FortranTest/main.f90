program test_bicgstab_device
    use iso_c_binding
    use cudafor                         ! PGI/NVHPC CUDA Fortran module
    use fortranbindings_mod             ! <--- ADDED: Use the Shroud module
    implicit none

    integer(C_SIZE_T), parameter :: n = 3
    integer(C_SIZE_T), parameter :: numDiags = 3

    real(c_double), device, allocatable, target :: A_d(:,:)
    integer(c_int32_t), device, allocatable, target :: inds_d(:)
    real(c_double), device, allocatable, target :: b_d(:)
    real(c_double), device, allocatable, target :: workspace_d(:,:)

    integer :: istat

    integer(c_size_t) :: A_addr, inds_addr, b_addr, work_addr

    ! Leading dimensions
    integer(c_size_t), parameter :: aLd = n
    integer(c_size_t), parameter :: bStride = 1
    integer(c_size_t), parameter :: indsStride = 1
    integer(c_size_t), parameter :: workLd = n

    integer(c_size_t), parameter :: maxIter = 1000
    real(c_double),   parameter :: tol = 1d-8

    real(c_double) :: b_h(n)
    ! -------------------------------
    ! Allocate device memory
    ! -------------------------------
    allocate(A_d(n, numDiags))
    allocate(inds_d(numDiags))
    allocate(b_d(n))
    allocate(workspace_d(n, 7))

    ! -------------------------------
    ! Initialize test values on host
    ! -------------------------------
    A_d = reshape([1d0, 3d0, 5d0,   &
                   2d0, 5d0, 6d0,   &
                   1d0, 2d0, 3d0],  &
            shape(A_d))

    inds_d = [1_c_int, -1_c_int, 0_c_int]

    b_d = [4d0, 12d0, 9d0]

    ! -------------------------------
    ! Get device pointer addresses
    ! -------------------------------

    A_addr    = transfer(c_loc(A_d), A_addr)
    inds_addr = transfer(c_loc(inds_d), inds_addr)
    b_addr    = transfer(c_loc(b_d), b_addr)
    work_addr = transfer(c_loc(workspace_d), work_addr)

    ! -------------------------------
    ! Call the C++ BiCGSTAB solver
    ! -------------------------------
    call solve_bi_cgstab( &
            A_addr, aLd, &
            inds_addr, indsStride, numDiags, &
            b_addr, bStride, n, &
            work_addr, workLd, &
            maxIter, tol )

    ! -------------------------------
    ! Retrieve solution
    ! -------------------------------

    b_h = b_d   ! copy from device to host
    print *, "Solution x (from device):"
    print *, b_h


end program test_bicgstab_device