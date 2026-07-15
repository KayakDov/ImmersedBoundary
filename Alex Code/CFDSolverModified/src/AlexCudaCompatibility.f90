module AlexCudaCompatibility
    use iso_c_binding, only : C_SIZE_T
    implicit none

    public :: TemperatureHandle, VxHandle, VyHandle, VzHandle, PressureHandle, PotentialHandle
    public :: Initialize_GPU_Solvers, GrPr

    integer(C_SIZE_T) :: TemperatureHandle = 0_C_SIZE_T
    integer(C_SIZE_T) :: VxHandle          = 0_C_SIZE_T
    integer(C_SIZE_T) :: VyHandle          = 0_C_SIZE_T
    integer(C_SIZE_T) :: VzHandle          = 0_C_SIZE_T
    integer(C_SIZE_T) :: PressureHandle    = 0_C_SIZE_T
    integer(C_SIZE_T) :: PotentialHandle   = 0_C_SIZE_T

    ! Temperature diffusivity scale of the CPU operator (Prandtl/DGr, or 1 when
    ! Prandtl == 0).  Set in Initialize_GPU_Solvers; time_step_Q2D.f90 uses it
    ! to scale the temperature RHS to match the shift convention below.
    real(kind=8) :: GrPr = 1.d0

contains

    Subroutine Initialize_GPU_Solvers()
        Use eigenbcgsolver_eigen_mod, only : init_eigen_decomp_d
        Use iso_c_binding, only : C_SIZE_T
        Use Numbers      ! Nx, Nx1, Ny, Ny1, Nz, Nz1
        Use Parameters   ! Prandtl, DGr
        Use Grid         ! Hx12, Hy12, Hz12, HPx, HPy, HPz
        Use Numerica     ! EVD_BCx/y/z, EVD_Pot_X/Y/Z, Ckor, Htime, Istat
        Implicit None

        real(kind=8) :: shiftTemperature, shiftVelocity
        logical :: tX, tY, tZ    ! temperature: Neumann on both ends of axis?
        logical :: pX, pY, pZ    ! potential:   Neumann on both ends of axis?

        ! ==================================================================
        ! SPACING ARRAYS.  For an axis with n unknowns, CudaBandedLib's
        ! VariableSegment expects n+1 deltas = distances between ADJACENT
        ! UNKNOWNS, where delta(0) is wall-to-first-unknown and delta(n) is
        ! last-unknown-to-wall.  Each axis of this code has two node
        ! families, so the correct array depends on the field:
        !
        !   cell centres X12(1..Nx1):  n = Nx1 unknowns -> deltas = HPx(0:Nx1)
        !       (Tmpr and Prs on all axes; the two tangential axes of each
        !        velocity component; the Potential y-axis)
        !   grid nodes   X(1..Nx):     n = Nx  unknowns -> deltas = Hx12(0:Nx)
        !       (the staggered axis of each velocity: Vx-x, Vy-y, Vz-z;
        !        the Potential x- and z-axes)
        !
        ! NEVER pass Hx12(0:Nx1) / Hy12(0:Ny1) / Hz12(0:Nz1): MeshStretch only
        ! fills Hx12(0:Nx), so the final element of those slices is 0.0 and a
        ! zero spacing gives a division by zero (NaN eigendecomposition).
        !
        ! isStaggered is ignored by the library on the variable-delta path,
        ! so it is set to .false. everywhere.
        ! ==================================================================

        ! ==================================================================
        ! HELMHOLTZ SHIFTS.  The CPU code does NOT invert plain Laplacians
        ! for temperature and velocity; EVD_Thomas applies
        !     pdum = -(Ckor/Htime)*Dtm + lambda_y + lambda_z
        ! i.e. it solves (alpha*L - (Ckor/Htime)*Dtm) x = rhs, with
        !     temperature: alpha = 1/GrPr, Dtm = Istat
        !     velocities : alpha = DGr,    Dtm = 1
        ! We solve the equivalent system  (L + shift) x = rhs / alpha :
        !     shift(temperature) = -Ckor*Istat*GrPr/Htime, rhs scaled by GrPr
        !     shift(velocity)    = -Ckor/(Htime*DGr),      rhs scaled by 1/DGr
        ! The rhs scaling lives in time_step_Q2D.f90 where the GPU staging
        ! arrays are filled.  Pressure and Potential are pure Poisson
        ! (shift = 0, no scaling).
        ! ==================================================================
        GrPr = Prandtl / DGr
        If (Prandtl == 0.D0) GrPr = 1.D0

        shiftTemperature =   Ckor * Dble(Istat) * GrPr / Htime   ! positive: solver computes (L - shift*I)
        shiftVelocity    =   Ckor / ( Htime * DGr )

        ! ==================================================================
        ! BOUNDARY CONDITIONS, mirroring the CPU operators exactly:
        !   Temperature (EVD_laptmpr): Neumann on both ends of an axis when
        !                              the runtime flag EVD_BC? == 1
        !   Velocities (EVD_lapVx/y/z): Dirichlet everywhere, unconditionally
        !   Pressure (EVD_lapP):        Neumann everywhere, unconditionally
        !   Potential (EVD_lap_Fi):     Neumann per axis when EVD_Pot_? == 1
        ! Axis naming in the library: left/right = x, top/bottom = y,
        ! front/back = z.  All boundary values in this code are homogeneous.
        ! ==================================================================
        tX = (EVD_BCx   == 1);  tY = (EVD_BCy   == 1);  tZ = (EVD_BCz   == 1)
        pX = (EVD_Pot_X == 1);  pY = (EVD_Pot_Y == 1);  pZ = (EVD_Pot_Z == 1)

        ! ---------------- Temperature: (Ny1, Nx1, Nz1), cell-centred -------
        call assert_spacing_sizes("Temperature", HPx(0:Nx1), HPy(0:Ny1), HPz(0:Nz1), Nx1, Ny1, Nz1)
        TemperatureHandle = init_eigen_decomp_d( &
                rows=int(Ny1,C_SIZE_T), cols=int(Nx1,C_SIZE_T), layers=int(Nz1,C_SIZE_T), &
                dx=HPx(0:Nx1), dy=HPy(0:Ny1), dz=HPz(0:Nz1), &
                uniformDeltaX=.false., uniformDeltaY=.false., uniformDeltaZ=.false., &
                leftIsNeumann=tX, rightIsNeumann=tX, &
                topIsNeumann=tY, bottomIsNeumann=tY, &
                frontIsNeumann=tZ, backIsNeumann=tZ, &
                leftVal=0.d0, rightVal=0.d0, topVal=0.d0, bottomVal=0.d0, &
                frontVal=0.d0, backVal=0.d0, isStaggered=.false., &
                helmholtzShift=shiftTemperature, thomas=.true. )

        ! ---------------- Vx: (Ny1, Nx, Nz1), node-centred in x ------------
        call assert_spacing_sizes("Vx", Hx12(0:Nx), HPy(0:Ny1), HPz(0:Nz1), Nx, Ny1, Nz1)
        VxHandle = init_eigen_decomp_d( &
                rows=int(Ny1,C_SIZE_T), cols=int(Nx,C_SIZE_T), layers=int(Nz1,C_SIZE_T), &
                dx=Hx12(0:Nx), dy=HPy(0:Ny1), dz=HPz(0:Nz1), &
                uniformDeltaX=.false., uniformDeltaY=.false., uniformDeltaZ=.false., &
                leftIsNeumann=.false., rightIsNeumann=.false., &
                topIsNeumann=.false., bottomIsNeumann=.false., &
                frontIsNeumann=.false., backIsNeumann=.false., &
                leftVal=0.d0, rightVal=0.d0, topVal=0.d0, bottomVal=0.d0, &
                frontVal=0.d0, backVal=0.d0, isStaggered=.false., &
                helmholtzShift=shiftVelocity, thomas=.true. )

        ! ---------------- Vy: (Ny, Nx1, Nz1), node-centred in y ------------
        call assert_spacing_sizes("Vy", HPx(0:Nx1), Hy12(0:Ny), HPz(0:Nz1), Nx1, Ny, Nz1)
        VyHandle = init_eigen_decomp_d( &
                rows=int(Ny,C_SIZE_T), cols=int(Nx1,C_SIZE_T), layers=int(Nz1,C_SIZE_T), &
                dx=HPx(0:Nx1), dy=Hy12(0:Ny), dz=HPz(0:Nz1), &
                uniformDeltaX=.false., uniformDeltaY=.false., uniformDeltaZ=.false., &
                leftIsNeumann=.false., rightIsNeumann=.false., &
                topIsNeumann=.false., bottomIsNeumann=.false., &
                frontIsNeumann=.false., backIsNeumann=.false., &
                leftVal=0.d0, rightVal=0.d0, topVal=0.d0, bottomVal=0.d0, &
                frontVal=0.d0, backVal=0.d0, isStaggered=.false., &
                helmholtzShift=shiftVelocity, thomas=.true. )

        ! ---------------- Vz: (Ny1, Nx1, Nz), node-centred in z ------------
        call assert_spacing_sizes("Vz", HPx(0:Nx1), HPy(0:Ny1), Hz12(0:Nz), Nx1, Ny1, Nz)
        VzHandle = init_eigen_decomp_d( &
                rows=int(Ny1,C_SIZE_T), cols=int(Nx1,C_SIZE_T), layers=int(Nz,C_SIZE_T), &
                dx=HPx(0:Nx1), dy=HPy(0:Ny1), dz=Hz12(0:Nz), &
                uniformDeltaX=.false., uniformDeltaY=.false., uniformDeltaZ=.false., &
                leftIsNeumann=.false., rightIsNeumann=.false., &
                topIsNeumann=.false., bottomIsNeumann=.false., &
                frontIsNeumann=.false., backIsNeumann=.false., &
                leftVal=0.d0, rightVal=0.d0, topVal=0.d0, bottomVal=0.d0, &
                frontVal=0.d0, backVal=0.d0, isStaggered=.false., &
                helmholtzShift=shiftVelocity, thomas=.true. )

        ! ---------------- Pressure: (Ny1, Nx1, Nz1), ALL NEUMANN -----------
        ! EVD_lapP applies Neumann on every boundary unconditionally.  The
        ! all-Neumann Laplacian is singular; the library's isSingular path
        ! (zeroing the constant spectral mode) handles this, exactly like the
        ! "if (abs(pdum) <= 1e-8) pdum = 1" guard in the CPU solver.
        call assert_spacing_sizes("Pressure", HPx(0:Nx1), HPy(0:Ny1), HPz(0:Nz1), Nx1, Ny1, Nz1)
        PressureHandle = init_eigen_decomp_d( &
                rows=int(Ny1,C_SIZE_T), cols=int(Nx1,C_SIZE_T), layers=int(Nz1,C_SIZE_T), &
                dx=HPx(0:Nx1), dy=HPy(0:Ny1), dz=HPz(0:Nz1), &
                uniformDeltaX=.false., uniformDeltaY=.false., uniformDeltaZ=.false., &
                leftIsNeumann=.true., rightIsNeumann=.true., &
                topIsNeumann=.true., bottomIsNeumann=.true., &
                frontIsNeumann=.true., backIsNeumann=.true., &
                leftVal=0.d0, rightVal=0.d0, topVal=0.d0, bottomVal=0.d0, &
                frontVal=0.d0, backVal=0.d0, isStaggered=.false., &
                helmholtzShift=0.d0, thomas=.false. )

        ! ---------------- Potential: (Ny1, Nx, Nz), nodes in x,z -----------
        call assert_spacing_sizes("Potential", Hx12(0:Nx), HPy(0:Ny1), Hz12(0:Nz), Nx, Ny1, Nz)
        PotentialHandle = init_eigen_decomp_d( &
                rows=int(Ny1,C_SIZE_T), cols=int(Nx,C_SIZE_T), layers=int(Nz,C_SIZE_T), &
                dx=Hx12(0:Nx), dy=HPy(0:Ny1), dz=Hz12(0:Nz), &
                uniformDeltaX=.false., uniformDeltaY=.false., uniformDeltaZ=.false., &
                leftIsNeumann=pX, rightIsNeumann=pX, &
                topIsNeumann=pY, bottomIsNeumann=pY, &
                frontIsNeumann=pZ, backIsNeumann=pZ, &
                leftVal=0.d0, rightVal=0.d0, topVal=0.d0, bottomVal=0.d0, &
                frontVal=0.d0, backVal=0.d0, isStaggered=.false., &
                helmholtzShift=0.d0, thomas=.false. )
    End Subroutine Initialize_GPU_Solvers

    subroutine assert_spacing_sizes(name, dx, dy, dz, ncol, nrow, nlay)
        implicit none
        character(len=*), intent(in) :: name
        real(kind=8), intent(in) :: dx(:), dy(:), dz(:)
        integer, intent(in) :: ncol, nrow, nlay
        ! The library reads exactly cols+1 / rows+1 / layers+1 elements.
        if (size(dx) /= ncol+1) error stop "dx size mismatch: "//trim(name)
        if (size(dy) /= nrow+1) error stop "dy size mismatch: "//trim(name)
        if (size(dz) /= nlay+1) error stop "dz size mismatch: "//trim(name)
        ! A zero or negative spacing means an unassigned element was passed in.
        if (minval(dx) <= 0.d0) error stop "dx has non-positive entry: "//trim(name)
        if (minval(dy) <= 0.d0) error stop "dy has non-positive entry: "//trim(name)
        if (minval(dz) <= 0.d0) error stop "dz has non-positive entry: "//trim(name)
    end subroutine assert_spacing_sizes

end module AlexCudaCompatibility
