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
        ! DIMENSION CONVENTION (isotropic API).  The library defines
        !     dim1 = the dimension whose FLATTENED indices change fastest,
        !     dim2 = next fastest,  dim3 = slowest.
        ! Fortran is column-major, so for our arrays laid out (Y, Z, X) --
        ! i.e. Field(j,k,i) with j the first index -- the mapping is simply
        ! the natural Fortran index order:
        !     dim1 = Y,   dim2 = Z,   dim3 = X.
        ! No transposition anywhere: dimNLength is just SIZE(field, N), and
        ! the deltas/BC flags for dimN belong to the axis of index N.
        ! ==================================================================

        ! ==================================================================
        ! SPACING ARRAYS.  For an axis with n unknowns, the library's
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
        ! The GPU library solves (L - helmholtzShift*I) x = b, so we pass a
        ! POSITIVE shift and solve the equivalent system
        !     (L - shift*I) x = rhs / alpha :
        !     shift(temperature) = +Ckor*Istat*GrPr/Htime, rhs scaled by GrPr
        !     shift(velocity)    = +Ckor/(Htime*DGr),      rhs scaled by 1/DGr
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
        ! All boundary VALUES in this code are homogeneous (0.d0).
        ! ==================================================================
        tX = (EVD_BCx == 1);   tY = (EVD_BCy == 1);   tZ = (EVD_BCz == 1)
        pX = (EVD_Pot_X == 1); pY = (EVD_Pot_Y == 1); pZ = (EVD_Pot_Z == 1)

        ! --- sanity checks: n+1 deltas per axis, all strictly positive -----
        Call assert_spacing_sizes()

        ! ==================================================================
        ! Handle 0: TEMPERATURE.  TmpNew(1:Ny1, 1:Nz1, 1:Nx1); cell-centred
        ! on every axis.  dim1=Y, dim2=Z, dim3=X.
        ! ==================================================================
        TemperatureHandle = init_eigen_decomp_d( &
                dim1Length = Int(Ny1, C_SIZE_T), &
                dim2Length = Int(Nz1, C_SIZE_T), &
                dim3Length = Int(Nx1, C_SIZE_T), &
                dim1Delta = HPy(0:Ny1), &
                dim2Delta = HPz(0:Nz1), &
                dim3Delta = HPx(0:Nx1), &
                dim1UniformDelta = .false., dim2UniformDelta = .false., dim3UniformDelta = .false., &
                dim1StartIsNeumann = tY, dim1EndIsNeumann = tY, &
                dim2StartIsNeumann = tZ, dim2EndIsNeumann = tZ, &
                dim3StartIsNeumann = tX, dim3EndIsNeumann = tX, &
                dim1StartVal = 0.d0, dim1EndVal = 0.d0, &
                dim2StartVal = 0.d0, dim2EndVal = 0.d0, &
                dim3StartVal = 0.d0, dim3EndVal = 0.d0, &
                isStaggered = .false., thomas = .true., &
                helmholtzShift = shiftTemperature )

        ! ==================================================================
        ! Handle 1: Vx.  VMxNew(1:Ny1, 1:Nz1, 1:Nx); node-centred along its
        ! own (x) axis, cell-centred along y and z.  Dirichlet everywhere.
        ! ==================================================================
        VxHandle = init_eigen_decomp_d( &
                dim1Length = Int(Ny1, C_SIZE_T), &
                dim2Length = Int(Nz1, C_SIZE_T), &
                dim3Length = Int(Nx,  C_SIZE_T), &
                dim1Delta = HPy(0:Ny1), &
                dim2Delta = HPz(0:Nz1), &
                dim3Delta = Hx12(0:Nx), &
                dim1UniformDelta = .false., dim2UniformDelta = .false., dim3UniformDelta = .false., &
                dim1StartIsNeumann = .false., dim1EndIsNeumann = .false., &
                dim2StartIsNeumann = .false., dim2EndIsNeumann = .false., &
                dim3StartIsNeumann = .false., dim3EndIsNeumann = .false., &
                dim1StartVal = 0.d0, dim1EndVal = 0.d0, &
                dim2StartVal = 0.d0, dim2EndVal = 0.d0, &
                dim3StartVal = 0.d0, dim3EndVal = 0.d0, &
                isStaggered = .false., thomas = .true., &
                helmholtzShift = shiftVelocity )

        ! ==================================================================
        ! Handle 2: Vy.  VMyNew(1:Ny, 1:Nz1, 1:Nx1); node-centred along y,
        ! cell-centred along z and x.  Dirichlet everywhere.
        ! ==================================================================
        VyHandle = init_eigen_decomp_d( &
                dim1Length = Int(Ny,  C_SIZE_T), &
                dim2Length = Int(Nz1, C_SIZE_T), &
                dim3Length = Int(Nx1, C_SIZE_T), &
                dim1Delta = Hy12(0:Ny), &
                dim2Delta = HPz(0:Nz1), &
                dim3Delta = HPx(0:Nx1), &
                dim1UniformDelta = .false., dim2UniformDelta = .false., dim3UniformDelta = .false., &
                dim1StartIsNeumann = .false., dim1EndIsNeumann = .false., &
                dim2StartIsNeumann = .false., dim2EndIsNeumann = .false., &
                dim3StartIsNeumann = .false., dim3EndIsNeumann = .false., &
                dim1StartVal = 0.d0, dim1EndVal = 0.d0, &
                dim2StartVal = 0.d0, dim2EndVal = 0.d0, &
                dim3StartVal = 0.d0, dim3EndVal = 0.d0, &
                isStaggered = .false., thomas = .true., &
                helmholtzShift = shiftVelocity )

        ! ==================================================================
        ! Handle 3: Vz.  VMzNew(1:Ny1, 1:Nz, 1:Nx1); node-centred along z,
        ! cell-centred along y and x.  Dirichlet everywhere.
        ! ==================================================================
        VzHandle = init_eigen_decomp_d( &
                dim1Length = Int(Ny1, C_SIZE_T), &
                dim2Length = Int(Nz,  C_SIZE_T), &
                dim3Length = Int(Nx1, C_SIZE_T), &
                dim1Delta = HPy(0:Ny1), &
                dim2Delta = Hz12(0:Nz), &
                dim3Delta = HPx(0:Nx1), &
                dim1UniformDelta = .false., dim2UniformDelta = .false., dim3UniformDelta = .false., &
                dim1StartIsNeumann = .false., dim1EndIsNeumann = .false., &
                dim2StartIsNeumann = .false., dim2EndIsNeumann = .false., &
                dim3StartIsNeumann = .false., dim3EndIsNeumann = .false., &
                dim1StartVal = 0.d0, dim1EndVal = 0.d0, &
                dim2StartVal = 0.d0, dim2EndVal = 0.d0, &
                dim3StartVal = 0.d0, dim3EndVal = 0.d0, &
                isStaggered = .false., thomas = .true., &
                helmholtzShift = shiftVelocity )

        ! ==================================================================
        ! Handle 4: PRESSURE (Dprs).  Dprs(1:Ny1, 1:Nz1, 1:Nx1); cell-centred
        ! on every axis.  Pure Poisson (shift = 0), Neumann everywhere,
        ! matching EVD_lapP which applies Neumann rows UNCONDITIONALLY.
        ! The all-Neumann system is singular; the library's singular-mode
        ! handling must stay engaged here (helmholtzShift == 0).
        ! ==================================================================
        PressureHandle = init_eigen_decomp_d( &
                dim1Length = Int(Ny1, C_SIZE_T), &
                dim2Length = Int(Nz1, C_SIZE_T), &
                dim3Length = Int(Nx1, C_SIZE_T), &
                dim1Delta = HPy(0:Ny1), &
                dim2Delta = HPz(0:Nz1), &
                dim3Delta = HPx(0:Nx1), &
                dim1UniformDelta = .false., dim2UniformDelta = .false., dim3UniformDelta = .false., &
                dim1StartIsNeumann = .true., dim1EndIsNeumann = .true., &
                dim2StartIsNeumann = .true., dim2EndIsNeumann = .true., &
                dim3StartIsNeumann = .true., dim3EndIsNeumann = .true., &
                dim1StartVal = 0.d0, dim1EndVal = 0.d0, &
                dim2StartVal = 0.d0, dim2EndVal = 0.d0, &
                dim3StartVal = 0.d0, dim3EndVal = 0.d0, &
                isStaggered = .false., thomas = .false., &
                helmholtzShift = 0.d0 )

        ! ==================================================================
        ! Handle 5: POTENTIAL (Fi).  Potential(1:Ny1, 1:Nz, 1:Nx);
        ! cell-centred along y, node-centred along z and x.  Pure Poisson
        ! (shift = 0); Neumann per axis follows the EVD_Pot_? flags,
        ! matching EVD_lap_Fi.
        ! ==================================================================
        PotentialHandle = init_eigen_decomp_d( &
                dim1Length = Int(Ny1, C_SIZE_T), &
                dim2Length = Int(Nz,  C_SIZE_T), &
                dim3Length = Int(Nx,  C_SIZE_T), &
                dim1Delta = HPy(0:Ny1), &
                dim2Delta = Hz12(0:Nz), &
                dim3Delta = Hx12(0:Nx), &
                dim1UniformDelta = .false., dim2UniformDelta = .false., dim3UniformDelta = .false., &
                dim1StartIsNeumann = pY, dim1EndIsNeumann = pY, &
                dim2StartIsNeumann = pZ, dim2EndIsNeumann = pZ, &
                dim3StartIsNeumann = pX, dim3EndIsNeumann = pX, &
                dim1StartVal = 0.d0, dim1EndVal = 0.d0, &
                dim2StartVal = 0.d0, dim2EndVal = 0.d0, &
                dim3StartVal = 0.d0, dim3EndVal = 0.d0, &
                isStaggered = .false., thomas = .false., &
                helmholtzShift = 0.d0 )

    End Subroutine Initialize_GPU_Solvers

    ! ----------------------------------------------------------------------
    ! Guard against the exact failure that produced the original NaN run:
    ! every delta slice must have n+1 entries and be strictly positive.
    ! ----------------------------------------------------------------------
    Subroutine assert_spacing_sizes()
        Use Numbers
        Use Grid
        Implicit None
        Call assert_positive(HPy (0:Ny1), 'HPy (0:Ny1)')
        Call assert_positive(HPz (0:Nz1), 'HPz (0:Nz1)')
        Call assert_positive(HPx (0:Nx1), 'HPx (0:Nx1)')
        Call assert_positive(Hy12(0:Ny ), 'Hy12(0:Ny )')
        Call assert_positive(Hz12(0:Nz ), 'Hz12(0:Nz )')
        Call assert_positive(Hx12(0:Nx ), 'Hx12(0:Nx )')
    End Subroutine assert_spacing_sizes

    Subroutine assert_positive(d, name)
        Implicit None
        Real(kind=8), Intent(in) :: d(:)
        Character(len=*), Intent(in) :: name
        If ( Minval(d) <= 0.d0 ) Then
            Write(*,*) 'FATAL: spacing array ', name, ' contains a non-positive entry: ', Minval(d)
            Write(*,*) '       (uninitialized element or mesh error) -- refusing to build a singular operator.'
            Stop 1
        End If
    End Subroutine assert_positive

end module AlexCudaCompatibility
