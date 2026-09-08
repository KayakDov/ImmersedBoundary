! ============================================================================
! AlexCudaCompatibility -- GPU eigendecomposition bridge for Alex's CFD code
! ============================================================================
! Replaces the six CPU eigendecomposition solves per time step (temperature,
! Vx, Vy, Vz, pressure, potential -- originally EVDLapTmpr/EVDLapVx/y/z/
! EVDLapP/EVD_Fi plus EVD_Thomas/EVDmethod) with CudaBandedLib's GPU solver.
! No array in the codebase is reindexed; dim1/dim2/dim3 below match Alex's
! native X/Y/Z declaration order exactly. See CHANGES.md for the full
! rationale, the axis-discretization table, and validation numbers.
! ============================================================================

module AlexCudaCompatibility
    use iso_c_binding, only : C_SIZE_T, C_INT
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
    ! Prandtl == 0). Set here; time_step_Q2D.f90 uses it to scale the
    ! temperature RHS to match the shift convention below.
    real(kind=8) :: GrPr = 1.d0

    ! Per-axis discretization selector, matching eigen::LaplOperatorT in
    ! poisson/LaplOperatorType.h:
    !   VARIABLE_DELTA_LAPL : pointwise 3-point Laplacian, unknowns at grid
    !                         nodes (use with Hx12/Hy12/Hz12 deltas)
    !   FLUX_LAPL           : conservative finite-volume Laplacian, unknowns
    !                         at cell centres (use with HPx/HPy/HPz deltas)
    integer(C_INT), parameter :: UNIFORM_NODE_CENTERED_LAPL = 0
    integer(C_INT), parameter :: UNIFORM_STAGGERED_LAPL     = 1
    integer(C_INT), parameter :: VARIABLE_DELTA_LAPL        = 2
    integer(C_INT), parameter :: FLUX_LAPL                  = 3

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

        ! Matches EVD_Thomas's pdum = -(Ckor/Htime)*Dtm + lambda_y + lambda_z,
        ! i.e. it solves (alpha*L - (Ckor/Htime)*Dtm) x = rhs, with
        ! temperature: alpha = 1/GrPr, Dtm = Istat; velocities: alpha = DGr,
        ! Dtm = 1. The GPU library solves (L - helmholtzShift*I) x = b, so we
        ! pass a positive shift and rhs/alpha:
        !   shift(temperature) = +Ckor*Istat*GrPr/Htime, rhs scaled by GrPr
        !   shift(velocity)    = +Ckor/(Htime*DGr),      rhs scaled by 1/DGr
        ! (rhs scaling happens in time_step_Q2D.f90). Pressure and Potential
        ! are pure Poisson (shift = 0, no scaling), matching EVDmethod's
        ! beta = 0.
        GrPr = Prandtl / DGr
        If (Prandtl == 0.D0) GrPr = 1.D0

        shiftTemperature =   Ckor * Dble(Istat) * GrPr / Htime
        shiftVelocity    =   Ckor / ( Htime * DGr )

        ! Boundary conditions, mirroring the CPU operators exactly: EVDLapTmpr
        ! is Neumann on an axis when EVD_BC? == 1; EVDLapVx/y/z are Dirichlet
        ! unconditionally; EVDLapP is Neumann unconditionally; EVD_Fi is
        ! Neumann per axis when EVD_Pot_? == 1. All boundary values are 0.
        tX = (EVD_BCx == 1);   tY = (EVD_BCy == 1);   tZ = (EVD_BCz == 1)
        pX = (EVD_Pot_X == 1); pY = (EVD_Pot_Y == 1); pZ = (EVD_Pot_Z == 1)

        Call assert_spacing_sizes()

        ! Temperature: Tmpr(1:Nx1,1:Ny1,1:Nz1), cell-centred on every axis
        ! (EVDLapTmpr's Hx12(i-1)*HPx(...) form on all three axes).
        TemperatureHandle = init_eigen_decomp_d( &
                dim1Length = Int(Nx1, C_SIZE_T), &
                dim2Length = Int(Ny1, C_SIZE_T), &
                dim3Length = Int(Nz1, C_SIZE_T), &
                dim1Delta = HPx(0:Nx1), &
                dim2Delta = HPy(0:Ny1), &
                dim3Delta = HPz(0:Nz1), &
                dim1SegType = FLUX_LAPL, &
                dim2SegType = FLUX_LAPL, &
                dim3SegType = FLUX_LAPL, &
                dim1StartIsNeumann = tX, dim1EndIsNeumann = tX, &
                dim2StartIsNeumann = tY, dim2EndIsNeumann = tY, &
                dim3StartIsNeumann = tZ, dim3EndIsNeumann = tZ, &
                dim1StartVal = 0.d0, dim1EndVal = 0.d0, &
                dim2StartVal = 0.d0, dim2EndVal = 0.d0, &
                dim3StartVal = 0.d0, dim3EndVal = 0.d0, &
                thomas = .true., &
                helmholtzShift = shiftTemperature , &
                gpuInd = 0 &
            )

        ! Vx: VMx(1:Nx,1:Ny1,1:Nz1); node-centred along its own (x) axis
        ! (EVDLapVx's HPx(i)*Hx12(...) form), cell-centred along y/z. Dirichlet.
        VxHandle = init_eigen_decomp_d( &
                dim1Length = Int(Nx,  C_SIZE_T), &
                dim2Length = Int(Ny1, C_SIZE_T), &
                dim3Length = Int(Nz1, C_SIZE_T), &
                dim1Delta = Hx12(0:Nx), &
                dim2Delta = HPy(0:Ny1), &
                dim3Delta = HPz(0:Nz1), &
                dim1SegType = VARIABLE_DELTA_LAPL, &
                dim2SegType = FLUX_LAPL, &
                dim3SegType = FLUX_LAPL, &
                dim1StartIsNeumann = .false., dim1EndIsNeumann = .false., &
                dim2StartIsNeumann = .false., dim2EndIsNeumann = .false., &
                dim3StartIsNeumann = .false., dim3EndIsNeumann = .false., &
                dim1StartVal = 0.d0, dim1EndVal = 0.d0, &
                dim2StartVal = 0.d0, dim2EndVal = 0.d0, &
                dim3StartVal = 0.d0, dim3EndVal = 0.d0, &
                thomas = .true., &
                helmholtzShift = shiftVelocity, &
                gpuInd = 0 &
            )

        ! Vy: VMy(1:Nx1,1:Ny,1:Nz1); node-centred along y, cell-centred along
        ! x/z. Dirichlet.
        VyHandle = init_eigen_decomp_d( &
                dim1Length = Int(Nx1, C_SIZE_T), &
                dim2Length = Int(Ny,  C_SIZE_T), &
                dim3Length = Int(Nz1, C_SIZE_T), &
                dim1Delta = HPx(0:Nx1), &
                dim2Delta = Hy12(0:Ny), &
                dim3Delta = HPz(0:Nz1), &
                dim1SegType = FLUX_LAPL, &
                dim2SegType = VARIABLE_DELTA_LAPL, &
                dim3SegType = FLUX_LAPL, &
                dim1StartIsNeumann = .false., dim1EndIsNeumann = .false., &
                dim2StartIsNeumann = .false., dim2EndIsNeumann = .false., &
                dim3StartIsNeumann = .false., dim3EndIsNeumann = .false., &
                dim1StartVal = 0.d0, dim1EndVal = 0.d0, &
                dim2StartVal = 0.d0, dim2EndVal = 0.d0, &
                dim3StartVal = 0.d0, dim3EndVal = 0.d0, &
                thomas = .true., &
                helmholtzShift = shiftVelocity, &
                gpuInd = 0 &
        )

        ! Vz: VMz(1:Nx1,1:Ny1,1:Nz); node-centred along z, cell-centred along
        ! x/y. Dirichlet.
        VzHandle = init_eigen_decomp_d( &
                dim1Length = Int(Nx1, C_SIZE_T), &
                dim2Length = Int(Ny1, C_SIZE_T), &
                dim3Length = Int(Nz,  C_SIZE_T), &
                dim1Delta = HPx(0:Nx1), &
                dim2Delta = HPy(0:Ny1), &
                dim3Delta = Hz12(0:Nz), &
                dim1SegType = FLUX_LAPL, &
                dim2SegType = FLUX_LAPL, &
                dim3SegType = VARIABLE_DELTA_LAPL, &
                dim1StartIsNeumann = .false., dim1EndIsNeumann = .false., &
                dim2StartIsNeumann = .false., dim2EndIsNeumann = .false., &
                dim3StartIsNeumann = .false., dim3EndIsNeumann = .false., &
                dim1StartVal = 0.d0, dim1EndVal = 0.d0, &
                dim2StartVal = 0.d0, dim2EndVal = 0.d0, &
                dim3StartVal = 0.d0, dim3EndVal = 0.d0, &
                thomas = .true., &
                helmholtzShift = shiftVelocity, &
                gpuInd = 0 &
            )

        ! Pressure: Dprs(1:Nx1,1:Ny1,1:Nz1), cell-centred on every axis. Pure
        ! Poisson, Neumann everywhere (matching EVDLapP, which applies Neumann
        ! rows unconditionally). This all-Neumann system is singular; the
        ! library's singular-mode handling must stay engaged (shift == 0).
        PressureHandle = init_eigen_decomp_d( &
                dim1Length = Int(Nx1, C_SIZE_T), &
                dim2Length = Int(Ny1, C_SIZE_T), &
                dim3Length = Int(Nz1, C_SIZE_T), &
                dim1Delta = HPx(0:Nx1), &
                dim2Delta = HPy(0:Ny1), &
                dim3Delta = HPz(0:Nz1), &
                dim1SegType = FLUX_LAPL, &
                dim2SegType = FLUX_LAPL, &
                dim3SegType = FLUX_LAPL, &
                dim1StartIsNeumann = .true., dim1EndIsNeumann = .true., &
                dim2StartIsNeumann = .true., dim2EndIsNeumann = .true., &
                dim3StartIsNeumann = .true., dim3EndIsNeumann = .true., &
                dim1StartVal = 0.d0, dim1EndVal = 0.d0, &
                dim2StartVal = 0.d0, dim2EndVal = 0.d0, &
                dim3StartVal = 0.d0, dim3EndVal = 0.d0, &
                thomas = .false., &
                helmholtzShift = 0.d0, &
                gpuInd = 0 &
            )

        ! Potential: Potential(1:Nx,1:Ny1,1:Nz); node-centred along x/z,
        ! cell-centred along y. Pure Poisson; Neumann per axis follows the
        ! EVD_Pot_? flags, matching EVD_Fi.
        PotentialHandle = init_eigen_decomp_d( &
                dim1Length = Int(Nx,  C_SIZE_T), &
                dim2Length = Int(Ny1, C_SIZE_T), &
                dim3Length = Int(Nz,  C_SIZE_T), &
                dim1Delta = Hx12(0:Nx), &
                dim2Delta = HPy(0:Ny1), &
                dim3Delta = Hz12(0:Nz), &
                dim1SegType = VARIABLE_DELTA_LAPL, &
                dim2SegType = FLUX_LAPL, &
                dim3SegType = VARIABLE_DELTA_LAPL, &
                dim1StartIsNeumann = pX, dim1EndIsNeumann = pX, &
                dim2StartIsNeumann = pY, dim2EndIsNeumann = pY, &
                dim3StartIsNeumann = pZ, dim3EndIsNeumann = pZ, &
                dim1StartVal = 0.d0, dim1EndVal = 0.d0, &
                dim2StartVal = 0.d0, dim2EndVal = 0.d0, &
                dim3StartVal = 0.d0, dim3EndVal = 0.d0, &
                thomas = .false., &
                helmholtzShift = 0.d0, &
                gpuInd = 0 &
            )

    End Subroutine Initialize_GPU_Solvers

    ! Every delta slice passed above must have n+1 entries and be strictly
    ! positive, or the operator built from it is singular/undefined.
    Subroutine assert_spacing_sizes()
        Use Numbers
        Use Grid
        Implicit None
        Call assert_positive(HPx (0:Nx1), 'HPx (0:Nx1)')
        Call assert_positive(HPy (0:Ny1), 'HPy (0:Ny1)')
        Call assert_positive(HPz (0:Nz1), 'HPz (0:Nz1)')
        Call assert_positive(Hx12(0:Nx ), 'Hx12(0:Nx )')
        Call assert_positive(Hy12(0:Ny ), 'Hy12(0:Ny )')
        Call assert_positive(Hz12(0:Nz ), 'Hz12(0:Nz )')
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
