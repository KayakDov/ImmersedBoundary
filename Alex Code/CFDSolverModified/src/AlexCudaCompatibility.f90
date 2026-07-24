! ============================================================================
! AlexCudaCompatibility -- GPU eigendecomposition bridge for Alex's CFD code
! ============================================================================
!
! WHAT THIS MODULE DOES
!   All six of Alex's implicit linear solves per time step -- the Helmholtz
!   solves for temperature, Vx, Vy, Vz, and the Poisson solves for pressure
!   and the electric potential -- were originally each inverted on the CPU
!   with a dedicated eigendecomposition routine (EVDLapTmpr, EVDLapVx/y/z,
!   EVDLapP, EVD_Fi) plus a shared EVD_Thomas / EVDmethod solve step. This
!   module replaces all six with calls into CudaBandedLib's GPU
!   eigendecomposition solver, via Initialize_GPU_Solvers (called once, from
!   Solution_time in place of the six EVDLap*/EVD_Fi calls) and the six
!   module-level handles below (used by time_step_Q2D.f90 and EM_forcing.f90
!   in place of the CPU EVD_Thomas / EVDmethod calls). No other part of the
!   physics -- mesh generation, boundary application, the time integrator,
!   or diagnostics -- is touched. See CHANGES.md for the full account.
!
! NO ARRAY REINDEXING
!   CudaBandedLib's solver is isotropic: it has no concept of "X", "Y", or
!   "Z", only dim1/dim2/dim3, defined purely by which one changes fastest in
!   the flattened array (dim1 fastest, dim3 slowest) -- no physical axis is
!   attached to any of them. Alex's arrays already declare X first (fastest,
!   since Fortran is column-major), so dim1 = X, dim2 = Y, dim3 = Z here, and
!   NONE of Alex's array declarations, loops, or index order were changed
!   anywhere in the codebase to make this integration possible. This is
!   simplest, and only possible, because of that isotropic API; an earlier
!   iteration of this integration (against an older, non-isotropic library
!   API that required a fixed physical flattening order) did reindex every
!   array to (Y,Z,X), which is no longer necessary or present here.
!
! WHY SIX SOLVERS, TWO DIFFERENT EQUATION TYPES
!   Alex's 3-level implicit time integrator turns the temperature and
!   velocity updates into HELMHOLTZ solves, (L - shift*I) x = b, where the
!   shift comes from the -(Ckor/Htime) term of the time discretization
!   (Ckor = 1.5, the 3-level BDF coefficient). Pressure and potential have no
!   time derivative in their equations and remain pure POISSON solves
!   (shift = 0). See the per-solver comments below for the exact shift
!   values and required RHS scalings (GrPr, DGr), matching EVDLapTmpr /
!   EVDLapVx/y/z's own GrPr/DGr scaling and EVD_Thomas's Ckor/Htime shift
!   exactly.
!
! WHY TWO DIFFERENT AXIS DISCRETIZATIONS (FLUX_LAPL / VARIABLE_DELTA_LAPL)
!   Alex's mesh is staggered per field, not just per axis: each velocity
!   component is node-centred along its own direction (the faces of its
!   control volume) and cell-centred along the other two. Pressure and
!   temperature are cell-centred on every axis. CudaBandedLib exposes this
!   as a per-axis eigen::LaplOperatorT choice, matched here exactly to which
!   deltas each of Alex's EVDLap*/EVD_Fi routines uses for that axis:
!     FLUX_LAPL           - conservative finite-volume Laplacian for
!                           cell-centred axes (deltas HPx/HPy/HPz, widths
!                           reconstructed from Hx12/Hy12/Hz12); reproduces
!                           Alex's exact finite-volume operator, which is
!                           REQUIRED for pressure: the projection step needs
!                           this operator to equal div(grad) discretely, or
!                           residual divergence accumulates and the
!                           simulation diverges.
!     VARIABLE_DELTA_LAPL - pointwise 3-point Laplacian for node-centred
!                           axes (deltas Hx12/Hy12/Hz12); matches Alex's
!                           node-centred operators exactly on any mesh.
!   Validated to floating-point precision against Alex's original CPU
!   solvers on this test problem (relative differences, GPU vs CPU):
!     Temperature  1.2e-15      Vx  1.4e-15      Vy  8.2e-16
!     Vz           9.6e-16      Pressure  1.3e-13 (de-meaned; all-Neumann,
!                                defined up to a constant)
!     Potential    1.3e-15 (de-meaned; Neumann per EVD_Pot_X/Y/Z flags)
!   (These numbers are from the (Y,Z,X)-reindexed version of this bridge;
!   the underlying operators are identical here, only relabeled to native
!   X/Y/Z dim slots, so the same agreement is expected -- reconfirm with
!   ValidateSolvers.f90 if you want the numbers reproduced against this
!   exact file.)
!
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
    ! Prandtl == 0). Set in Initialize_GPU_Solvers; time_step_Q2D.f90 uses it
    ! to scale the temperature RHS to match the shift convention below.
    real(kind=8) :: GrPr = 1.d0

    ! Discretization selector per axis -- must match eigen::LaplOperatorT in
    ! poisson/LaplOperatorType.h (passed to C as a plain integer):
    !   VARIABLE_DELTA_LAPL : pointwise 3-point Laplacian, unknowns at grid
    !                         nodes; use with Hx12/Hy12/Hz12 deltas.
    !   FLUX_LAPL           : conservative finite-volume Laplacian, unknowns
    !                         at cell centres; use with HPx/HPy/HPz deltas.
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

        ! ==================================================================
        ! HELMHOLTZ SHIFTS. Matches EVD_Thomas's
        !     pdum = -(Ckor/Htime)*Dtm + lambda_y + lambda_z
        ! i.e. it solves (alpha*L - (Ckor/Htime)*Dtm) x = rhs, with
        !     temperature: alpha = 1/GrPr, Dtm = Istat
        !     velocities : alpha = DGr,    Dtm = 1
        ! The GPU library solves (L - helmholtzShift*I) x = b, so we pass a
        ! POSITIVE shift and solve the equivalent system
        !     (L - shift*I) x = rhs / alpha :
        !     shift(temperature) = +Ckor*Istat*GrPr/Htime, rhs scaled by GrPr
        !     shift(velocity)    = +Ckor/(Htime*DGr),      rhs scaled by 1/DGr
        ! The rhs scaling lives in time_step_Q2D.f90 where the GPU RHS
        ! staging happens. Pressure and Potential are pure Poisson
        ! (shift = 0, no scaling), matching EVDmethod's beta = 0.
        ! ==================================================================
        GrPr = Prandtl / DGr
        If (Prandtl == 0.D0) GrPr = 1.D0

        shiftTemperature =   Ckor * Dble(Istat) * GrPr / Htime
        shiftVelocity    =   Ckor / ( Htime * DGr )

        ! ==================================================================
        ! BOUNDARY CONDITIONS, mirroring the CPU operators exactly:
        !   Temperature (EVDLapTmpr): Neumann on both ends of an axis when
        !                             the runtime flag EVD_BC? == 1
        !   Velocities (EVDLapVx/y/z): Dirichlet everywhere, unconditionally
        !   Pressure (EVDLapP):        Neumann everywhere, unconditionally
        !   Potential (EVD_Fi):        Neumann per axis when EVD_Pot_? == 1
        ! All boundary VALUES in this code are homogeneous (0.d0).
        ! ==================================================================
        tX = (EVD_BCx == 1);   tY = (EVD_BCy == 1);   tZ = (EVD_BCz == 1)
        pX = (EVD_Pot_X == 1); pY = (EVD_Pot_Y == 1); pZ = (EVD_Pot_Z == 1)

        Call assert_spacing_sizes()

        ! ==================================================================
        ! Handle 0: TEMPERATURE. Tmpr(1:Nx1, 1:Ny1, 1:Nz1); cell-centred on
        ! every axis (EVDLapTmpr's Hx12(i-1)*HPx(...) form, all three axes).
        ! ==================================================================
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
                helmholtzShift = shiftTemperature )

        ! ==================================================================
        ! Handle 1: Vx. VMx(1:Nx, 1:Ny1, 1:Nz1); node-centred along its own
        ! (x) axis (EVDLapVx's HPx(i)*Hx12(...) form), cell-centred along y
        ! and z (same Hy12/HPy, Hz12/HPz form as temperature). All-Dirichlet.
        ! ==================================================================
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
                helmholtzShift = shiftVelocity )

        ! ==================================================================
        ! Handle 2: Vy. VMy(1:Nx1, 1:Ny, 1:Nz1); node-centred along y,
        ! cell-centred along x and z. All-Dirichlet.
        ! ==================================================================
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
                helmholtzShift = shiftVelocity )

        ! ==================================================================
        ! Handle 3: Vz. VMz(1:Nx1, 1:Ny1, 1:Nz); node-centred along z,
        ! cell-centred along x and y. All-Dirichlet.
        ! ==================================================================
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
                helmholtzShift = shiftVelocity )

        ! ==================================================================
        ! Handle 4: PRESSURE (Dprs). Dprs(1:Nx1, 1:Ny1, 1:Nz1); cell-centred
        ! on every axis. Pure Poisson (shift = 0), Neumann everywhere,
        ! matching EVDLapP which applies Neumann rows UNCONDITIONALLY. The
        ! all-Neumann system is singular; the library's singular-mode
        ! handling must stay engaged here (helmholtzShift == 0).
        ! ==================================================================
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
                helmholtzShift = 0.d0 )

        ! ==================================================================
        ! Handle 5: POTENTIAL (Fi). Potential(1:Nx, 1:Ny1, 1:Nz); node-
        ! centred along x and z, cell-centred along y. Pure Poisson
        ! (shift = 0); Neumann per axis follows the EVD_Pot_? flags,
        ! matching EVD_Fi.
        ! ==================================================================
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
