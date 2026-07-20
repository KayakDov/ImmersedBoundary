! ============================================================================
! ValidateSolvers.f90 -- compare each GPU solver against Alex's original CPU
! solver on the exact same problem.  For validation only; remove afterwards.
!
! Requirements to compile this in:
!   1. Add these files from Alex's ORIGINAL source tree, UNCHANGED:
!        EVD_laptmpr_3D.f90   EVD_lapVx_3D.f90   EVD_lapVy_3D.f90
!        EVD_lapVz_3D.f90     EVD_lapP_3D.f90    EVD_lap_Fi_3D.f90
!        EVD_Thomas3D_dgemm_OMP_v2_time.f90      EVD_solver_DGEMM_3D_v2.f90
!        EVD_eigenvector_ag1_R16.f90             rg_eispack_R16.f
!      One pre-existing typo in EVD_solver_DGEMM_3D_v2.f90 line ~94 must be
!      fixed or it traps under bounds checking (numerics were unaffected):
!         f_new(1:Nxsol,j,1:Nzsol) = Transpose( Amat2(1:Nzsol,1:Nysol) )
!      -> f_new(1:Nxsol,j,1:Nzsol) = Transpose( Amat2(1:Nzsol,1:Nxsol) )
!   2. Add EVD_Modules.f90 (in this package) -- the EVD_Operators and
!      Thomas_coefficients modules those files need, restored verbatim.
!   3. Add this file, and in ConvMain_3D_Q2D.f90 insert
!            Call Validate_GPU_Solvers
!      immediately after   Call Initialize_GPU_Solvers   (then run once,
!      read the report, and remove the call).
!   The CPU files link against MKL DGEMM, which your build already uses.
!
! What it does per solver: builds the CPU operator, fills a deterministic
! pseudo-random right-hand side, solves with the CPU solver exactly as the
! original time_step_Q2D.f90 / EM_forcing.f90 did, solves the same system on
! the GPU (with the RHS scaling matching AlexCudaCompatibility's shift
! convention), and prints the max relative difference.
!
! How to read the numbers:
!   *  O(1) differences  -> wiring bug (wrong deltas / BCs / missing shift).
!   *  Small differences that SHRINK ~4x when you double the resolution ->
!      expected: on cell-centred axes the library's variable-delta stencil
!      and Alex's finite-volume stencil are different (both valid,
!      second-order) discretizations near walls, so solutions agree to
!      truncation error, not to machine precision.  See LIBRARY_CHANGES.md
!      section 6 for the optional library change that makes them identical.
!   *  Pressure and (if all EVD_Pot flags are 1) Potential are singular
!      all-Neumann systems defined up to a constant; both solutions are
!      de-meaned before comparison.
! ============================================================================

Module ValidateGPU
    Implicit None
Contains

    Subroutine Validate_GPU_Solvers
        Use Numbers
        Use Parameters
        Use Numerica
        Use EVD_Operators
        Use Thomas_coefficients
        Use AlexCudaCompatibility   ! handles + GrPr
        Use eigenbcgsolver_eigen_mod, only : solve_eigen_decomp_d
        Implicit None

        Real(kind=8), Allocatable :: rhs_c(:,:,:), x_c(:,:,:)   ! CPU layout (x,y,z)
        Real(kind=8), Allocatable :: rhs_g(:,:,:), x_g(:,:,:)   ! GPU layout (y,z,x)
        Real(kind=8) :: dt_temp
        Logical      :: potSingular

        Write(*,*) ''
        Write(*,*) ' ================ CPU vs GPU solver validation ================'

        ! Build all CPU operators (fills EVD_Operators + Thomas_coefficients).
        Call EVDLapTmpr
        Call EVDLapVx
        Call EVDLapVy
        Call EVDLapVz
!        Call EVDLapP
!        Call EVD_Fi

        dt_temp = Dble(Istat)

        ! ------------------- Temperature (Nx1 x Ny1 x Nz1) -------------------
        ! CPU solves ((1/GrPr) L - (Ckor/Htime)*Istat) T = rhs.
        ! GPU solves (L - Ckor*Istat*GrPr/Htime) T = GrPr * rhs.
        Allocate( rhs_c(1:Nx1,1:Ny1,1:Nz1), x_c(1:Nx1,1:Ny1,1:Nz1) )
        Allocate( rhs_g(1:Ny1,1:Nz1,1:Nx1), x_g(1:Ny1,1:Nz1,1:Nx1) )
        Call fill_rhs(rhs_c, Nx1, Ny1, Nz1)
        Call EVD_Thomas (x_c, rhs_c, &
                EyTemp(1:Ny1,1:Ny1), Ey_invTemp(1:Ny1,1:Ny1), &
                EzTemp(1:Nz1,1:Nz1), Ez_invTemp(1:Nz1,1:Nz1), &
                LambyTemp(1:Ny1), LambzTemp(1:Nz1), &
                T_left(1:Nx1), T_center(1:Nx1), T_right(1:Nx1), &
                Nx1, Ny1, Nz1, dt_temp)
        Call to_gpu_layout(rhs_c, rhs_g, Nx1, Ny1, Nz1, GrPr)
        Call solve_eigen_decomp_d(TemperatureHandle, x_g, rhs_g)
        Call report('Temperature', x_c, x_g, Nx1, Ny1, Nz1, .false.)
        Deallocate(rhs_c, x_c, rhs_g, x_g)

        ! ------------------- Vx (Nx x Ny1 x Nz1) -----------------------------
        ! CPU solves (DGr*L - Ckor/Htime) V = rhs.
        ! GPU solves (L - Ckor/(Htime*DGr)) V = rhs / DGr.
        Allocate( rhs_c(1:Nx,1:Ny1,1:Nz1), x_c(1:Nx,1:Ny1,1:Nz1) )
        Allocate( rhs_g(1:Ny1,1:Nz1,1:Nx), x_g(1:Ny1,1:Nz1,1:Nx) )
        Call fill_rhs(rhs_c, Nx, Ny1, Nz1)
        Call EVD_Thomas (x_c, rhs_c, &
                EyVx(1:Ny1,1:Ny1), Ey_invVx(1:Ny1,1:Ny1), &
                EzVx(1:Nz1,1:Nz1), Ez_invVx(1:Nz1,1:Nz1), &
                LambyVx(1:Ny1), LambzVx(1:Nz1), &
                Vx_left(1:Nx), Vx_center(1:Nx), Vx_right(1:Nx), &
                Nx, Ny1, Nz1, 1.D0)
        Call to_gpu_layout(rhs_c, rhs_g, Nx, Ny1, Nz1, 1.D0/DGr)
        Call solve_eigen_decomp_d(VxHandle, x_g, rhs_g)
        Call report('Vx         ', x_c, x_g, Nx, Ny1, Nz1, .false.)
        Deallocate(rhs_c, x_c, rhs_g, x_g)

        ! ------------------- Vy (Nx1 x Ny x Nz1) -----------------------------
        Allocate( rhs_c(1:Nx1,1:Ny,1:Nz1), x_c(1:Nx1,1:Ny,1:Nz1) )
        Allocate( rhs_g(1:Ny,1:Nz1,1:Nx1), x_g(1:Ny,1:Nz1,1:Nx1) )
        Call fill_rhs(rhs_c, Nx1, Ny, Nz1)
        Call EVD_Thomas (x_c, rhs_c, &
                EyVy(1:Ny,1:Ny),   Ey_invVy(1:Ny,1:Ny), &
                EzVy(1:Nz1,1:Nz1), Ez_invVy(1:Nz1,1:Nz1), &
                LambyVy(1:Ny), LambzVy(1:Nz1), &
                Vy_left(1:Nx1), Vy_center(1:Nx1), Vy_right(1:Nx1), &
                Nx1, Ny, Nz1, 1.D0)
        Call to_gpu_layout(rhs_c, rhs_g, Nx1, Ny, Nz1, 1.D0/DGr)
        Call solve_eigen_decomp_d(VyHandle, x_g, rhs_g)
        Call report('Vy         ', x_c, x_g, Nx1, Ny, Nz1, .false.)
        Deallocate(rhs_c, x_c, rhs_g, x_g)

        ! ------------------- Vz (Nx1 x Ny1 x Nz) -----------------------------
        Allocate( rhs_c(1:Nx1,1:Ny1,1:Nz), x_c(1:Nx1,1:Ny1,1:Nz) )
        Allocate( rhs_g(1:Ny1,1:Nz,1:Nx1), x_g(1:Ny1,1:Nz,1:Nx1) )
        Call fill_rhs(rhs_c, Nx1, Ny1, Nz)
        Call EVD_Thomas (x_c, rhs_c, &
                EyVz(1:Ny1,1:Ny1), Ey_invVz(1:Ny1,1:Ny1), &
                EzVz(1:Nz,1:Nz),   Ez_invVz(1:Nz,1:Nz), &
                LambyVz(1:Ny1), LambzVz(1:Nz), &
                Vz_left(1:Nx1), Vz_center(1:Nx1), Vz_right(1:Nx1), &
                Nx1, Ny1, Nz, 1.D0)
        Call to_gpu_layout(rhs_c, rhs_g, Nx1, Ny1, Nz, 1.D0/DGr)
        Call solve_eigen_decomp_d(VzHandle, x_g, rhs_g)
        Call report('Vz         ', x_c, x_g, Nx1, Ny1, Nz, .false.)
        Deallocate(rhs_c, x_c, rhs_g, x_g)

        ! Pressure intentionally remains on Alex's CPU solver.
        Write(*,*) '   Pressure     uses Alex CPU solver (GPU comparison skipped)'  ! ------------------- Potential: intentionally on Alex's CPU solver ----
        Write(*,*) '   Potential    uses Alex CPU solver (GPU comparison skipped)'

        Write(*,*) ' ==============================================================='
        Write(*,*) ''
    End Subroutine Validate_GPU_Solvers

    ! Deterministic, mesh-independent pseudo-random RHS in CPU (x,y,z) layout.
    Subroutine fill_rhs(f, N1, N2, N3)
        Integer, Intent(in) :: N1, N2, N3
        Real(kind=8), Intent(out) :: f(1:N1,1:N2,1:N3)
        Integer :: i, j, k
        Do k=1,N3
            Do j=1,N2
                Do i=1,N1
                    f(i,j,k) = Sin(0.7D0*i + 1.3D0*j + 2.1D0*k) &
                             + 0.5D0 * Cos(1.9D0*i - 0.8D0*j + 0.3D0*k)
                End Do
            End Do
        End Do
    End Subroutine fill_rhs

    ! (x,y,z) -> (y,z,x), with a scale factor for the RHS conventions.
    Subroutine to_gpu_layout(fc, fg, N1, N2, N3, scale)
        Integer, Intent(in) :: N1, N2, N3
        Real(kind=8), Intent(in)  :: fc(1:N1,1:N2,1:N3)
        Real(kind=8), Intent(out) :: fg(1:N2,1:N3,1:N1)
        Real(kind=8), Intent(in)  :: scale
        Integer :: i, j, k
        Do k=1,N3
            Do j=1,N2
                Do i=1,N1
                    fg(j,k,i) = fc(i,j,k) * scale
                End Do
            End Do
        End Do
    End Subroutine to_gpu_layout

    Subroutine report(name, xc, xg, N1, N2, N3, demean)
        Character(len=*), Intent(in) :: name
        Integer, Intent(in) :: N1, N2, N3
        Real(kind=8), Intent(in) :: xc(1:N1,1:N2,1:N3)
        Real(kind=8), Intent(in) :: xg(1:N2,1:N3,1:N1)
        Logical, Intent(in) :: demean
        Real(kind=8) :: cmean, gmean, err, nrm, dc, dg
        Integer :: i, j, k
        cmean = 0.D0;  gmean = 0.D0
        If (demean) Then
            cmean = Sum(xc) / Dble(N1*N2*N3)
            gmean = Sum(xg) / Dble(N1*N2*N3)
        End If
        err = 0.D0;  nrm = 1.D-300
        Do k=1,N3
            Do j=1,N2
                Do i=1,N1
                    dc = xc(i,j,k) - cmean
                    dg = xg(j,k,i) - gmean
                    err = Max(err, Abs(dg - dc))
                    nrm = Max(nrm, Abs(dc))
                End Do
            End Do
        End Do
        Write(*,'(A,A,ES12.4,A,ES12.4,A)') '   ', name//'  max|cpu-gpu| = ', err, &
              '   relative = ', err/nrm, Merge('  (de-meaned)', '             ', demean)
    End Subroutine report

End Module ValidateGPU
