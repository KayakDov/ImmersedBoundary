! ============================================================================
! ValidateSolvers.f90 -- compare each GPU solver against Alex's original CPU
! solver on the exact same problem, using Alex's ACTUAL Conv.dat (real
! resolution). For validation only; remove afterwards along with the ten
! EVD_* files, EVD_Modules.f90, and the two lines this adds to
! Solution_time_Q2D.f90.
!
! NO REINDEXING / NO TRANSPOSE: unlike the harness used earlier in this
! project (against a since-abandoned (Y,Z,X)-reindexed version of the
! bridge), this version needs no "to_gpu_layout" step at all. Alex's
! original arrays and the GPU library both use the same native (X,Y,Z)
! order (dim1=X, dim2=Y, dim3=Z -- see AlexCudaCompatibility.f90), so the
! identical array is passed to both the CPU and GPU solve calls.
!
! Requirements to compile this in (see this package's README):
!   1. Add the ten EVD_* files + rg_eispack_R16.f from this package (verbatim
!      copies of Alex's original CPU solver, except one pre-existing,
!      normally-dormant typo in EVD_solver_DGEMM_3D_v2.f90 fixed for
!      bounds-check safety -- see the comment at that line).
!   2. Add EVD_Modules.f90 (in this package) -- restores the EVD_Operators /
!      Thomas_coefficients module data those files need.
!   3. Add this file, and in Solution_time_Q2D.f90 insert
!            Call Validate_GPU_Solvers
!      immediately after the (already-relocated) "Call Initialize_GPU_Solvers()"
!      line, plus "Use ValidateGPU" in that subroutine's Use block. Both are
!      marked "VALIDATION ONLY" in the patch; remove them afterwards.
!
! What it does per solver: builds the CPU reference operator (Alex's
! EVDLapTmpr/EVDLapVx/y/z/EVDLapP/EVD_Fi), fills a deterministic
! pseudo-random right-hand side directly in Alex's native array layout,
! solves with the CPU solver exactly as time_step_Q2D.f90/EM_forcing.f90
! originally did, solves the SAME array with the GPU solver (scaled
! identically to how time_step_Q2D.f90/EM_forcing.f90 now do it), and prints
! the max relative difference. No copying between two different index
! orders anywhere -- both solves read/write the same-shaped array.
!
! Runs at Alex's ACTUAL Conv.dat resolution (100x100x100 for the current
! file), so this will take substantially longer than the earlier debug-grid
! version: building the CPU reference operators means Vgeev-decomposing six
! dense matrices up to 101x101, and both solve paths do real work at this
! size. Expect this to run for a while; that cost is the entire point of
! running it at real resolution rather than a 3x2x4 debug grid.
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

        Real(kind=8), Allocatable :: rhs(:,:,:), x_cpu(:,:,:), x_gpu(:,:,:)
        Real(kind=8) :: dt_temp
        Logical      :: potSingular

        Write(*,*) ''
        Write(*,*) ' ================ CPU vs GPU solver validation (native X,Y,Z; N=', Nx, ') ================'

        ! Build all six CPU operators (fills EVD_Operators + Thomas_coefficients).
        Call EVDLapTmpr
        Call EVDLapVx
        Call EVDLapVy
        Call EVDLapVz
        Call EVDLapP
        Call EVD_Fi

        dt_temp = Dble(Istat)

        ! ------------------- Temperature (Nx1 x Ny1 x Nz1) -------------------
        Allocate( rhs(1:Nx1,1:Ny1,1:Nz1), x_cpu(1:Nx1,1:Ny1,1:Nz1), x_gpu(1:Nx1,1:Ny1,1:Nz1) )
        Call fill_rhs(rhs, Nx1, Ny1, Nz1)
        Call EVD_Thomas (x_cpu, rhs, &
                EyTemp(1:Ny1,1:Ny1), Ey_invTemp(1:Ny1,1:Ny1), &
                EzTemp(1:Nz1,1:Nz1), Ez_invTemp(1:Nz1,1:Nz1), &
                LambyTemp(1:Ny1), LambzTemp(1:Nz1), &
                T_left(1:Nx1), T_center(1:Nx1), T_right(1:Nx1), &
                Nx1, Ny1, Nz1, dt_temp)
        Call solve_eigen_decomp_d(TemperatureHandle, x_gpu, rhs * GrPr)
        Call report('Temperature', x_cpu, x_gpu, Nx1, Ny1, Nz1, .false.)
        Deallocate(rhs, x_cpu, x_gpu)

        ! ------------------- Vx (Nx x Ny1 x Nz1) -----------------------------
        Allocate( rhs(1:Nx,1:Ny1,1:Nz1), x_cpu(1:Nx,1:Ny1,1:Nz1), x_gpu(1:Nx,1:Ny1,1:Nz1) )
        Call fill_rhs(rhs, Nx, Ny1, Nz1)
        Call EVD_Thomas (x_cpu, rhs, &
                EyVx(1:Ny1,1:Ny1), Ey_invVx(1:Ny1,1:Ny1), &
                EzVx(1:Nz1,1:Nz1), Ez_invVx(1:Nz1,1:Nz1), &
                LambyVx(1:Ny1), LambzVx(1:Nz1), &
                Vx_left(1:Nx), Vx_center(1:Nx), Vx_right(1:Nx), &
                Nx, Ny1, Nz1, 1.D0)
        Call solve_eigen_decomp_d(VxHandle, x_gpu, rhs / DGr)
        Call report('Vx         ', x_cpu, x_gpu, Nx, Ny1, Nz1, .false.)
        Deallocate(rhs, x_cpu, x_gpu)

        ! ------------------- Vy (Nx1 x Ny x Nz1) -----------------------------
        Allocate( rhs(1:Nx1,1:Ny,1:Nz1), x_cpu(1:Nx1,1:Ny,1:Nz1), x_gpu(1:Nx1,1:Ny,1:Nz1) )
        Call fill_rhs(rhs, Nx1, Ny, Nz1)
        Call EVD_Thomas (x_cpu, rhs, &
                EyVy(1:Ny,1:Ny),   Ey_invVy(1:Ny,1:Ny), &
                EzVy(1:Nz1,1:Nz1), Ez_invVy(1:Nz1,1:Nz1), &
                LambyVy(1:Ny), LambzVy(1:Nz1), &
                Vy_left(1:Nx1), Vy_center(1:Nx1), Vy_right(1:Nx1), &
                Nx1, Ny, Nz1, 1.D0)
        Call solve_eigen_decomp_d(VyHandle, x_gpu, rhs / DGr)
        Call report('Vy         ', x_cpu, x_gpu, Nx1, Ny, Nz1, .false.)
        Deallocate(rhs, x_cpu, x_gpu)

        ! ------------------- Vz (Nx1 x Ny1 x Nz) -----------------------------
        Allocate( rhs(1:Nx1,1:Ny1,1:Nz), x_cpu(1:Nx1,1:Ny1,1:Nz), x_gpu(1:Nx1,1:Ny1,1:Nz) )
        Call fill_rhs(rhs, Nx1, Ny1, Nz)
        Call EVD_Thomas (x_cpu, rhs, &
                EyVz(1:Ny1,1:Ny1), Ey_invVz(1:Ny1,1:Ny1), &
                EzVz(1:Nz,1:Nz),   Ez_invVz(1:Nz,1:Nz), &
                LambyVz(1:Ny1), LambzVz(1:Nz), &
                Vz_left(1:Nx1), Vz_center(1:Nx1), Vz_right(1:Nx1), &
                Nx1, Ny1, Nz, 1.D0)
        Call solve_eigen_decomp_d(VzHandle, x_gpu, rhs / DGr)
        Call report('Vz         ', x_cpu, x_gpu, Nx1, Ny1, Nz, .false.)
        Deallocate(rhs, x_cpu, x_gpu)

        ! ------------------- Pressure (Nx1 x Ny1 x Nz1), all-Neumann ---------
        ! Singular system: solutions defined up to a constant; compare de-meaned.
        Allocate( rhs(1:Nx1,1:Ny1,1:Nz1), x_cpu(1:Nx1,1:Ny1,1:Nz1), x_gpu(1:Nx1,1:Ny1,1:Nz1) )
        Call fill_rhs(rhs, Nx1, Ny1, Nz1)
        Call EVDmethod (x_cpu, rhs, &
                ExxP(1:Nx1,1:Nx1), Ex_invP(1:Nx1,1:Nx1), &
                EyP(1:Ny1,1:Ny1),  Ey_invP(1:Ny1,1:Ny1), &
                EzP(1:Nz1,1:Nz1),  Ez_invP(1:Nz1,1:Nz1), &
                LambxP(1:Nx1), LambyP(1:Ny1), LambzP(1:Nz1), &
                Nx1, Ny1, Nz1, 1.D0, 1.D0, 1.D0, 0.D0)
        Call solve_eigen_decomp_d(PressureHandle, x_gpu, rhs)
        Call report('Pressure   ', x_cpu, x_gpu, Nx1, Ny1, Nz1, .true.)
        Deallocate(rhs, x_cpu, x_gpu)

        ! ------------------- Potential (Nx x Ny1 x Nz) -----------------------
        potSingular = (EVD_Pot_X == 1) .and. (EVD_Pot_Y == 1) .and. (EVD_Pot_Z == 1)
        Allocate( rhs(1:Nx,1:Ny1,1:Nz), x_cpu(1:Nx,1:Ny1,1:Nz), x_gpu(1:Nx,1:Ny1,1:Nz) )
        Call fill_rhs(rhs, Nx, Ny1, Nz)
        Call EVDmethod (x_cpu, rhs, &
                ExxFi(1:Nx,1:Nx),  Ex_invFi(1:Nx,1:Nx), &
                EyFi(1:Ny1,1:Ny1), Ey_invFi(1:Ny1,1:Ny1), &
                EzFi(1:Nz,1:Nz),   Ez_invFi(1:Nz,1:Nz), &
                LambxFi(1:Nx), LambyFi(1:Ny1), LambzFi(1:Nz), &
                Nx, Ny1, Nz, 1.D0, 1.D0, 1.D0, 0.D0)
        Call solve_eigen_decomp_d(PotentialHandle, x_gpu, rhs)
        Call report('Potential  ', x_cpu, x_gpu, Nx, Ny1, Nz, potSingular)
        Deallocate(rhs, x_cpu, x_gpu)

        Write(*,*) ' ==============================================================='
        Write(*,*) ''
    End Subroutine Validate_GPU_Solvers

    ! Deterministic, mesh-independent pseudo-random RHS in Alex's native
    ! (X,Y,Z) layout -- same array serves both the CPU and GPU solve calls.
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

    Subroutine report(name, xc, xg, N1, N2, N3, demean)
        Character(len=*), Intent(in) :: name
        Integer, Intent(in) :: N1, N2, N3
        Real(kind=8), Intent(in) :: xc(1:N1,1:N2,1:N3)
        Real(kind=8), Intent(in) :: xg(1:N1,1:N2,1:N3)
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
                    dg = xg(i,j,k) - gmean
                    err = Max(err, Abs(dg - dc))
                    nrm = Max(nrm, Abs(dc))
                End Do
            End Do
        End Do
        Write(*,'(A,A,ES12.4,A,ES12.4,A)') '   ', name//'  max|cpu-gpu| = ', err, &
              '   relative = ', err/nrm, Merge('  (de-meaned)', '             ', demean)
    End Subroutine report

End Module ValidateGPU
