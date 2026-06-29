! ************************************************************
! *  Benchmark: CUDA TPF vs CUDA TPT sweep over grid sizes  *
! *  Replaces the CPU EVDLapTmpr + EVDmethod / EVD_Thomas    *
! *  calls with init_eigen_decomp_d / solve_eigen_decomp_d  *
! *  from CudaBandedLib.                                     *
! *                                                          *
! *  Columns: N, N^3, TPF(ms), TPT(ms),                     *
! *           ||L*x_TPF - b||, ||L*x_TPT - b||              *
! ************************************************************

         Use Numbers
         Use Grid
         Use Variables
         Use Thomas_coefficients
         Use eigenbcgsolver_eigen_mod
         Use iso_c_binding, only : C_SIZE_T, C_BOOL, C_DOUBLE
         Use omp_lib

         Implicit Real(kind=8) (A-H,O-Z)

         Integer(C_SIZE_T) :: solverTPF, solverTPT
         Integer(kind=8)   :: t_start, t_end, t_rate
         Real(kind=8)      :: tpf_ms, tpt_ms, res_tpf, res_tpt
         Integer           :: Nn, Ncpus, iunit
         Parameter         (iunit = 20)

         ! dx/dy/dz arrays passed to init: HPx/HPy/HPz are the
         ! dual-cell widths (cell-centre spacings) on [0..Nxx1].
         ! The solver expects size = axis_dim + 1 for non-uniform grids,
         ! which matches exactly: HPx has indices 0..Nx1, i.e. Nx1+1 entries.
         ! We pass uniformDeltaX/Y/Z = .false. so the full arrays are used.

! ======================================================================
! One-time setup
! ======================================================================

         AspRa  = 1.d0
         WidRa  = 1.d0
         EVD_BCx = 0
         EVD_BCy = 0
         EVD_BCz = 0

         FDRHP = 1.d0

         Write (*,*) ' How many CPUs?'
         Read  (*,*) Ncpus
         Call omp_set_num_threads(Ncpus)
         Write (*,'(A,I3,A)') '  Using ', omp_get_max_threads(), ' CPUs'
         Write (*,*) ' Writing results to benchmark_results.txt ...'

! ======================================================================
! Open output file
! ======================================================================

         Open(unit=iunit, file='benchmark_results.txt', &
              status='replace', action='write')

         Write (iunit,'(A6,A15,A20,A22,A25,A25)') &
               'N', 'N^3', 'TPF (ms)', 'TPT (ms)', &
               '||L*x_TPF - b||', '||L*x_TPT - b||'
         Write (iunit,'(A113)') Repeat('-', 113)

         Call system_clock(count_rate=t_rate)

! ======================================================================
! Sweep loop: n=2,4,...,198, 200,201,...,350
! ======================================================================

         Nn = 2
         Do While (Nn <= 350)

            Nx  = Nn;  Ny  = Nn;  Nz  = Nn
            Nx1 = Nx+1; Nx2 = Nx+2
            Ny1 = Ny+1; Ny2 = Ny+2
            Nz1 = Nz+1; Nz2 = Nz+2

            Call Mesh

            ! ----------------------------------------------------------
            ! Build the Thomas tridiagonal coefficients for ComputeResidual.
            ! This replicates the coefficient-filling part of EVDLapTmpr
            ! (the eigenvector computation is no longer needed here).
            ! ----------------------------------------------------------
            Do i = 1, Nx1
               P1 = 1.D0 / ( Hx12(i-1) * HPx(i-1) )
               P2 = 1.D0 / ( Hx12(i-1) * HPx( i ) )
               If (i /= 1)   T_left(i)   = P1
               If (i /= Nx1) T_right(i)  = P2
                             T_center(i) = -(P1+P2)
               If (EVD_BCx == 1) Then
                  If (i == 1)   T_center(i) = -P2
                  If (i == Nx1) T_center(i) = -P1
               End If
            End Do

            ! ----------------------------------------------------------
            ! Initialise two CUDA eigendecomposition solvers:
            !   solverTPF  -- full DGEMM back-transform  (thomas=.false.)
            !   solverTPT  -- Thomas x-direction variant (thomas=.true.)
            !
            ! HPx(0..Nx1) has Nx1+1 entries  ->  rows  = Nx1, dx size = Nx1+1
            ! HPy(0..Ny1) has Ny1+1 entries  ->  cols  = Ny1, dy size = Ny1+1
            ! HPz(0..Nz1) has Nz1+1 entries  ->  layers= Nz1, dz size = Nz1+1
            !
            ! Boundary conditions: EVD_BCx==0 => Dirichlet on both sides.
            ! The library takes separate left/right flags; both are .false.
            ! for Dirichlet (EVD_BCx==0) or both .true. for Neumann (==1).
            ! ----------------------------------------------------------

            solverTPF = init_eigen_decomp_d( &
                 rows   = int(Nx1, C_SIZE_T), &
                 cols   = int(Ny1, C_SIZE_T), &
                 layers = int(Nz1, C_SIZE_T), &
                 dx = HPx(0:Nx1), &
                 dy = HPy(0:Ny1), &
                 dz = HPz(0:Nz1), &
                 uniformDeltaX = .false., &
                 uniformDeltaY = .false., &
                 uniformDeltaZ = .false., &
                 leftIsNeumann   = (EVD_BCx == 1), &
                 rightIsNeumann  = (EVD_BCx == 1), &
                 topIsNeumann    = (EVD_BCy == 1), &
                 bottomIsNeumann = (EVD_BCy == 1), &
                 backIsNeumann   = (EVD_BCz == 1), &
                 frontIsNeumann  = (EVD_BCz == 1), &
                 leftVal   = 0.d0, rightVal  = 0.d0, &
                 topVal    = 0.d0, bottomVal = 0.d0, &
                 frontVal  = 0.d0, backVal   = 0.d0, &
                 isStaggered = .false., &
                 thomas      = .false.)

            solverTPT = init_eigen_decomp_d( &
                 rows   = int(Nx1, C_SIZE_T), &
                 cols   = int(Ny1, C_SIZE_T), &
                 layers = int(Nz1, C_SIZE_T), &
                 dx = HPx(0:Nx1), &
                 dy = HPy(0:Ny1), &
                 dz = HPz(0:Nz1), &
                 uniformDeltaX = .false., &
                 uniformDeltaY = .false., &
                 uniformDeltaZ = .false., &
                 leftIsNeumann   = (EVD_BCx == 1), &
                 rightIsNeumann  = (EVD_BCx == 1), &
                 topIsNeumann    = (EVD_BCy == 1), &
                 bottomIsNeumann = (EVD_BCy == 1), &
                 backIsNeumann   = (EVD_BCz == 1), &
                 frontIsNeumann  = (EVD_BCz == 1), &
                 leftVal   = 0.d0, rightVal  = 0.d0, &
                 topVal    = 0.d0, bottomVal = 0.d0, &
                 frontVal  = 0.d0, backVal   = 0.d0, &
                 isStaggered = .false., &
                 thomas      = .true.)

! ---------- Time TPF: CUDA solve (full DGEMM back-transform) ----------

            Call system_clock(t_start)

            Call solve_eigen_decomp_d( &
                 solverHandle = solverTPF, &
                 x = TmpOld(1:Nx1, 1:Ny1, 1:Nz1), &
                 b = FDRHP(1:Nx1, 1:Ny1, 1:Nz1))

            Call system_clock(t_end)
            tpf_ms = Real(t_end - t_start, 8) / Real(t_rate, 8) * 1.d3

! ---------- Time TPT: CUDA solve (Thomas x-direction variant) ---------

            Call system_clock(t_start)

            Call solve_eigen_decomp_d( &
                 solverHandle = solverTPT, &
                 x = TmpNew(1:Nx1, 1:Ny1, 1:Nz1), &
                 b = FDRHP(1:Nx1, 1:Ny1, 1:Nz1))

            Call system_clock(t_end)
            tpt_ms = Real(t_end - t_start, 8) / Real(t_rate, 8) * 1.d3

! ---------- Residual norms -------------------------------------------

            Call ComputeResidual( &
                 TmpOld(1:Nx1,1:Ny1,1:Nz1), &
                 FDRHP(1:Nx1,1:Ny1,1:Nz1), &
                 res_tpf, Nx1, Ny1, Nz1)

            Call ComputeResidual( &
                 TmpNew(1:Nx1,1:Ny1,1:Nz1), &
                 FDRHP(1:Nx1,1:Ny1,1:Nz1), &
                 res_tpt, Nx1, Ny1, Nz1)

! ---------- Release GPU resources for this grid size -----------------
! finalize_eigen_decomp_d() releases ALL handles at once; call it here
! at the end of each grid-size iteration so GPU memory doesn't pile up.

            Call finalize_eigen_decomp_d()

! ---------- Print row and advance ------------------------------------

            Write (iunit, '(I6,I15,F20.3,F22.3,E25.6,E25.6)') &
                  Nn, Nn**3, tpf_ms, tpt_ms, res_tpf, res_tpt

            Flush(iunit)

            If (Nn < 200) Then
               Nn = Nn + 2
            Else
               Nn = Nn + 1
            End If

         End Do

         Close(iunit)
         Write (*,*) ' Done.'

         Stop
         End
