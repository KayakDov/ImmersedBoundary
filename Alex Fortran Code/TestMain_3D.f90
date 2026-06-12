! ************************************************************
! *  Benchmark: TPF vs TPT sweep over grid sizes 2..350     *
! *  Table written to benchmark_results.txt                 *
! *  Columns: N, N^3, TPF(ms), TPT(ms),                    *
! *           ||L*x_TPF - b||, ||L*x_TPT - b||             *
! ************************************************************

         Use Numbers
         Use Grid
         Use Variables
         Use EVD_Operators
         Use Thomas_coefficients
         Use omp_lib

         Implicit Real(kind=8) (A-H,O-Z)

         Integer(kind=8) :: t_start, t_end, t_rate
         Real(kind=8)    :: tpf_ms, tpt_ms, res_tpf, res_tpt
         Integer         :: Nn, Ncpus, iunit
         Parameter       (iunit = 20)

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

! ---------- Time TPF: EVDLapTmpr + EVDmethod -------------------------

            Call system_clock(t_start)

            Call EVDLapTmpr

            Call EVDmethod( &
                 TmpOld(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1), &
                 ExTemp(1:Nx1,1:Nx1), Ex_invTemp(1:Nx1,1:Nx1), &
                 EyTemp(1:Ny1,1:Ny1), Ey_invTemp(1:Ny1,1:Ny1), &
                 EzTemp(1:Nz1,1:Nz1), Ez_invTemp(1:Nz1,1:Nz1), &
                 LambxTemp(1:Nx1), LambyTemp(1:Ny1), LambzTemp(1:Nz1), &
                 Nx1, Ny1, Nz1, 1.d0, 1.d0, 1.d0, 0.d0)

            Call system_clock(t_end)
            tpf_ms = Real(t_end - t_start, 8) / Real(t_rate, 8) * 1.d3

! ---------- Time TPT: EVDLapTmpr + EVD_Thomas -------------------------

            Call system_clock(t_start)

            Call EVDLapTmpr

            Call EVD_Thomas( &
                 TmpNew(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1), &
                 EyTemp(1:Ny1,1:Ny1), Ey_invTemp(1:Ny1,1:Ny1), &
                 EzTemp(1:Nz1,1:Nz1), Ez_invTemp(1:Nz1,1:Nz1), &
                 LambyTemp(1:Ny1), LambzTemp(1:Nz1), &
                 T_left, T_center, T_right, Nx1, Ny1, Nz1, 1.d0)

            Call system_clock(t_end)
            tpt_ms = Real(t_end - t_start, 8) / Real(t_rate, 8) * 1.d3

! ---------- Residual norms -------------------------------------------
! T_left/T_center/T_right hold the x-direction Laplacian coefficients
! from the most recent EVDLapTmpr call.  For a uniform isotropic grid
! (Nx=Ny=Nz, equal spacing in all directions) these are identical to
! the y and z direction coefficients, so ComputeResidual reuses them
! for all three directions.

            Call ComputeResidual( &
                 TmpOld(1:Nx1,1:Ny1,1:Nz1), &
                 FDRHP(1:Nx1,1:Ny1,1:Nz1), &
                 res_tpf, Nx1, Ny1, Nz1)

            Call ComputeResidual( &
                 TmpNew(1:Nx1,1:Ny1,1:Nz1), &
                 FDRHP(1:Nx1,1:Ny1,1:Nz1), &
                 res_tpt, Nx1, Ny1, Nz1)

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
