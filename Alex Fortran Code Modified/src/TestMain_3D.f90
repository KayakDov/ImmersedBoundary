! ************************************************************
! *  Main program for unsteady convection                    *
! *            in rectangular cavities                       *
! *                                                          *
! ************************************************************

         Use Numbers
         Use Grid
         Use Variables
         Use EVD_Operators
         use Thomas_coefficients
         use eigenbcgsolver_eigen_mod
         use iso_c_binding, only : C_SIZE_T

        Implicit Real(kind=8) (A-H,O-Z)

        Integer ::  omp_get_max_threads
        
        Real(kind=4), Dimension(2) :: Time
        Real(kind=4)               :: t0, t1

! ----------------------------------------------------------------------
! GPU solver handles (CudaBandedLib). See SOLVER_REPLACEMENT_NOTES.md for
! the full reasoning behind everything in this block.
!
! Alex's original code ran four CPU routines that reduce to two distinct
! algorithms (a dense-eigenvector solve in X, and a tridiagonal/Thomas
! solve in X); CudaBandedLib's "thomas" flag selects between exactly
! those same two algorithms at solver-creation time, so one handle is
! created per algorithm rather than one per original routine.
! ----------------------------------------------------------------------
        integer(C_SIZE_T) :: denseSolverHandle   ! thomas=.false., replaces EVDmethod / EVDmethod1
        integer(C_SIZE_T) :: thomasSolverHandle  ! thomas=.true.,  replaces EVD_Thomas / EVD_Thomas1

! ----------------------------------------------------------------------
! Boundary condition / boundary value choices made for this replacement.
!
! Axis/face mapping for CudaBandedLib's init_eigen_decomp_d:
!   left/right = X axis,  top/bottom = Y axis,  front/back = Z axis
! ----------------------------------------------------------------------
        logical :: EVD_BCxLeft, EVD_BCxRight    ! X axis (left/right)
        logical :: EVD_BCyTop,  EVD_BCyBottom   ! Y axis (top/bottom)
        logical :: EVD_BCzFront, EVD_BCzBack    ! Z axis (front/back)
        real(kind=8) :: homogeneousBoundaryValue

! ======================================================================

        Nn = 256
        Nx = Nn;  Ny=Nn;  Nz=Nn
        Nx1 = Nx + 1
        Nx2 = Nx + 2
        Nx3 = Nx + 3
        Ny1 = Ny + 1
        Ny2 = Ny + 2
        Ny3 = Ny + 3
        Nz1 = Nz + 1
        Nz2 = Nz + 2
        Nz3 = Nz + 3
        
! ............ Make grid ...........................
        
        Call   Mesh

! .................. Make eigenvalues decompositions ...................
!
! Removed: Call EVDLapTmpr
!
! EVDLapTmpr's entire job was building the per-axis Laplacian operator
! and running LAPACK DGEEV on it to produce ExTemp/EyTemp/EzTemp,
! Ex_invTemp/Ey_invTemp/Ez_invTemp, and LambxTemp/LambyTemp/LambzTemp.
! init_eigen_decomp_d below does this same eigendecomposition internally
! on the GPU, so this CPU setup step - and the large dense eigenvector
! matrices it produced - are no longer needed at all.

! ----------------------------------------------------------------------
! Boundary conditions: Neumann on all six faces (see notes above).
! ----------------------------------------------------------------------
        EVD_BCxLeft   = .true.;  EVD_BCxRight  = .true.
        EVD_BCyTop    = .true.;  EVD_BCyBottom = .true.
        EVD_BCzFront  = .true.;  EVD_BCzBack   = .true.
        homogeneousBoundaryValue = 0.0d0

! ----------------------------------------------------------------------
! Initialize the two GPU solver handles. Grid spacing is passed as the
! full per-axis array (HPx/HPy/HPz, the staggered-point spacing used in
! the original Laplacian stencil) with uniformDeltaX/Y/Z = .false., so
! this keeps working unmodified if/when MeshStretch.f90's stretching
! terms (a, b, c) are re-enabled and the mesh becomes genuinely
! non-uniform. Today they happen to be uniform (a=b=c=0), but that's
! incidental, not relied upon here.
! ----------------------------------------------------------------------

        denseSolverHandle = init_eigen_decomp_d( &
                int(Ny1, C_SIZE_T), &
&           int(Nx1, C_SIZE_T), &
            int(Nz1, C_SIZE_T), &
            HPy(0:Ny1), &
&           HPx(0:Nx1), &
            HPz(0:Nz1), &
&           .false., .false., .false., &
&           EVD_BCxLeft, EVD_BCxRight, EVD_BCyTop, EVD_BCyBottom, &
&           EVD_BCzBack, EVD_BCzFront, &
&           homogeneousBoundaryValue, homogeneousBoundaryValue, &
&           homogeneousBoundaryValue, homogeneousBoundaryValue, &
&           homogeneousBoundaryValue, homogeneousBoundaryValue, &
&           .false., .false.)

        thomasSolverHandle = init_eigen_decomp_d( &
&           int(Ny1, C_SIZE_T), int(Nx1, C_SIZE_T),  int(Nz1, C_SIZE_T), &
&           HPy(0:Ny1), HPx(0:Nx1), HPz(0:Nz1), &
&           .false., .false., .false., &
&           EVD_BCxLeft, EVD_BCxRight, EVD_BCyTop, EVD_BCyBottom, &
&           EVD_BCzBack, EVD_BCzFront, &
&           homogeneousBoundaryValue, homogeneousBoundaryValue, &
&           homogeneousBoundaryValue, homogeneousBoundaryValue, &
&           homogeneousBoundaryValue, homogeneousBoundaryValue, &
&           .false., .true.)

! ....1. Make r.h.s. .....

         ! Instead of a uniform 1.d0, create a linear gradient along the X-axis.
         ! This antisymmetric initialization ensures the domain sums exactly to 0.0,
         ! satisfying the compatibility condition for a singular Neumann system.

         do k = 1, Nz1
                 do j = 1, Ny1
                         do i = 1, Nx1
                                 FDRHP(i,j,k) = dble(i) - (dble(Nx1 + 1) / 2.0d0)
                         end do
                 end do
         end do
! .......  Define number of CPUs ..........

        Write (*,*) ' How many CPUs?'
        Read (*,*) Ncpus
        
        Call omp_set_num_threads(Ncpus)
        Write (*,*) omp_get_max_threads(), '  CPUs will be used'

! ...........  Run dense-eigenvector solve (GPU, thomas=.false.) ............................
! Replaces: EVDmethod and EVDmethod1 (numerically identical to each
! other in the original code - see SOLVER_REPLACEMENT_NOTES.md).
 
        t0 = etime(Time);  t0 = Time(1)

        Call solve_eigen_decomp_d(denseSolverHandle, &
&           TmpOld(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1))

        t1 = etime(Time); t1 = Time(1)
        Write (*,*) '  GPU dense-eigenvector solve worked ',t1-t0, ' secs', &
&           '   Tmp=',Sum(abs(TmpOld(1:Nx1,1:Ny1,1:Nz1)))        
        
! ...........  Run Thomas solve (GPU, thomas=.true.) ........................................
! Replaces: EVD_Thomas and EVD_Thomas1 (numerically identical to each
! other in the original code - see SOLVER_REPLACEMENT_NOTES.md).

        t0 = etime(Time);  t0 = Time(1)

        Call solve_eigen_decomp_d(thomasSolverHandle, &
&           TmpNew(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1))

        t1 = etime(Time); t1 = Time(1)
        Write (*,*) '  GPU Thomas solve worked ',t1-t0, ' secs', &
&           '   Tmp=',Sum(abs(TmpNew(1:Nx1,1:Ny1,1:Nz1)))    
        Write (*,*) ' Diff=', maxval(abs( TmpNew(1:Nx1,1:Ny1,1:Nz1) - TmpOld(1:Nx1,1:Ny1,1:Nz1) ))

! ----------------------------------------------------------------------
! Release GPU resources for both solver handles.
! ----------------------------------------------------------------------
        Call finalize_eigen_decomp_d()

        Stop
        End
