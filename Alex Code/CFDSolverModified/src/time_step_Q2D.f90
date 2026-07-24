! ************************************************************
! *   Subroutine for straight-forward solution  of           *     
! *       FD problem for convection in rectangulars          *
! *                                                          *
! *   This is version with 3-level time integrator           *
! *                                                          *
! ************************************************************

        Subroutine  TimeStep ( Istp, RNSx, RNSy, RNSz, RTmpr, RDP )

         Use Numbers
         Use Parameters
         Use Numerica
         Use Grid
         Use Operators
         Use Variables
         Use AlexCudaCompatibility, only : TemperatureHandle, VxHandle, VyHandle, VzHandle, &
                 PressureHandle, GrPr
         Use eigenbcgsolver_eigen_mod, only : solve_eigen_decomp_d

        Implicit Real(kind=8) (A-H,O-Z)

        ! Small scaled-RHS scratch buffers: the CPU solvers absorbed the
        ! GrPr/DGr operator scaling into their own dedicated coefficient
        ! arrays (T_left/T_center/..., Vx_left/..., built once in
        ! EVDLapTmpr/EVDLapVx/y/z); the GPU operator is built un-scaled
        ! instead, so the RHS is scaled here to match (see
        ! AlexCudaCompatibility.f90's Initialize_GPU_Solvers for the exact
        ! shift/scale derivation). No index reordering -- these are plain
        ! same-shape, same-order copies of FDRHP/RHSx/RHSy/RHSz.
        Real(kind=8), Allocatable, Save :: GPU_RHS_T(:,:,:)
        Real(kind=8), Allocatable, Save :: GPU_RHS_Vx(:,:,:), GPU_RHS_Vy(:,:,:), GPU_RHS_Vz(:,:,:)
        Logical, Save :: GPU_RHS_Allocated = .false.
	 
          Ht = 2.D0 * Htime

          If (.not. GPU_RHS_Allocated) Then
              Allocate( GPU_RHS_T (1:Nx1,1:Ny1,1:Nz1) )
              Allocate( GPU_RHS_Vx(1:Nx ,1:Ny1,1:Nz1) )
              Allocate( GPU_RHS_Vy(1:Nx1,1:Ny ,1:Nz1) )
              Allocate( GPU_RHS_Vz(1:Nx1,1:Ny1,1:Nz ) )
              GPU_RHS_Allocated = .true.
          End If
        
          dt_temp = Dble(Istat) 
          
! ########### Time step for temperature ##################
          
!          Write (*,*) ' time step entered', dt_temp

! +++++++++++ Right hand side ++++++++++++++++++++++++++++++++
        
           FDRHP = 0.D0
           Call  VgrTmp
!           Write (*,*) '1 FDRHP =', Sum(abs(FDRHP))

! +++++++++++ Time derivative +++++++++++++++++++++++++++++++
           
           FDRHP(1:Nx1,1:Ny1,1:Nz1) = FDRHP(1:Nx1,1:Ny1,1:Nz1) -          &
     &              dt_temp * ( 4.D0 * Tmpr(1:Nx1,1:Ny1,1:Nz1) -          &
     &                               TmpOld(1:Nx1,1:Ny1,1:Nz1)     )  / Ht 

 !          Write (*,*) '2 FDRHP =', Maxval(abs(FDRHP))

! ......... Solution ........................................................

           ! GPU: (L - shift*I) x = b, shift and GrPr scaling built in
           ! Initialize_GPU_Solvers (see AlexCudaCompatibility.f90). Was:
           ! Call EVD_Thomas(..., dt_temp) with the CPU's own GrPr-scaled
           ! T_left/T_center/T_right coefficients.
           GPU_RHS_T = FDRHP(1:Nx1,1:Ny1,1:Nz1) * GrPr
           Call solve_eigen_decomp_d(TemperatureHandle, TmpNew(1:Nx1,1:Ny1,1:Nz1), GPU_RHS_T)
           
 !          Write (*,*) ' TmpNew=', Maxval(abs(TMpNew))

      Call EVDbounds 

      RTmpr = Dist2D (TmpNew, Tmpr, Nx2, Ny2, Nz2, Nx2, Ny2, Nz2)
!      Write (*,*) ' RTmpr=', RTmpr, Minval(TmpNew), Maxval(tmpnew)
!        Write (*,*) 'DT=', Maxval(abs(TmpNew - Tmpr) ), Maxloc(abs(TmpNew - Tmpr) ), Maxval(abs(TmpNew(:,:,102)))
!stop

! ########### Inversing the Stokes operator #################

           Call Get_Potential
      
! ======== Make r.h.s. of momentum equations ==================

          FDRHP = 0.D0
          Call   VgrdVx 
          Call   GradPx( RHSx(1:Nx ,1:Ny1,1:Nz1), Prs )
          RHSx(1:Nx,1:Ny1,1:Nz1) = RHSx(1:Nx,1:Ny1,1:Nz1) + FDRHP(1:Nx,1:Ny1,1:Nz1)

          FDRHP = 0.D0
          Call   VgrdVy 
          Call   GradPy( RHSy(1:Nx1,1:Ny ,1:Nz1), Prs )
          RHSy(1:Nx1,1:Ny,1:Nz1) = RHSy(1:Nx1,1:Ny,1:Nz1) + FDRHP(1:Nx1,1:Ny,1:Nz1)

          FDRHP = 0.D0
          Call   VgrdVz 
          Call   GradPz( RHSz(1:Nx1,1:Ny1,1:Nz ), Prs )
          RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz) + FDRHP(1:Nx1,1:Ny1,1:Nz)

! .............. Add bouyancy force ......................
 
           RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz)  &
     &     - 0.5d0 * Bu_Gr * ( TmpNew(1:Nx1,1:Ny1,1:Nz) + TmpNew(1:Nx1,1:Ny1,2:Nz1) ) &
     &     - 0.5d0 * Bu_Gr *  (   Teta(1:Nx1,1:Ny1,1:Nz) +   Teta(1:Nx1,1:Ny1,2:Nz1) ) 
    
! +++++++++++ Straight-forward step +++++++++++++++++++++          

           RHSx(1:Nx,1:Ny1,1:Nz1) = RHSx(1:Nx,1:Ny1,1:Nz1) - &
     &                  ( 4.D0 * VMx(1:Nx,1:Ny1,1:Nz1) - VMxOld(1:Nx,1:Ny1,1:Nz1) )/ Ht

           RHSy(1:Nx1,1:Ny,1:Nz1) = RHSy(1:Nx1,1:Ny,1:Nz1) - &
     &                  ( 4.D0 * VMy(1:Nx1,1:Ny,1:Nz1) - VMyOld(1:Nx1,1:Ny,1:Nz1) )/ Ht
 
           RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz) - &
     &                  ( 4.D0 * VMz(1:Nx1,1:Ny1,1:Nz) - VMzOld(1:Nx1,1:Ny1,1:Nz) )/ Ht
           
! +++++++++ Electromagnetic force ++++++++++++++
          
           Call EM_force

! ++++++++++++ [Calculate Lap(u)^-1]*RHSx +++++++++++++++++++++++
      
      ! GPU: was Call EVD_Thomas(..., 1.D0) with the CPU's own
      ! DGr-scaled Vx_left/Vx_center/Vx_right coefficients.
      GPU_RHS_Vx = RHSx(1:Nx,1:Ny1,1:Nz1) / DGr
      Call solve_eigen_decomp_d(VxHandle, VMxNew(1:Nx,1:Ny1,1:Nz1), GPU_RHS_Vx)
	     
! ++++++++++++++ [Calculate Lap(v)^-1]*RHSy ++++++++++++++++++++++++++
      
      ! GPU: was Call EVD_Thomas(..., 1.D0), same DGr scaling as Vx.
      GPU_RHS_Vy = RHSy(1:Nx1,1:Ny,1:Nz1) / DGr
      Call solve_eigen_decomp_d(VyHandle, VMyNew(1:Nx1,1:Ny,1:Nz1), GPU_RHS_Vy)

! ++++++++++++++ [Calculate Lap(w)^-1]*RHSz ++++++++++++++++++++++++++
      
      ! GPU: was Call EVD_Thomas(..., 1.D0), same DGr scaling as Vx/Vy.
      GPU_RHS_Vz = RHSz(1:Nx1,1:Ny1,1:Nz) / DGr
      Call solve_eigen_decomp_d(VzHandle, VMzNew(1:Nx1,1:Ny1,1:Nz), GPU_RHS_Vz)

      Call EVDbounds 
         
 !        RNSx = Dist2D (VMx, VMxNew, Nx1, Ny2, Nz2, Nx1, Ny2, Nz2)
 !        RNSy = Dist2D (VMy, VMyNew, Nx2, Ny1, Nz2, Nx2, Ny1, Nz2)
 !        RNSz = Dist2D (VMz, VMzNew, Nx2, Ny2, Nz1, Nx2, Ny2, Nz1)
 !                 
 !        Write (*,*) ' RNS=', RNSx, RNSy, RNSz
!stop

! ++++++++ Calcualte pressure correction ++++++++++++

      FDRHP= 0.d0
      
      Call FdDiv
  
      FDRHP = FDRHP * Ckor / Htime
      
  !    Write (*,*) ' FDRHP=', Sum(abs(FDRHP))

!      Call   EVD_Thomas (Dprs(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1),  &
!     &                   EyP(1:Ny1,1:Ny1), Ey_invP(1:Ny1,1:Ny1),             &
!     &                   EzP(1:Nz1,1:Nz1), Ez_invP(1:Nz1,1:Nz1),             &
!     &                   LambyP(1:Ny1), LambzP(1:Nz1),                       &
!     &                   P_left(1:Nx1), P_center(1:Nx1), P_right(1:Nx1),     &
!     &                   Nx1, Ny1, Nz1, 0.D0)

       ! GPU: pure Poisson (shift = 0, alpha = 1 on all axes -- no RHS
       ! scaling needed), was Call EVDmethod(..., 1,1,1, beta=0).
       Call solve_eigen_decomp_d(PressureHandle, Dprs(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1))
 !   Write (*,*) ' Dprs=', Sum(abs(Dprs))

! ++++++++++++ Calculate velocities ++++++++++++++++++++++++++++++

! ........... calculate grad(Prs) .....................

      Call GradPx( RHSx(1:Nx ,1:Ny1,1:Nz1), Dprs )
      Call GradPy( RHSy(1:Nx1,1:Ny ,1:Nz1), Dprs )
      Call GradPz( RHSz(1:Nx1,1:Ny1,1:Nz ), Dprs )
      
! ........... Correct velocities .........................................

      VMxNew(1:Nx,1:Ny1,1:Nz1) = VMxNew(1:Nx,1:Ny1,1:Nz1) - RHSx(1:Nx ,1:Ny1,1:Nz1) * Htime / Ckor
      VMyNew(1:Nx1,1:Ny,1:Nz1) = VMyNew(1:Nx1,1:Ny,1:Nz1) - RHSy(1:Nx1,1:Ny ,1:Nz1) * Htime / Ckor
      VMzNew(1:Nx1,1:Ny1,1:Nz) = VMzNew(1:Nx1,1:Ny1,1:Nz) - RHSz(1:Nx1,1:Ny1,1:Nz ) * Htime / Ckor

      Prs = Prs + DPrs

      call EVDbounds 

         RNSx = Dist2D (VMx, VMxNew, Nx1, Ny2, Nz2, Nx1, Ny2, Nz2)
         RNSy = Dist2D (VMy, VMyNew, Nx2, Ny1, Nz2, Nx2, Ny1, Nz2)
         RNSz = Dist2D (VMz, VMzNew, Nx2, Ny2, Nz1, Nx2, Ny2, Nz1)
         
         RDP = MaxVal(Abs(Dprs) )
         
 !     FDRHP= 0.d0     
 !     Call FdDiv    
  !    Write (*,*) ' DivV=', Sum(abs(FDRHP))
 !        Write (*,*) RNSx, RNSy, RNSz, RDP
 !        stop

! ######## Check of results #############################

      FDRHP= 0.d0
      Call FdDiv

        If (Icheck .EQ. 0)  Call    Check
                    
! ######## Shift of the time step ########################

444     		Continue

          VMxOld = VMx
          VMyOld = VMy
          VMzOld = VMz
          TmpOld = Tmpr

          VMx  = VMxNew
          VMy  = VMyNew
          VMz  = VMzNew
          Tmpr = TmpNew

        Return
        End
        
