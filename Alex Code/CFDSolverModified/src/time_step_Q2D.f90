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
         Use eigenbcgsolver_eigen_mod, only : solve_eigen_decomp_d, synch_d

        Implicit Real(kind=8) (A-H,O-Z)

        ! Small scaled-RHS scratch buffers: the CPU solvers absorbed the
        ! GrPr/DGr operator scaling into their own dedicated coefficient
        ! arrays (T_left/T_center/..., Vx_left/..., built once in
        ! EVDLapTmpr/EVDLapVx/y/z); the GPU operator is built un-scaled
        ! instead, so the RHS is scaled here to match (see
        ! AlexCudaCompatibility.f90's Initialize_GPU_Solvers for the exact
        ! shift/scale derivation). No index reordering -- these are plain
        ! same-shape, same-order copies of FDRHP/RHSx/RHSy/RHSz.
         ! GPU_SOL_T/Vx/Vy/Vz are still needed: TmpNew/VMxNew/VMyNew/VMzNew are
         ! all ghost-padded (e.g. TmpNew is (0:Nx2,0:Ny2,0:Nz2) but the solve
         ! region is (1:Nx1,1:Ny1,1:Nz1)), so passing a slice of them directly
         ! to synch_d would be non-contiguous and force a hidden, non-pinned
         ! compiler temporary. GPU_SOL_P is NOT needed: Dprs's declared shape
         ! (1:Nx1,1:Ny1,1:Nz1) is an EXACT match to the pressure solve region,
         ! so synch_d can write directly into Dprs with no staging buffer.
         Real(kind=8), Allocatable, Save :: GPU_RHS_T(:,:,:), GPU_SOL_T(:,:,:)
         Real(kind=8), Allocatable, Save :: GPU_RHS_Vx(:,:,:), GPU_RHS_Vy(:,:,:), GPU_RHS_Vz(:,:,:)
         Real(kind=8), Allocatable, Save :: GPU_SOL_Vx(:,:,:), GPU_SOL_Vy(:,:,:), GPU_SOL_Vz(:,:,:)
         Real(kind=8), Allocatable, Save :: GPU_RHS_P(:,:,:)
         Logical, Save :: GPU_RHS_Allocated = .false.

         ! Loop counters / per-array bounds for the explicit-loop rewrites
         ! below (see the comments at each site). Implicit typing would
         ! already make these Integer (i,j,k fall in the I-N range), but
         ! they're declared explicitly for clarity since they're new.
         Integer :: i, j, k, iLo, iHi, jLo, jHi, kLo, kHi
	 
          Ht = 2.D0 * Htime

         If (.not. GPU_RHS_Allocated) Then
             Allocate( GPU_RHS_T (1:Nx1,1:Ny1,1:Nz1) )
             Allocate( GPU_SOL_T (1:Nx1,1:Ny1,1:Nz1) )
             Allocate( GPU_RHS_Vx(1:Nx ,1:Ny1,1:Nz1) )
             Allocate( GPU_SOL_Vx(1:Nx ,1:Ny1,1:Nz1) )
             Allocate( GPU_RHS_Vy(1:Nx1,1:Ny ,1:Nz1) )
             Allocate( GPU_SOL_Vy(1:Nx1,1:Ny ,1:Nz1) )
             Allocate( GPU_RHS_Vz(1:Nx1,1:Ny1,1:Nz ) )
             Allocate( GPU_SOL_Vz(1:Nx1,1:Ny1,1:Nz ) )
             Allocate( GPU_RHS_P (1:Nx1,1:Ny1,1:Nz1) )
             ! (no GPU_SOL_P -- Dprs itself is the contiguous destination)
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

! ......... Launch (async -- solve_eigen_decomp_d no longer synchs) .......

           ! GPU: (L - shift*I) x = b, shift and GrPr scaling built in
           ! Initialize_GPU_Solvers (see AlexCudaCompatibility.f90). Was:
           ! Call EVD_Thomas(..., dt_temp) with the CPU's own GrPr-scaled
           ! T_left/T_center/T_right coefficients.
           ! Not synched here: TmpNew isn't needed until EVDbounds below,
           ! so Vy's and Potential's independent right-hand sides are built
           ! (and their own solves launched) while this one is still running.
         GPU_RHS_T = FDRHP(1:Nx1,1:Ny1,1:Nz1) * GrPr
         Call solve_eigen_decomp_d(TemperatureHandle, GPU_RHS_T)

! ========= RHS/launch for Vy -- depends on neither Temperature nor =======
! ========= Potential (unlike Vx/Vz, never touched by EM_force),   ========
! ========= so it can go out immediately.                          ========

          FDRHP = 0.D0
          Call   VgrdVy 
          Call   GradPy( RHSy(1:Nx1,1:Ny ,1:Nz1), Prs )
          RHSy(1:Nx1,1:Ny,1:Nz1) = RHSy(1:Nx1,1:Ny,1:Nz1) + FDRHP(1:Nx1,1:Ny,1:Nz1)

           RHSy(1:Nx1,1:Ny,1:Nz1) = RHSy(1:Nx1,1:Ny,1:Nz1) - &
     &                  ( 4.D0 * VMy(1:Nx1,1:Ny,1:Nz1) - VMyOld(1:Nx1,1:Ny,1:Nz1) )/ Ht

      ! GPU: was Call EVD_Thomas(..., 1.D0), same DGr scaling as Vx.
      ! Not synched here -- VMyNew isn't needed until the EVDbounds call
      ! after Vx/Vz below, by which point this has had the whole
      ! Temperature/Potential/Vx/Vz sequence to finish on the GPU.
         GPU_RHS_Vy = RHSy(1:Nx1,1:Ny,1:Nz1) / DGr
         Call solve_eigen_decomp_d(VyHandle, GPU_RHS_Vy)

! ========= RHS/launch for Potential -- only needs VMx/VMz from the =======
! ========= start of this step, so it's independent of Temperature  =======
! ========= and Vy above too.                                       =======

           Call Get_Potential_Launch

! ########### Temperature is needed now: synch #############################

         Call synch_d(TemperatureHandle, GPU_SOL_T)
         ! Was: TmpNew(1:Nx1,1:Ny1,1:Nz1) = GPU_SOL_T -- TmpNew is
         ! ghost-padded (0:Nx2,0:Ny2,0:Nz2, see ConvMain_3D_Q2D.f90), so
         ! this slice isn't contiguous. Same fix as Get_Potential_Finish:
         ! explicit loop instead of whole-array syntax on a module array,
         ! plus free OMP parallelism. Bounds unchanged (still 1:Nx1,1:Ny1,1:Nz1).
!$OMP Parallel Do Private(i,j,k)
         Do i = 1, Nx1
           Do j = 1, Ny1
             Do k = 1, Nz1
               TmpNew(i,j,k) = GPU_SOL_T(i,j,k)
             End Do
           End Do
         End Do
           
 !          Write (*,*) ' TmpNew=', Maxval(abs(TMpNew))

      Call EVDbounds 

      RTmpr = Dist2D (TmpNew, Tmpr, Nx2, Ny2, Nz2, Nx2, Ny2, Nz2)
!      Write (*,*) ' RTmpr=', RTmpr, Minval(TmpNew), Maxval(tmpnew)
!        Write (*,*) 'DT=', Maxval(abs(TmpNew - Tmpr) ), Maxloc(abs(TmpNew - Tmpr) ), Maxval(abs(TmpNew(:,:,102)))
!stop

! ======== Make the parts of RHSx/RHSz that don't need Potential yet ======

          FDRHP = 0.D0
          Call   VgrdVx 
          Call   GradPx( RHSx(1:Nx ,1:Ny1,1:Nz1), Prs )
          RHSx(1:Nx,1:Ny1,1:Nz1) = RHSx(1:Nx,1:Ny1,1:Nz1) + FDRHP(1:Nx,1:Ny1,1:Nz1)

          FDRHP = 0.D0
          Call   VgrdVz 
          Call   GradPz( RHSz(1:Nx1,1:Ny1,1:Nz ), Prs )
          RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz) + FDRHP(1:Nx1,1:Ny1,1:Nz)

! .............. Add bouyancy force (needs TmpNew, already synched above)

           RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz)  &
     &     - 0.5d0 * Bu_Gr * ( TmpNew(1:Nx1,1:Ny1,1:Nz) + TmpNew(1:Nx1,1:Ny1,2:Nz1) ) &
     &     - 0.5d0 * Bu_Gr *  (   Teta(1:Nx1,1:Ny1,1:Nz) +   Teta(1:Nx1,1:Ny1,2:Nz1) ) 
    
! +++++++++++ Straight-forward step +++++++++++++++++++++          

           RHSx(1:Nx,1:Ny1,1:Nz1) = RHSx(1:Nx,1:Ny1,1:Nz1) - &
     &                  ( 4.D0 * VMx(1:Nx,1:Ny1,1:Nz1) - VMxOld(1:Nx,1:Ny1,1:Nz1) )/ Ht

           RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz) - &
     &                  ( 4.D0 * VMz(1:Nx1,1:Ny1,1:Nz) - VMzOld(1:Nx1,1:Ny1,1:Nz) )/ Ht
           
! ########### Potential is needed now: synch (via Finish) ##################

           Call Get_Potential_Finish

! +++++++++ Electromagnetic force (needs Potential, just synched) ++++++++++
          
           Call EM_force

! ++++++++++++ [Calculate Lap(u)^-1]*RHSx +++++++++++++++++++++++
      
      ! GPU: was Call EVD_Thomas(..., 1.D0) with the CPU's own
      ! DGr-scaled Vx_left/Vx_center/Vx_right coefficients.
      ! Not synched here -- VMxNew isn't needed until the EVDbounds call
      ! below, after Vz is launched too.
         GPU_RHS_Vx = RHSx(1:Nx,1:Ny1,1:Nz1) / DGr
         Call solve_eigen_decomp_d(VxHandle, GPU_RHS_Vx)

! ++++++++++++++ [Calculate Lap(w)^-1]*RHSz ++++++++++++++++++++++++++
      
      ! GPU: was Call EVD_Thomas(..., 1.D0), same DGr scaling as Vx/Vy.
         GPU_RHS_Vz = RHSz(1:Nx1,1:Ny1,1:Nz) / DGr
         Call solve_eigen_decomp_d(VzHandle, GPU_RHS_Vz)

! ########### Vy, Vx, Vz are all needed now: synch each #####################

         Call synch_d(VyHandle, GPU_SOL_Vy)
         ! Was: VMyNew(1:Nx1,1:Ny,1:Nz1) = GPU_SOL_Vy -- same ghost-padding
         ! non-contiguity as TmpNew above; same fix.
!$OMP Parallel Do Private(i,j,k)
         Do i = 1, Nx1
           Do j = 1, Ny
             Do k = 1, Nz1
               VMyNew(i,j,k) = GPU_SOL_Vy(i,j,k)
             End Do
           End Do
         End Do

         Call synch_d(VxHandle, GPU_SOL_Vx)
         ! Was: VMxNew(1:Nx,1:Ny1,1:Nz1) = GPU_SOL_Vx
!$OMP Parallel Do Private(i,j,k)
         Do i = 1, Nx
           Do j = 1, Ny1
             Do k = 1, Nz1
               VMxNew(i,j,k) = GPU_SOL_Vx(i,j,k)
             End Do
           End Do
         End Do

         Call synch_d(VzHandle, GPU_SOL_Vz)
         ! Was: VMzNew(1:Nx1,1:Ny1,1:Nz) = GPU_SOL_Vz
!$OMP Parallel Do Private(i,j,k)
         Do i = 1, Nx1
           Do j = 1, Ny1
             Do k = 1, Nz
               VMzNew(i,j,k) = GPU_SOL_Vz(i,j,k)
             End Do
           End Do
         End Do

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
       ! Nothing independent is left to overlap with at this point in the
       ! step, so synch immediately -- Dprs is needed by the very next line.
         ! Dprs(1:Nx1,1:Ny1,1:Nz1) IS the array's whole declared extent (see
         ! the Allocate in ConvMain_3D_Q2D.f90) -- contiguous, so synch_d can
         ! write directly into it. No GPU_SOL_P needed (unlike Temperature/
         ! Vx/Vy/Vz/Potential, whose destination arrays are ghost-padded).
         GPU_RHS_P = FDRHP(1:Nx1,1:Ny1,1:Nz1)
         Call solve_eigen_decomp_d(PressureHandle, GPU_RHS_P)
         Call synch_d(PressureHandle, Dprs(1:Nx1,1:Ny1,1:Nz1))
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

          ! Was 8 separate bare whole-array copies:
          !   VMxOld = VMx;  VMyOld = VMy;  VMzOld = VMz;  TmpOld = Tmpr
          !   VMx = VMxNew;  VMy = VMyNew;  VMz = VMzNew;  Tmpr = TmpNew
          ! Bare colons/no bounds means each one walks its FULL declared
          ! (ghost-padded) extent -- the largest copies in this routine,
          ! done single-threaded every timestep. Same fix as elsewhere:
          ! explicit loops under OMP, with bounds read via LBound/UBound so
          ! this touches exactly the same elements as before. Each
          ! variable's Old<-Current and Current<-New steps are fused into
          ! one pass over (i,j,k): that's safe because for a given cell,
          ! the old value is fully read into *Old before that same cell of
          ! the variable is overwritten from *New, which is exactly the
          ! order the original two separate statements executed in.
          iLo = LBound(VMx,1);  iHi = UBound(VMx,1)
          jLo = LBound(VMx,2);  jHi = UBound(VMx,2)
          kLo = LBound(VMx,3);  kHi = UBound(VMx,3)
!$OMP Parallel Do Private(i,j,k)
          Do i = iLo, iHi
            Do j = jLo, jHi
              Do k = kLo, kHi
                VMxOld(i,j,k) = VMx(i,j,k)
                VMx(i,j,k)    = VMxNew(i,j,k)
              End Do
            End Do
          End Do

          iLo = LBound(VMy,1);  iHi = UBound(VMy,1)
          jLo = LBound(VMy,2);  jHi = UBound(VMy,2)
          kLo = LBound(VMy,3);  kHi = UBound(VMy,3)
!$OMP Parallel Do Private(i,j,k)
          Do i = iLo, iHi
            Do j = jLo, jHi
              Do k = kLo, kHi
                VMyOld(i,j,k) = VMy(i,j,k)
                VMy(i,j,k)    = VMyNew(i,j,k)
              End Do
            End Do
          End Do

          iLo = LBound(VMz,1);  iHi = UBound(VMz,1)
          jLo = LBound(VMz,2);  jHi = UBound(VMz,2)
          kLo = LBound(VMz,3);  kHi = UBound(VMz,3)
!$OMP Parallel Do Private(i,j,k)
          Do i = iLo, iHi
            Do j = jLo, jHi
              Do k = kLo, kHi
                VMzOld(i,j,k) = VMz(i,j,k)
                VMz(i,j,k)    = VMzNew(i,j,k)
              End Do
            End Do
          End Do

          iLo = LBound(Tmpr,1);  iHi = UBound(Tmpr,1)
          jLo = LBound(Tmpr,2);  jHi = UBound(Tmpr,2)
          kLo = LBound(Tmpr,3);  kHi = UBound(Tmpr,3)
!$OMP Parallel Do Private(i,j,k)
          Do i = iLo, iHi
            Do j = jLo, jHi
              Do k = kLo, kHi
                TmpOld(i,j,k) = Tmpr(i,j,k)
                Tmpr(i,j,k)   = TmpNew(i,j,k)
              End Do
            End Do
          End Do

        Return
        End
        
