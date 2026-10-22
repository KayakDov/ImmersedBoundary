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
                 PressureHandle, GrPr, &
                 Temperature_pinned, Vx_pinned, Vy_pinned, Vz_pinned, Pressure_pinned
         Use eigenbcgsolver_eigen_mod, only : solve_eigen_decomp_d, synch_d

        Implicit Real(kind=8) (A-H,O-Z)

        ! No GPU_RHS_*/GPU_SOL_* scratch arrays anymore: each field's RHS is
        ! written directly into its pinned buffer (Temperature_pinned etc.,
        ! aliased once in AlexCudaCompatibility.f90's Initialize_GPU_Solvers)
        ! instead of into a separate array that then got memcpy'd into the
        ! library's own pinned buffer. Same for the solution on the way back
        ! out -- read directly from the pinned pointer instead of from a
        ! GPU_SOL_* array. The scale-factor computation (GrPr/DGr) and the
        ! padded<->unpadded remap below are unchanged; only which array they
        ! write into/read from has changed.

         ! Loop counters / per-array bounds for the explicit-loop rewrites
         ! below (see the comments at each site). Implicit typing would
         ! already make these Integer (i,j,k fall in the I-N range), but
         ! they're declared explicitly for clarity since they're new.
         Integer :: i, j, k, iLo, iHi, jLo, jHi, kLo, kHi

          Ht = 2.D0 * Htime

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
         Temperature_pinned = FDRHP(1:Nx1,1:Ny1,1:Nz1) * GrPr
         Call solve_eigen_decomp_d(TemperatureHandle)

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
         Vy_pinned = RHSy(1:Nx1,1:Ny,1:Nz1) / DGr
         ! TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): Vy/RHSy have
         ! never actually been sampled in this whole investigation -- every
         ! print so far only covered Vx/Vz/Potential. Vy feeds directly
         ! into FdDiv (Dprs's RHS), so a bug here would corrupt the
         ! pressure correction -- and hence BOTH Vx and Vz's final values
         ! -- even though Vx/Vz's own RHS-build-and-solve pipeline, checked
         ! in isolation, is already proven correct to 9-10 significant
         ! figures.
         If (Istp == 1) Then
             Write (*,*) ' EM: Sum|RHSy/DGr(pre-solve)|=', Sum(Abs(Vy_pinned))
         End If
         Call solve_eigen_decomp_d(VyHandle)

! ========= RHS/launch for Potential -- only needs VMx/VMz from the =======
! ========= start of this step, so it's independent of Temperature  =======
! ========= and Vy above too.                                       =======

           Call Get_Potential_Launch

! ======== Build the parts of RHSx/RHSz that don't need Potential or =======
! ======== TmpNew yet -- moved up from after Temperature's sync so    =======
! ======== this CPU-only work fills the wait instead of happening     =======
! ======== after it: Temperature, Vy, and Potential all get strictly  =======
! ======== more overlap time before anything below asks for their     =======
! ======== result. No dependency changes -- VgrdVx/GradPx and         =======
! ======== VgrdVz/GradPz only ever needed previous-timestep state     =======
! ======== (VMx/VMxOld, VMz/VMzOld, Prs), same as they always have.   =======

          FDRHP = 0.D0
          Call   VgrdVx
          Call   GradPx( RHSx(1:Nx ,1:Ny1,1:Nz1), Prs )
          RHSx(1:Nx,1:Ny1,1:Nz1) = RHSx(1:Nx,1:Ny1,1:Nz1) + FDRHP(1:Nx,1:Ny1,1:Nz1)

          FDRHP = 0.D0
          Call   VgrdVz
          Call   GradPz( RHSz(1:Nx1,1:Ny1,1:Nz ), Prs )
          RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz) + FDRHP(1:Nx1,1:Ny1,1:Nz)

           RHSx(1:Nx,1:Ny1,1:Nz1) = RHSx(1:Nx,1:Ny1,1:Nz1) - &
     &                  ( 4.D0 * VMx(1:Nx,1:Ny1,1:Nz1) - VMxOld(1:Nx,1:Ny1,1:Nz1) )/ Ht

           RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz) - &
     &                  ( 4.D0 * VMz(1:Nx1,1:Ny1,1:Nz) - VMzOld(1:Nx1,1:Ny1,1:Nz) )/ Ht

! ########### Temperature is needed now: synch #############################

         Call synch_d(TemperatureHandle)
         ! Was: TmpNew(1:Nx1,1:Ny1,1:Nz1) = GPU_SOL_T, and before that,
         ! TmpNew(...) = Temperature_pinned directly here -- TmpNew is
         ! ghost-padded (0:Nx2,0:Ny2,0:Nz2, see ConvMain_3D_Q2D.f90), so
         ! this slice isn't contiguous. Same fix as Get_Potential_Finish:
         ! explicit loop instead of whole-array syntax on a module array,
         ! plus free OMP parallelism. Bounds unchanged (still 1:Nx1,1:Ny1,1:Nz1).
         ! This loop itself is still needed -- it's the padded<->unpadded
         ! remap, not the redundant copy that was eliminated -- it now just
         ! reads from Temperature_pinned instead of from GPU_SOL_T.
!$OMP Parallel Do Private(i,j,k)
         Do i = 1, Nx1
           Do j = 1, Ny1
             Do k = 1, Nz1
               TmpNew(i,j,k) = Temperature_pinned(i,j,k)
             End Do
           End Do
         End Do

 !          Write (*,*) ' TmpNew=', Maxval(abs(TMpNew))

      Call EVDbounds

      RTmpr = Dist2D (TmpNew, Tmpr, Nx2, Ny2, Nz2, Nx2, Ny2, Nz2)
!      Write (*,*) ' RTmpr=', RTmpr, Minval(TmpNew), Maxval(tmpnew)
!        Write (*,*) 'DT=', Maxval(abs(TmpNew - Tmpr) ), Maxloc(abs(TmpNew - Tmpr) ), Maxval(abs(TmpNew(:,:,102)))
!stop

! .............. Add bouyancy force -- moved up from after the RHSx/RHSz ===
! .............. stencil block: this only needs TmpNew, which is already ==
! .............. valid at this point, and not Potential -- so it happens ==
! .............. before Potential's sync below instead of after it,      ==
! .............. giving Potential's solve that much more overlap time.   ==

           RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz)  &
     &     - 0.5d0 * Bu_Gr * ( TmpNew(1:Nx1,1:Ny1,1:Nz) + TmpNew(1:Nx1,1:Ny1,2:Nz1) ) &
     &     - 0.5d0 * Bu_Gr *  (   Teta(1:Nx1,1:Ny1,1:Nz) +   Teta(1:Nx1,1:Ny1,2:Nz1) )

! ########### Potential is needed now: synch (via Finish) ##################

           Call Get_Potential_Finish

! +++++++++ Electromagnetic force (needs Potential, just synched) ++++++++++

           Call EM_force

! ++++++++++++ [Calculate Lap(u)^-1]*RHSx +++++++++++++++++++++++

      ! GPU: was Call EVD_Thomas(..., 1.D0) with the CPU's own
      ! DGr-scaled Vx_left/Vx_center/Vx_right coefficients.
      ! Not synched here -- VMxNew isn't needed until the EVDbounds call
      ! below, after Vz is launched too.
         Vx_pinned = RHSx(1:Nx,1:Ny1,1:Nz1) / DGr
         ! TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): same reasoning
         ! as the C++ port's mirrored print right after its own 1/DGr scale --
         ! Sum|Vx|/Sum|Vz| after the correction step already disagree with
         ! C++ by 6-22% at Istp=1, well above GPU run-to-run noise, so this
         ! checks whether that discrepancy is already present in the array
         ! going INTO solve_eigen_decomp_d(VxHandle), or only appears after.
         If (Istp == 1) Then
             Write (*,*) ' EM: Sum|RHSx/DGr(pre-solve)|=', Sum(Abs(Vx_pinned))
         End If
         Call solve_eigen_decomp_d(VxHandle)

! ++++++++++++++ [Calculate Lap(w)^-1]*RHSz ++++++++++++++++++++++++++

      ! GPU: was Call EVD_Thomas(..., 1.D0), same DGr scaling as Vx/Vy.
         Vz_pinned = RHSz(1:Nx1,1:Ny1,1:Nz) / DGr
         If (Istp == 1) Then
             Write (*,*) ' EM: Sum|RHSz/DGr(pre-solve)|=', Sum(Abs(Vz_pinned))
         End If
         Call solve_eigen_decomp_d(VzHandle)

! ########### Vy, Vx, Vz are all needed now: synch each #####################

         Call synch_d(VyHandle)
         ! Padded<->unpadded remap, reading from Vy_pinned now instead of
         ! GPU_SOL_Vy -- same ghost-padding non-contiguity as TmpNew above.
!$OMP Parallel Do Private(i,j,k)
         Do i = 1, Nx1
           Do j = 1, Ny
             Do k = 1, Nz1
               VMyNew(i,j,k) = Vy_pinned(i,j,k)
             End Do
           End Do
         End Do

         If (Istp == 1) Then
             Write (*,*) ' EM: Sum|Vy(post-solve)|=', Sum(Abs(VMyNew(1:Nx1,1:Ny,1:Nz1)))
             Write (*,*) ' EM: Vy(post-solve)(10,50,50)=', VMyNew(10,50,50)
             Write (*,*) ' EM: Vy(post-solve)(90,50,50)=', VMyNew(90,50,50)
             Write (*,*) ' EM: Vy(post-solve)(50,50,10)=', VMyNew(50,50,10)
             Write (*,*) ' EM: Vy(post-solve)(50,50,90)=', VMyNew(50,50,90)
         End If

         Call synch_d(VxHandle)
!$OMP Parallel Do Private(i,j,k)
         Do i = 1, Nx
           Do j = 1, Ny1
             Do k = 1, Nz1
               VMxNew(i,j,k) = Vx_pinned(i,j,k)
             End Do
           End Do
         End Do

         Call synch_d(VzHandle)
!$OMP Parallel Do Private(i,j,k)
         Do i = 1, Nx1
           Do j = 1, Ny1
             Do k = 1, Nz
               VMzNew(i,j,k) = Vz_pinned(i,j,k)
             End Do
           End Do
         End Do

         ! TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): bisection
         ! point between the two remaining suspects. Sum|RHSx/DGr| and
         ! Sum|RHSz/DGr| above (right before solve_eigen_decomp_d) already
         ! match the C++ port to 8-9 significant figures, so the RHS build
         ! is exonerated. Sum|Vx|/Sum|Vz| below (after the pressure
         ! correction) still disagree with C++ by 22%/6%. This samples
         ! VMxNew/VMzNew at the one point in between: right after
         ! synch_d(VxHandle)/synch_d(VzHandle) and the padded<->unpadded
         ! remap, but BEFORE EVDbounds and the pressure-correction block
         ! touch them at all -- same bisection point as the C++ port's
         ! mirrored print right after its own synchVelocities(). If these
         ! already disagree with C++'s labels 5/6 by a similar margin, the
         ! bug is inside the implicit solve itself. If they match, the
         ! solve is innocent and the bug is in EVDbounds or the
         ! pressure-correction block that follows.
         If (Istp == 1) Then
             Write (*,*) ' EM: Sum|Vx(post-solve,pre-correction)|=', Sum(Abs(VMxNew(1:Nx,1:Ny1,1:Nz1)))
             Write (*,*) ' EM: Sum|Vz(post-solve,pre-correction)|=', Sum(Abs(VMzNew(1:Nx1,1:Ny1,1:Nz)))
             Write (*,*) ' EM: Vx(post-solve,pre-correction)(10,50,50)=', VMxNew(10,50,50)
             Write (*,*) ' EM: Vx(post-solve,pre-correction)(90,50,50)=', VMxNew(90,50,50)
             Write (*,*) ' EM: Vx(post-solve,pre-correction)(50,50,10)=', VMxNew(50,50,10)
             Write (*,*) ' EM: Vx(post-solve,pre-correction)(50,50,90)=', VMxNew(50,50,90)
             Write (*,*) ' EM: Vz(post-solve,pre-correction)(10,50,50)=', VMzNew(10,50,50)
             Write (*,*) ' EM: Vz(post-solve,pre-correction)(90,50,50)=', VMzNew(90,50,50)
             Write (*,*) ' EM: Vz(post-solve,pre-correction)(50,50,10)=', VMzNew(50,50,10)
             Write (*,*) ' EM: Vz(post-solve,pre-correction)(50,50,90)=', VMzNew(50,50,90)
         End If

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

         ! TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): the scaled
         ! RHS below (Sum|pressure RHS(pre-solve)|, unchanged further down)
         ! disagrees with the C++ port by ~23.7% even though every formula
         ! and every aggregate input has independently checked out: FdDiv's
         ! formula above matches divergenceKernel character-for-character,
         ! Subroutine Mesh's Hx12/Hy12/Hz12 match computeAxisSpacing, Ckor/
         ! Htime matches Real(1.5)/cfg_.timeStep exactly (Htime is read
         ! straight off the input deck and used nowhere as anything but the
         ! raw timestep -- it is NOT secretly 2*dt; Ht=2.D0*Htime elsewhere
         ! is a different, local quantity), EVDbounds's unconditional
         ! VMxNew/VMyNew/VMzNew zero blocks match
         ! BoundaryConditions::readVelocity's unconditional zero exactly (no
         ! Dirichlet/Neumann branch on velocity at all), and Sum|Vx|/
         ! Sum|Vy|/Sum|Vz| above already match the C++ port to 9-12 digits.
         ! The one thing a Sum(|.|) genuinely cannot rule out is a pointwise
         ! difference in VMxNew/VMyNew/VMzNew that cancels in their own
         ! aggregate but still lands on the wrong side of FdDiv's
         ! neighbor-difference stencil. This prints the raw divergence,
         ! before vs. after the Ckor/Htime scale, at the same four points
         ! now also sampled on VMxNew/VMyNew/VMzNew themselves above --
         ! directly comparable, cell by cell, against the C++ port's
         ! labels 5/6/8/11/10.
         If (Istp == 1) Then
             Write (*,*) ' EM: Sum|divergence(pre-scale)|=', Sum(Abs(FDRHP(1:Nx1,1:Ny1,1:Nz1)))
             Write (*,*) ' EM: divergence(pre-scale)(10,50,50)=', FDRHP(10,50,50)
             Write (*,*) ' EM: divergence(pre-scale)(90,50,50)=', FDRHP(90,50,50)
             Write (*,*) ' EM: divergence(pre-scale)(50,50,10)=', FDRHP(50,50,10)
             Write (*,*) ' EM: divergence(pre-scale)(50,50,90)=', FDRHP(50,50,90)
         End If

      FDRHP = FDRHP * Ckor / Htime

         If (Istp == 1) Then
             Write (*,*) ' EM: pressure RHS(pre-solve)(10,50,50)=', FDRHP(10,50,50)
             Write (*,*) ' EM: pressure RHS(pre-solve)(90,50,50)=', FDRHP(90,50,50)
             Write (*,*) ' EM: pressure RHS(pre-solve)(50,50,10)=', FDRHP(50,50,10)
             Write (*,*) ' EM: pressure RHS(pre-solve)(50,50,90)=', FDRHP(50,50,90)
         End If

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
         ! NOTE on this field specifically: Dprs(1:Nx1,1:Ny1,1:Nz1) is the
         ! array's whole declared extent (see the Allocate in
         ! ConvMain_3D_Q2D.f90), so under the OLD design synch_d could write
         ! the solution directly into Dprs with zero Fortran-side copy on
         ! the way out -- Pressure was the one field that never needed a
         ! GPU_SOL_P. Under this uniform pinned-pointer design, the solution
         ! lands in Pressure_pinned instead, and the explicit copy below is
         ! now needed to get it into Dprs. Net effect for Pressure
         ! specifically: the redundant incoming copy this change removes
         ! elsewhere is gone here too, but an outgoing copy that Pressure
         ! previously didn't need has been added -- roughly a wash for this
         ! one field, not a net reduction like the other five get.
         Pressure_pinned = FDRHP(1:Nx1,1:Ny1,1:Nz1)
         ! TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): Dprs
         ! post-solve disagrees with the C++ port by 31% (the Sum|Dprs...|
         ! line below) -- the largest mismatch found yet, bigger than Vx's
         ! 22% or Vz's 6%, and those are explained by it (both read this
         ! same Dprs through GradPx/GradPz, already proven-correct
         ! formulas). Vx/Vy/Vz's RHS AND post-solve values all matched to
         ! 9-12 digits, so this is the one piece left unverified. Same
         ! bisection as every other field: if this already disagrees with
         ! the C++ port, the bug is upstream (FdDiv/the Ckor/Htime scale);
         ! if it matches, the bug is inside the solve itself.
         If (Istp == 1) Then
             Write (*,*) ' EM: Sum|pressure RHS(pre-solve)|=', Sum(Abs(Pressure_pinned))
         End If
         Call solve_eigen_decomp_d(PressureHandle)
         Call synch_d(PressureHandle)
         Dprs(1:Nx1,1:Ny1,1:Nz1) = Pressure_pinned
 !   Write (*,*) ' Dprs=', Sum(abs(Dprs))

         ! TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation): Dprs has
         ! never been sampled either. Pressure's solver construction
         ! (dim lengths, deltas, all-Neumann BCs, FLUX_LAPL segments,
         ! shift=0) matches the C++ port's exactly, and this all-Neumann
         ! system is singular -- per the comment a few lines up, "the
         ! library's singular-mode handling must stay engaged" -- so a
         ! difference in how that singular mode is actually handled
         ! wouldn't show up as a parameter mismatch, only as a numeric
         ! one here.
         If (Istp == 1) Then
             Write (*,*) ' EM: Sum|Dprs(pressure-correction,post-solve)|=', Sum(Abs(Dprs(1:Nx1,1:Ny1,1:Nz1)))
         End If

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

      ! TEMPORARY DIAGNOSTIC (Hartmann=1000 investigation, next step): the
      ! C++ port's curl RHS (Potential's RHS build) already matches this
      ! run's Sum() closely at Istp=1, but the SOLVED Potential coming back
      ! from that same, parameter-identical CudaBandedLib call does not
      ! (wrong sign, ~9x magnitude). Since both codebases call the
      ! identical library routine with identical parameters for that solve,
      ! and it's correct here, the library itself is a dead end -- the one
      ! thing that can still differ is what's actually in the RHS buffer
      ! cell-by-cell, which traces back to VMx/VMz. So: sample VMxNew/
      ! VMzNew themselves, right here, right after this correction -- these
      ! are exactly what next step's curl-RHS build (Get_Potential_Launch)
      ! will read as VMx/VMz once the Old<-New shift below runs. Points
      ! spread near the x-wall and z-wall (where VariableDeltaLapl/mesh
      ! stretching concentrate resolution, i.e. where Hartmann layers would
      ! form) rather than clustered in one spot, since the last point
      ! samples (two cells of Potential's curl RHS) turned out to land in a
      ! near-zero/decaying bulk region that was too noisy to read anything
      ! into. Same four points, both fields, as the C++ port's mirrored
      ! diagnostic in correctVelocitiesAndPressure().
      If (Istp == 1) Then
          Write (*,*) ' EM: Vx(10,50,50)=', VMxNew(10,50,50)
          Write (*,*) ' EM: Vx(90,50,50)=', VMxNew(90,50,50)
          Write (*,*) ' EM: Vx(50,50,10)=', VMxNew(50,50,10)
          Write (*,*) ' EM: Vx(50,50,90)=', VMxNew(50,50,90)
          Write (*,*) ' EM: Vz(10,50,50)=', VMzNew(10,50,50)
          Write (*,*) ' EM: Vz(90,50,50)=', VMzNew(90,50,50)
          Write (*,*) ' EM: Vz(50,50,10)=', VMzNew(50,50,10)
          Write (*,*) ' EM: Vz(50,50,90)=', VMzNew(50,50,90)
          ! Robust tiebreaker: Sum(Abs(.)), not Sum(.) -- can't hide a real
          ! discrepancy behind positive/negative cancellation the way a
          ! plain Sum() (or a couple of point samples right at the
          ! quiescent-start noise floor) can.
          Write (*,*) ' EM: Sum|Vx|=', Sum(Abs(VMxNew(1:Nx,1:Ny1,1:Nz1)))
          Write (*,*) ' EM: Sum|Vz|=', Sum(Abs(VMzNew(1:Nx1,1:Ny1,1:Nz)))
      End If

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
