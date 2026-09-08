! ************************************************************
! *   Subroutine for straight-forward solution  of           *     
! *       FD problem for convection in rectangulars          *
! *                                                          *
! *   This is version with 3-level time integrator           *
! *                                                          *
! ************************************************************

        Subroutine  TimeStep ( Istp, RNSx, RNSy, RNSz, RDP )

         Use Numbers
         Use Parameters
         Use Numerica
         Use Grid
         Use Operators
         Use Variables
         Use EVD_Operators
         Use Thomas_coefficients
         Use IBsetupInetrpRegular
         Use EvdProcedures
         Use MatrixFormAndOperate
         Use FVOperators
         Use, Intrinsic :: ISO_C_BINDING, Only: C_SIZE_T
         Implicit None
         Integer nbd, Istp, loc_start, loc_end, sz, i, j, k
         Integer :: uSizeU, uSizeV, uSizeW, uOffset2, uOffset3
         Real(kind=8) :: Nusselt,Nusseltx,Nusselt1y,Nusselt2y,Nusselt1z,Nusselt2z, tempsum
	     Real(kind=8):: Dist2D, Ht, RNSx, RNSy, RNSz, RTmpr, RDP, omp_get_wtime, t1,t2,t3,t4
         Real*8 Proba11(1:10)
	     Real(kind=8) :: UGammaVec(3*TotalUnknownsP), resultPPrime(Nx1*Ny1*Nz1)
	     Real(kind=8), Allocatable :: uStar(:)
          Ht = 2.D0 * Htime

         ! uStar layout: three segments, one per velocity component, each
         ! sized per ImerssedEquation.cu's setRHSPPrime slicing (component
         ! staggered +1 in its own direction) -- see the call site below.
         uSizeU = Ny1*(Nx1+1)*Nz1
         uSizeV = Nx1*(Ny1+1)*Nz1
         uSizeW = Nx1*Ny1*(Nz1+1)
         uOffset2 = uSizeU
         uOffset3 = uSizeU + uSizeV
         IF (ALLOCATED(uStar)) DEALLOCATE(uStar)
         ALLOCATE(uStar(uSizeU + uSizeV + uSizeW))
        
! ########### Inversing the Stokes operator #################

! ======== Make r.h.s. of momentum equations ==================

          RHSx=0.d0
          RHSy=0.d0
          RHSz=0.d0
          
          RHP=>RHSx
          PresP=>Prs
          Call GradPx( RHP, PresP )
 
          RHP=>RHSy
          PresP=>Prs
          Call GradPy( RHP, PresP )
          
          RHP=>RHSz
          PresP=>Prs
          Call GradPz( RHP, PresP )
          
          DO nbd=1,n_body 
            Call regularization(nbd) ! Yuri added on Aug 23 2021 
          END DO
          
          Call   VgrdVx 
          Call   VgrdVy 
          Call   VgrdVz 
! +++++++++++ Straight-forward step +++++++++++++++++++++          

           RHSx(1:Nx,1:Ny1,1:Nz1) = RHSx(1:Nx,1:Ny1,1:Nz1) - &
     &                  ( 4.D0 * VMx(1:Nx,1:Ny1,1:Nz1) - VMxOld(1:Nx,1:Ny1,1:Nz1) )/ Ht

           RHSy(1:Nx1,1:Ny,1:Nz1) = RHSy(1:Nx1,1:Ny,1:Nz1) - &
     &                  ( 4.D0 * VMy(1:Nx1,1:Ny,1:Nz1) - VMyOld(1:Nx1,1:Ny,1:Nz1) )/ Ht
 
           RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz) - &
     &                  ( 4.D0 * VMz(1:Nx1,1:Ny1,1:Nz) - VMzOld(1:Nx1,1:Ny1,1:Nz) )/ Ht
           
 
! ......... Driving Lid .............................................

           RHSx(:,:,Nz2) = Cos(angle)
           RHSy(:,:,Nz2) = Sin(angle)
           RHSz(:,:,Nz1) = 0.d0

! ++++++++++++ [Calculate Lap(u)^-1]*RHSx +++++++++++++++++++++++
      Thomas_f_New=>VmxNew
      Thomas_f_rhs=>RHSx
      Call   EVD_Thomas_z (Thomas_f_New, Thomas_f_rhs,  &
     &                   ExVx(1:Nx,1:Nx),Ex_invVx(1:Nx,1:Nx),            &
     &                   EyVx(1:Ny1,1:Ny1),Ey_invVx(1:Ny1,1:Ny1),            &
     &                   LambxVx(1:Nx), LambyVx(1:Ny1),                     &
     &                   Vx_left(1:Nz2), Vx_center(1:Nz2), Vx_right(1:Nz2),  &
     &                   Nx, Ny1, Nz2, 1.D0)

! ++++++++++++++ [Calculate Lap(v)^-1]*RHSy ++++++++++++++++++++++++++
      Thomas_f_New=>VMyNew
      Thomas_f_rhs=>RHSy
      Call   EVD_Thomas_z (Thomas_f_New, Thomas_f_rhs,    &
     &                   ExVy(1:Nx1,1:Nx1),Ex_invVy(1:Nx1,1:Nx1),            &
     &                   EyVy(1:Ny,1:Ny),  Ey_invVy(1:Ny,1:Ny),              &
     &                   LambxVy(1:Nx1), LambyVy(1:Ny),                      &
     &                   Vy_left(1:Nz2), Vy_center(1:Nz2), Vy_right(1:Nz2),  &
     &                   Nx1, Ny, Nz2, 1.D0)

! ++++++++++++++ [Calculate Lap(w)^-1]*RHSz ++++++++++++++++++++++++++
      Thomas_f_New=>VMzNew
      Thomas_f_rhs=>RHSz
      Call   EVD_Thomas_z (Thomas_f_New, Thomas_f_rhs,        &
     &                   ExVz(1:Nx1,1:Nx1),  Ex_invVz(1:Nx1,1:Nx1),              &
     &                   EyVz(1:Ny1,1:Ny1),Ey_invVz(1:Ny1,1:Ny1),            &
     &                   LambxVz(1:Nx1), LambyVz(1:Ny1),                     &
     &                   Vz_left(1:Nz1), Vz_center(1:Nz1), Vz_right(1:Nz1),  &
     &                   Nx1, Ny1, Nz1, 1.D0)

      Call EVDbounds      
      
!++++++++ Calcualte pressure correction ++++++++++++

      FDRHP= 0.d0
      
      Call FdDiv
  
      FDRHP = FDRHP * Ckor / Htime !This is RHS_p_prime
   
! ++++++++++++ Calculate velocities ++++++++++++++++++++++++++++++

      ! ---- GPU (CudaBandedLib) coupled pressure/force solve ----
      ! Replaces the old interpolation()+RHS_F_tag block, Precond_RHS_P,
      ! BICG5D, and MKL_B_MatVec/F_tag block above. solve_immersed_eq_primes_d_i32
      ! computes R^T*uStar itself from the raw velocity field and R (so
      ! interpolation() is no longer called here), and also reconstructs the
      ! -3/(2*Htime)*div(u*) term itself from uStar internally.
      !
      ! uStar's THREE segments are NOT all Nx1*Ny1*Nz1 -- ImerssedEquation.cu's
      ! setRHSPPrime slices velocities into u=Ny1*(Nx1+1)*Nz1, v=Nx1*(Ny1+1)*Nz1,
      ! w=Nx1*Ny1*(Nz1+1) (each component staggered +1 in its own direction,
      ! matching Yuri's own VMxNew(0,:,:)/VMxNew(Nx1,:,:) etc. wall assignments
      ! in bounds_Lid3D_z.f90 -- confirmed, not guessed). Within each segment,
      ! the library's Y-fastest/X-middle/Z-slowest order applies (see
      ! MatrixFormAndOperate.f90's Init_GPU_IBM_Solver header) using that
      ! segment's OWN (possibly +1) extent.
      !
      ! Still not verified:
      ! (a) UGamma = bdy(:)%V{x,y,z}_b concatenated in the same x-then-y-then-z,
      !     body-by-body order as F_tag/R's columns -- Vx_b/Vy_b/Vz_b are
      !     zero-initialized and used elsewhere as a "target" boundary
      !     velocity (fb_x=(Vx_b-Vx_interp_New)/Htime), which is why I've
      !     mapped them to U^Gamma, but I have not confirmed this against
      !     Yuri or the paper.
      ! (b) Whether setRHSPPrime's divergence stencil (2nd-order central,
      !     scaled by 3/(2*Htime)) matches Yuri's FdDiv/Ckor exactly.

      DO k=1,Nz1
       DO j=1,Ny1
        DO i=0,Nx1
          uStar(j + i*Ny1 + (k-1)*Ny1*(Nx1+1)) = VMxNew(i,j,k)
        END DO
       END DO
      END DO

      DO k=1,Nz1
       DO j=0,Ny1
        DO i=1,Nx1
          uStar(uOffset2 + (j+1) + (i-1)*(Ny1+1) + (k-1)*(Ny1+1)*Nx1) = VMyNew(i,j,k)
        END DO
       END DO
      END DO

      DO k=0,Nz1
       DO j=1,Ny1
        DO i=1,Nx1
          uStar(uOffset3 + j + (i-1)*Ny1 + k*Ny1*Nx1) = VMzNew(i,j,k)
        END DO
       END DO
      END DO

      loc_start=1
      DO nbd=1, n_body
        loc_end=bdy(nbd)%Npts+loc_start-1
        UGammaVec(loc_start:loc_end)=bdy(nbd)%Vx_b
        loc_start=loc_end+1
        loc_end=bdy(nbd)%Npts+loc_start-1
        UGammaVec(loc_start:loc_end)=bdy(nbd)%Vy_b
        loc_start=loc_end+1
        loc_end=bdy(nbd)%Npts+loc_start-1
        UGammaVec(loc_start:loc_end)=bdy(nbd)%Vz_b
        loc_start=loc_end+1
      END DO

      CALL solve_immersed_eq_primes_d_i32( &
          resultPPrime, F_tag, &
          INT(SIZE(B_CSR_Prs), C_SIZE_T), B_RowOffsets0, B_ColInds0, B_CSR_Prs, &
          INT(SIZE(R_CSC_Val), C_SIZE_T), R_ColOffsets0, R_RowInds0, R_CSC_Val, &
          UGammaVec, uStar)

      DO k=1,Nz1
       DO j=1,Ny1
        DO i=1,Nx1
          Dprs(i,j,k) = resultPPrime(LibGridIdx(i,j,k))
        END DO
       END DO
      END DO
     
           
          RHSx=0.d0
          RHSy=0.d0
          RHSz=0.d0
      
      loc_start=1             
      DO nbd=1,  n_body
             
        loc_end=bdy(nbd)%Npts+loc_start-1
        
        bdy(nbd)%fb_x_tag=F_tag(loc_start:loc_end)
        bdy(nbd)%fb_x= bdy(nbd)%fb_x+bdy(nbd)%fb_x_tag 
        
        loc_start=loc_end+1
        loc_end= bdy(nbd)%Npts+loc_start-1
        
        bdy(nbd)%fb_y_tag=F_tag(loc_start:loc_end)
        bdy(nbd)%fb_y=bdy(nbd)%fb_y+bdy(nbd)%fb_y_tag 
        
        loc_start=loc_end+1
        loc_end=bdy(nbd)%Npts+loc_start-1
        
        bdy(nbd)%fb_z_tag=F_tag(loc_start:loc_end)
        bdy(nbd)%fb_z=bdy(nbd)%fb_z+bdy(nbd)%fb_z_tag 
        
        Call regularization_tag(nbd)  
        loc_start=loc_end+1      
        
      END DO
 !....... First correction of u'for R[F']...................!  
        
         VMxNew(1:Nx,1:Ny1,1:Nz1) = VMxNew(1:Nx,1:Ny1,1:Nz1) +RHSx(1:Nx ,1:Ny1,1:Nz1) * Htime / Ckor
         VMyNew(1:Nx1,1:Ny,1:Nz1) = VMyNew(1:Nx1,1:Ny,1:Nz1) +RHSy(1:Nx1,1:Ny ,1:Nz1) * Htime / Ckor
         VMzNew(1:Nx1,1:Ny1,1:Nz) = VMzNew(1:Nx1,1:Ny1,1:Nz) +RHSz(1:Nx1,1:Ny1,1:Nz ) * Htime / Ckor
        
!............................................................!          
          
      
 
      RHP=>RHSx
      PresP=>Dprs
      Call GradPx( RHP, PresP )
      
   
      RHP=>RHSy
      PresP=>Dprs
      Call GradPy( RHP, PresP )
      
    
      RHP=>RHSz
      PresP=>Dprs
      Call GradPz( RHP, PresP )
      
! ........... Second corrector of  velocities for grad (p') .........................................

      VMxNew(1:Nx,1:Ny1,1:Nz1) = VMxNew(1:Nx,1:Ny1,1:Nz1) -RHSx(1:Nx ,1:Ny1,1:Nz1) * Htime / Ckor
      VMyNew(1:Nx1,1:Ny,1:Nz1) = VMyNew(1:Nx1,1:Ny,1:Nz1) -RHSy(1:Nx1,1:Ny ,1:Nz1) * Htime / Ckor
      VMzNew(1:Nx1,1:Ny1,1:Nz) = VMzNew(1:Nx1,1:Ny1,1:Nz) -RHSz(1:Nx1,1:Ny1,1:Nz ) * Htime / Ckor

      Prs = Prs +DPrs
      !call EVDbounds 
     

         RNSx = Dist2D (VMx, VMxNew, Nx1, Ny2, Nz2, Nx1, Ny2, Nz2)
         RNSy = Dist2D (VMy, VMyNew, Nx2, Ny1, Nz2, Nx2, Ny1, Nz2)
         RNSz = Dist2D (VMz, VMzNew, Nx2, Ny2, Nz1, Nx2, Ny2, Nz1)
         
         RDP = MaxVal(Abs(Dprs) )

  !       Write (*,*) RNSx, RNSy, RNSz, RDP
  !       stop
         
! ######## Check of results #############################

     ! FDRHP= 0.d0
     ! Call FdDiv
     ! write(*,*) maxval( abs(FDRHP))
                    
! ######## Shift of the time step ########################

444     		Continue

          VMxOld = VMx
          VMyOld = VMy
          VMzOld = VMz

          VMx  = VMxNew
          VMy  = VMyNew
          VMz  = VMzNew

        Return
        End
        
