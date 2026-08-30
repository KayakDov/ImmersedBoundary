MODULE EvdProcedures
   Use Numbers
   Use Numerica
   Use Grid
   Use Parameters
   Use EVD_Operators
   Use Thomas_coefficients
 
CONTAINS
    
! ***************************************************
! *  Eigen Value decomposition of Lap(P)         *
! *                                                 *
! ***************************************************  
    
Subroutine  EVDLapP
 

   Implicit Real(kind=8) (A-H,O-Z)

! ====================== X-direction ===============================

        D2_dx2 = 0.D0

	Do  i=1,Nx1
                      P1 = 1.D0 /( Hx12(i-1) * HPx(i-1) )
                      P2 = 1.D0 /( Hx12(i-1) * HPx( i ) )
                      
                      if (i/=1  ) D2_dx2(i,i-1) = P1
                      if (i/=Nx1) D2_dx2(i,i+1) = P2

      	              D2_dx2(i,i)   = -(P1+P2) 
    End do
   
	Call Vgeev (D2_dx2(1:Nx1,1:Nx1),ExxP(1:Nx1,1:Nx1),  &
	    &          Ex_invP(1:Nx1,1:Nx1),LambxP(1:Nx1),Nx1)
	    
! ==================== Y-direction =============================
  
        D2_dy2 = 0.D0
  
    Do i=1,Ny1
                      P1 = 1.D0 /( Hy12(i-1) * HPy(i-1) )
                      P2 = 1.D0 /( Hy12(i-1) * HPy( i ) )

                      if (i/=1  )	D2_dy2(i,i-1) = P1                
                      if (i/=Ny1)	D2_dy2(i,i+1) = P2
                
                      D2_dy2(i,i)   = -(P1+P2) 

                      if (i==1)   D2_dy2(i,i)   = -P2
                      if (i==Ny1) D2_dy2(i,i)   = -P1   
 	End do

	Call Vgeev  (D2_dy2(1:Ny1,1:Ny1),EyP(1:Ny1,1:Ny1),    &
	     &          Ey_invP(1:Ny1,1:Ny1),LambyP(1:Ny1),Ny1)
	 
! ==================== Z-direction =============================
  
        D2_dz2 = 0.D0
  
    Do i=1,Nz1
                      P1 = 1.D0 /( Hz12(i-1) * HPz(i-1) )
                      P2 = 1.D0 /( Hz12(i-1) * HPz( i ) )

                      if (i/=  1)	D2_dz2(i,i-1) = P1                
                      	            D2_dz2(i,i+1) = P2
                
                      D2_dz2(i,i)   = -(P1+P2) 

                      if (i==1)   D2_dz2(i,i)   = -P2
                      if (i==Nz1) D2_dz2(i,i)   = -P1   

                      if (i/=1)    P_left(i) = D2_dz2(i,i-1) 
                      if (i/=Nz1) P_right(i) = D2_dz2(i,i+1) 
                                 P_center(i) = D2_dz2(i,i  )   
 	End do

!	Call Vgeev  (D2_dz2(1:Nz1,1:Nz1),EzP(1:Nz1,1:Nz1),    &
!	     &          Ez_invP(1:Nz1,1:Nz1),LambzP(1:Nz1),Nz1)
	 

End Subroutine  EVDLapP

! ***************************************************
! *  Eigen Value decomposition of Lap(Vx)         *
! *                                                 *
! ***************************************************
Subroutine  EVDLapVx
 
        
  Implicit Real(kind=8) (A-H,O-Z)

! ================== X - direction ==============================
        D2_dx2 = 0.D0

	Do  i=1,Nx
                      P1 = 1.D0 / ( HPx(i) * Hx12(i-1) )
                      P2 = 1.D0 / ( HPx(i) * Hx12( i ) )
                                    
                      if (i/=1)	  D2_dx2(i,i-1) = P1
                    If(i /= Nx)   D2_dx2(i,i+1) = P2

      	              D2_dx2(i,i)   = -(P1+P2) 
 
!                      if (i/=1)   Vx_left(i)  = D2_dx2(i,i-1) / Reynolds
!                                 Vx_right(i)  = D2_dx2(i,i+1) / Reynolds 
!                                Vx_center(i)  = D2_dx2(i,i  ) / Reynolds
    End do

	call Vgeev (D2_dx2(1:Nx,1:Nx),ExVx(1:Nx,1:Nx),   &
	&              Ex_invVx(1:Nx,1:Nx),LambxVx(1:Nx),Nx)

	LambxVx = LambxVx / Reynolds
	
! ================= Y - direction =================================
     
        D2_dy2 = 0.D0
  
    Do i=1,Ny1
                    P1 = 1.D0 / ( HPy(i-1) * Hy12(i-1) )
                    P2 = 1.D0 / ( HPy( i ) * Hy12(i-1) )

                    if (i/=1)	D2_dy2(i,i-1) = P1                
                    if (i/=Ny1)	D2_dy2(i,i+1) = P2
                
                    D2_dy2(i,i)   = -(P1+P2) 
 	End do

	Call Vgeev  (D2_dy2(1:Ny1,1:Ny1),EyVx(1:Ny1,1:Ny1),   &
	&               Ey_invVx(1:Ny1,1:Ny1),LambyVx(1:Ny1),Ny1)
	 
	 LambyVx = LambyVx / Reynolds
          
! ================= Z - direction =================================
     
        D2_dz2 = 0.D0
  
    Do i=1,Nz1
                    P1 = 1.D0 / ( HPz(i-1) * Hz12(i-1) )
                    P2 = 1.D0 / ( HPz( i ) * Hz12(i-1) )

                    if (i/=1)	D2_dz2(i,i-1) = P1                
                                D2_dz2(i,i+1) = P2
                
                    D2_dz2(i,i)   = -(P1+P2) 
 
                      if (i/=1)   Vx_left(i)  = D2_dz2(i,i-1) / Reynolds
                                 Vx_right(i)  = D2_dz2(i,i+1) / Reynolds 
                                Vx_center(i)  = D2_dz2(i,i  ) / Reynolds
 	End do

            Vx_left(Nz2) = 0.d0;  Vx_right(Nz2) = 0.d0;   Vx_center(Nz2) = 1.d0

!	Call Vgeev  (D2_dz2(1:Nz1,1:Nz1),EzVx(1:Nz1,1:Nz1),   &
!	&               Ez_invVx(1:Ny1,1:Ny1),LambzVx(1:Nz1),Nz1)
	 
!	 LambzVx = LambzVx / Reynolds

End Subroutine  EVDLapVx

! ***************************************************
! *  Eigen Value decomposition of Lap(Vy)         *
! *                                                 *
! ***************************************************
Subroutine  EVDLapVy
        
  Implicit Real(kind=8) (A-H,O-Z)

! ===================== X-direction =============================
        D2_dx2 = 0.D0

	Do  i=1,Nx1
                       P1 = 1.D0 / ( Hx12(i-1) * HPx(i-1) )
                       P2 = 1.D0 / ( Hx12(i-1) * HPx( i ) )
                   
                       if (i/=1)	D2_dx2(i,i-1) = P1
                     If(i /= Nx1)   D2_dx2(i,i+1) = P2
      	                            D2_dx2(i,i)   = -(P1+P2) 
    End do

	Call Vgeev (D2_dx2(1:Nx1,1:Nx1),ExVy(1:Nx1,1:Nx1),   &
	&              Ex_invVy(1:Nx1,1:Nx1),LambxVy(1:Nx1),Nx1)
	 
	 LambxVy = LambxVy / Reynolds
	
! ==================== Y-direction ===============================
     
        D2_dy2 = 0.D0
  
    Do i=1,Ny
                        P1 = 1.D0 / ( HPy(i) * Hy12(i-1) )
                        P2 = 1.D0 / ( HPy(i) * Hy12( i ) )

                        if (i/=1)	D2_dy2(i,i-1) = P1                
                        if (i/=Ny)	D2_dy2(i,i+1) = P2
                
                      D2_dy2(i,i)   = -(P1+P2) 
 	End do

	Call Vgeev  (D2_dy2(1:Ny,1:Ny),EyVy(1:Ny,1:Ny),   &
	&               Ey_invVy(1:Ny,1:Ny),LambyVy(1:Ny),Ny)
	 
	 LambyVy = LambyVy / Reynolds
          
! ==================== Z-direction ===============================
     
        D2_dz2 = 0.D0
  
    Do i=1,Nz1
                       P1 = 1.D0 / ( Hz12(i-1) * HPz(i-1) )
                       P2 = 1.D0 / ( Hz12(i-1) * HPz( i ) )

                        if (i/=1)	D2_dz2(i,i-1) = P1                
                                 	D2_dz2(i,i+1) = P2
                
                      D2_dz2(i,i)   = -(P1+P2) 

                   if (i/=1) Vy_left(i)  = D2_dz2(i,i-1) / Reynolds
                            Vy_right(i)  = D2_dz2(i,i+1) / Reynolds 
                           Vy_center(i)  = D2_dz2(i,i  ) / Reynolds 
 	End do

            Vy_left(Nz2) = 0.d0;  Vy_right(Nz2) = 0.d0;   Vy_center(Nz2) = 1.d0

!	Call Vgeev  (D2_dz2(1:Nz1,1:Nz1),EzVy(1:Nz1,1:Nz1),   &
!	&               Ez_invVy(1:Nz1,1:Nz1),LambzVy(1:Nz1),Nz1)
	 
!	 LambzVy = LambzVy / Reynolds

End Subroutine  EVDLapVy

! ***************************************************
! *  Eigen Value decomposition of Lap(Vy)           *
! *                                                 *
! ***************************************************
Subroutine  EVDLapVz
 
  Use Numbers
  Use Numerica
  Use Grid
  Use Parameters
  Use EVD_Operators
  Use Thomas_coefficients
        
  Implicit Real(kind=8) (A-H,O-Z)

! ===================== X-direction =============================
        D2_dx2 = 0.D0

	Do  i=1,Nx1
                       P1 = 1.D0 / ( Hx12(i-1) * HPx(i-1) )
                       P2 = 1.D0 / ( Hx12(i-1) * HPx( i ) )
                   
                       if (i/=1)	D2_dx2(i,i-1) = P1
                     If(i /= Nx1)   D2_dx2(i,i+1) = P2
                         	        D2_dx2(i,i)   = -(P1+P2) 
    End do


	Call Vgeev (D2_dx2(1:Nx1,1:Nx1),ExVz(1:Nx1,1:Nx1),   &
	&              Ex_invVz(1:Nx1,1:Nx1),LambxVz(1:Nx1),Nx1)
	 
	 LambxVz = LambxVz / Reynolds
	
! ==================== Y-direction ===============================
     
        D2_dy2 = 0.D0
  
    Do i=1,Ny1
                       P1 = 1.D0 / ( Hy12(i-1) * HPy(i-1) )
                       P2 = 1.D0 / ( Hy12(i-1) * HPy( i ) )

                        if (i/=1)	D2_dy2(i,i-1) = P1                
                        if (i/=Ny1)	D2_dy2(i,i+1) = P2
                
                      D2_dy2(i,i)   = -(P1+P2) 
 	End do

	Call Vgeev  (D2_dy2(1:Ny1,1:Ny1),EyVz(1:Ny1,1:Ny1),   &
	&               Ey_invVz(1:Ny1,1:Ny1),LambyVz(1:Ny1),Ny1)
	 
	 LambyVz = LambyVz / Reynolds
          
! ==================== Z-direction ===============================
     
        D2_dz2 = 0.D0
  
    Do i=1,Nz
                        P1 = 1.D0 / ( HPz(i) * Hz12(i-1) )
                        P2 = 1.D0 / ( HPz(i) * Hz12( i ) )

                        if (i/=1)	D2_dz2(i,i-1) = P1                
                                 	D2_dz2(i,i+1) = P2
                                    D2_dz2(i,i)   = -(P1+P2) 

                if (i/=1)	 Vz_left(i) = D2_dz2(i,i-1) / Reynolds
                             Vz_right(i) = D2_dz2(i,i+1) / Reynolds
                            Vz_center(i) = D2_dz2(i,i  ) / Reynolds 
 	End do
            Vz_left(Nz1) = 0.d0;  Vz_right(Nz1) = 0.d0;   Vz_center(Nz1) = 1.d0

!	Call Vgeev  (D2_dz2(1:Nz,1:Nz),EzVz(1:Nz,1:Nz),   &
!	&               Ez_invVz(1:Nz,1:Nz),LambzVz(1:Nz),Nz)
	 
!	 LambzVz = LambzVz / Reynolds

End Subroutine  EVDLapVz
    
    
    Subroutine  EVD_Thomas_z (f_New, f_rhs, Ex,Ex_inv, Ey,Ey_inv, &
     &                                Lambx,Lamby,P_a, P_bb, P_c,Nxsol,Nysol,Nzsol,Dtm)
        
   Use Numbers
   Use Numerica
   Use Grid
   Use Parameters
   Use EVD_Operators
   Use Thomas_coefficients

   Implicit Real(kind=8) (A-H,O-Z)
         
   real(kind=8), dimension(1:Nysol,1:Nysol):: Ey,Ey_inv
   real(kind=8), dimension(1:Nysol)        :: Lamby
   real(kind=8), dimension(1:Nxsol,1:Nxsol):: Ex,Ex_inv
   real(kind=8), dimension(1:Nxsol)        :: Lambx
   
   real(kind=8), POINTER:: f_new(:,:,:), f_rhs(:,:,:)

   real(kind=8), dimension(1:Nzsol)        :: X_Thom
   real(kind=8), dimension(1:Nzsol)        :: P_a, P_b, P_c, P_bb
   real(kind=8), dimension(1:Nzsol)        :: d_RHS, c_tag, d_tag

! ===================================================================

!$OMP Parallel do Private(j,k)
     Do k=1,Nzsol
      Do j=1,Nysol
        f_New(1:NxSol,j,k) = matmul( Ex_Inv(1:Nxsol,1:Nxsol), f_rhs(1:NxSol,j,k) )
      End Do
     End Do
 !    Write (*,*) '1 fnew =' , Sum( abs(F_new(1:Nxsol,1:Nysol,1:Nzsol) ))

!$OMP Parallel do Private(i,k)
     Do i=1,Nxsol
      Do k=1,Nzsol
         f_New(i,1:Nysol,k) = matmul( Ey_Inv(1:Nysol,1:Nysol), f_New(i,1:Nysol,k) )
      End Do
     End Do
 !    Write (*,*) '2 fnew =' , Sum( abs(F_new(1:Nxsol,1:Nysol,1:Nzsol) ))

!$OMP Parallel Do Private(j,k,i,pdum,zdum,c_tag, d_tag, X_thom, d_RHS, P_b) 
	 do j=1,Nysol
	  do k=1,Nxsol
	            pdum =( -(Ckor/Htime)*Dtm +  Lamby(j) + Lambx(k) )
	            if (abs(pdum) <= 1.D-8) pdum = 1.D0

             P_b(1:Nzsol) = P_bb(1:Nzsol) 
           d_RHS(1:Nzsol) = f_New(k,j,1:Nzsol)

           If(Dtm == 0.d0) then
                                P_b(1:Nzsol) = P_b(1:Nzsol) + pdum
                           else
                                P_b(1:Nzsol-1) = P_b(1:Nzsol-1) + pdum
                          !      P_b(Nxsol)     = P_b(Nxsol) + Lamby(j) + Lambz(k)
           End If

! ============ Embedded Thomas algorithm ====================

       ! /* Modify the coefficients */             
                  
         c_tag(1) = P_c(1)/P_b(1)
         d_tag(1) = d_RHS(1)/P_b(1)
         
         do i=2,Nzsol
            zdum = P_b(i) - c_tag(i-1)*P_a(i)
            c_tag(i) = P_c(i) / zdum
            d_tag(i) =( d_RHS(i) - d_tag(i-1)*P_a(i) ) / zdum
         end do
 	 
	  !  /* Now back substitute */

	      X_Thom(Nzsol) = d_tag(Nzsol)
	     
	     do i=Nzsol-1,1,-1
	        X_Thom(i) = d_tag(i) - c_tag(i) * X_Thom(i+1)
	     end do

           f_new(k,j,1:Nzsol) = X_Thom(1:Nzsol)
      end do
     end do

 !    Write (*,*) '3 fnew =' , Sum( abs(F_new(1:Nxsol,1:Nysol,1:Nzsol) ))

!$OMP Parallel do Private(i,k)
     Do i=1,Nxsol
      Do k=1,Nzsol
         f_New(i,1:Nysol,k) = matmul( Ey(1:Nysol,1:Nysol), f_New(i,1:Nysol,k) )
      End Do
     End Do
 !    Write (*,*) '4 fnew =' , Sum( abs(F_new(1:Nxsol,1:Nysol,1:Nzsol) ))

!$OMP Parallel do Private(k,j)
     Do k=1,Nzsol
      Do j=1,Nysol
         f_New(1:NxSol,j,k) = matmul( Ex(1:Nxsol,1:Nxsol), f_New(1:NxSol,j,k) )
      End Do
     End Do
!    Write (*,*) '5 fnew =' , Sum( abs(F_new(1:Nxsol,1:Nysol,1:Nzsol) ))

   Return
 
     End Subroutine  EVD_Thomas_z
     
     
     Subroutine Vgeev (Mat,E,E_inv,Lamb,Ne)

   implicit real(kind=8)(a-h,o-z)
	
   real(kind=8), dimension(1:Ne,1:Ne):: E, E_inv, VL, Mat, Check
   real(kind=8), dimension(1:Ne)     :: Lamb, WI

   real(kind=8), dimension(6*(Ne+1)) :: WORK
   integer,      dimension(Ne+1)     :: IPIV

! ==================================================================================
      LWORKevd=4*Ne
      E = 0.D0;   E_inv = 0.D0;   VL=0.D0; Lamb = 0.D0 
      Check(1:Ne,1:Ne)= Mat(1:Ne,1:Ne)

	  Call DGEEV('N','V',Ne, Mat(1:Ne,1:Ne),Ne,Lamb,WI,VL(1:Ne,1:Ne),Ne,E(1:Ne,1:Ne),Ne,WORK,LWORKevd,INFO) 
		       If(info /= 0)  Write (*,*) 'ZGEEV: info=', info
		       If(MaxVal(Abs(WI(1:NE))) > 1.D-06) Write (*,*) ' Complex eigenvalue !!!'
		
      E_inv(1:Ne,1:Ne)  = E(1:Ne,1:Ne)

      Call DGETF2 (Ne,Ne,E_Inv(1:Ne,1:Ne),Ne,IPIV, INFO)
		If(info /= 0)  Write (*,*) 'VR_inv (DGETF2): info=', info
 
      Call DGETRI (Ne,E_Inv(1:Ne,1:Ne), Ne,IPIV, WORK, Ne, INFO)
		If(info /= 0)  Write (*,*) 'VR_inv (DGETRI): info=', info

!=======  Checking Inverse matrix ===============================

   flag=1
   if (flag==1) then
			VL = 0.D0
			do i=1,Ne
			    VL(i,i)= 1.D0
		    end do
			
            Write(*,*) "max (inverse matrix)=", maxval(abs(matmul(E_inv,E)-VL))
 
 !=======  Checking Eigenvalue decomposition  =====================
            VL = 0.D0
		    do i=1,Ne
			   VL(i,i)= Lamb(i)
		    end do

			VL(1:Ne,1:Ne)=matmul(E(1:Ne,1:Ne),VL(1:Ne,1:Ne))
			VL(1:Ne,1:Ne)=matmul(VL(1:Ne,1:Ne),E_inv(1:Ne,1:Ne))
			
            Write(*,*) "max (eigenvalue decomposition)=", maxval(abs(Check-VL))
   end if

End Subroutine Vgeev 


     
     END MODULE EvdProcedures
  
