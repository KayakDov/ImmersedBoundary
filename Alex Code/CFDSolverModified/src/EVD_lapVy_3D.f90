! ***************************************************
! *  Eigen Value decomposition of Lap(Vy)         *
! *                                                 *
! ***************************************************
Subroutine  EVDLapVy
 
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
                       if (i/=Nx1) D2_dx2(i,i+1) = P2

      	        D2_dx2(i,i)   = -(P1+P2) 

                if (i/=1)	 Vy_left(i)  = D2_dx2(i,i-1) * DGr
                if (i/=Nx1) Vy_right(i)  = D2_dx2(i,i+1) * DGr 
                           Vy_center(i)  = D2_dx2(i,i  ) * DGr 
    End do

!	Call Vgeev (D2_dx2(1:Nx1,1:Nx1),ExVy(1:Nx1,1:Nx1),   &
!	&              Ex_invVy(1:Nx1,1:Nx1),LambxVy(1:Nx1),Nx1)
	 
!	 LambxVy = LambxVy * DGr
	
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
	 
	 LambyVy = LambyVy * DGr
          
! ==================== Z-direction ===============================
     
        D2_dz2 = 0.D0
  
    Do i=1,Nz1
                       P1 = 1.D0 / ( Hz12(i-1) * HPz(i-1) )
                       P2 = 1.D0 / ( Hz12(i-1) * HPz( i ) )

                        if (i/=1)	D2_dz2(i,i-1) = P1                
                        if (i/=Nz1)	D2_dz2(i,i+1) = P2
                
                      D2_dz2(i,i)   = -(P1+P2) 
 	End do

	Call Vgeev  (D2_dz2(1:Nz1,1:Nz1),EzVy(1:Nz1,1:Nz1),   &
	&               Ez_invVy(1:Nz1,1:Nz1),LambzVy(1:Nz1),Nz1)
	 
	 LambzVy = LambzVy * DGr

End Subroutine  EVDLapVy
  