! ***************************************************
! *  Eigen Value decomposition of Lap(Vz)           *
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
                       if (i/=Nx1)  D2_dx2(i,i+1) = P2

      	        D2_dx2(i,i)   = -(P1+P2) 

                if (i/=1)	 Vz_left(i) = D2_dx2(i,i-1) * DGr
                if (i/=Nx1)	Vz_right(i) = D2_dx2(i,i+1) * DGr 
                           Vz_center(i) = D2_dx2(i,i  ) * DGr 
    End do

!	Call Vgeev (D2_dx2(1:Nx1,1:Nx1),ExVz(1:Nx1,1:Nx1),   &
!	&              Ex_invVz(1:Nx1,1:Nx1),LambxVz(1:Nx1),Nx1)
	 
!	 LambxVz = LambxVz * DGr
	
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
	 
	 LambyVz = LambyVz * DGr
          
! ==================== Z-direction ===============================
     
        D2_dz2 = 0.D0
  
    Do i=1,Nz
                        P1 = 1.D0 / ( HPz(i) * Hz12(i-1) )
                        P2 = 1.D0 / ( HPz(i) * Hz12( i ) )

                        if (i/=1)	D2_dz2(i,i-1) = P1                
                        if (i/=Nz)	D2_dz2(i,i+1) = P2
                
                      D2_dz2(i,i)   = -(P1+P2) 
 	End do

	Call Vgeev  (D2_dz2(1:Nz,1:Nz),EzVz(1:Nz,1:Nz),   &
	&               Ez_invVz(1:Nz,1:Nz),LambzVz(1:Nz),Nz)
	 
	 LambzVz = LambzVz * DGr

End Subroutine  EVDLapVz
