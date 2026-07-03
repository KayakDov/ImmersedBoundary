! ***************************************************
! *  Eigen Value decomposition of Lap(P)         *
! *                                                 *
! ***************************************************
Subroutine  EVDLapP
 
   Use Numbers
   Use Numerica
   Use Grid
   Use Parameters
   Use EVD_Operators
   Use Thomas_coefficients

   Implicit Real(kind=8) (A-H,O-Z)

! ====================== X-direction ===============================

        D2_dx2 = 0.D0

	Do  i=1,Nx1
                      P1 = 1.D0 /( Hx12(i-1) * HPx(i-1) )
                      P2 = 1.D0 /( Hx12(i-1) * HPx( i ) )
                      
                      if (i/=1  ) D2_dx2(i,i-1) = P1
                      if (i/=Nx1) D2_dx2(i,i+1) = P2

      	              D2_dx2(i,i)   = -(P1+P2) 

                      if (i==1)   D2_dx2(i,i)   = -P2 
                      if (i==Nx1) D2_dx2(i,i)   = -P1
                      
                      if (i/=1)    P_left(i) = D2_dx2(i,i-1) 
                      if (i/=Nx1) P_right(i) = D2_dx2(i,i+1) 
                                 P_center(i) = D2_dx2(i,i  )   
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
                      if (i/=Nz1)	D2_dz2(i,i+1) = P2
                
                      D2_dz2(i,i)   = -(P1+P2) 

                      if (i==1)   D2_dz2(i,i)   = -P2
                      if (i==Nz1) D2_dz2(i,i)   = -P1   
 	End do

	Call Vgeev  (D2_dz2(1:Nz1,1:Nz1),EzP(1:Nz1,1:Nz1),    &
	     &          Ez_invP(1:Nz1,1:Nz1),LambzP(1:Nz1),Nz1)
	 

End Subroutine  EVDLapP
