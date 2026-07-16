! ***************************************************
! *  Eigen Value decomposition of Lap(Vx)         *
! *                                                 *
! ***************************************************
Subroutine  EVDLapVx
 
  Use Numbers
  Use Numerica
  Use Grid
  Use Parameters
  Use EVD_Operators
  Use Thomas_coefficients
        
  Implicit Real(kind=8) (A-H,O-Z)

! ================== X - direction ==============================
        D2_dx2 = 0.D0

	Do  i=1,Nx
                      P1 = 1.D0 / ( HPx(i) * Hx12(i-1) )
                      P2 = 1.D0 / ( HPx(i) * Hx12( i ) )
                                    
                      if (i/=1)	  D2_dx2(i,i-1) = P1
                      if (i/=Nx)  D2_dx2(i,i+1) = P2

      	              D2_dx2(i,i)   = -(P1+P2) 
 
                      if (i/=1)	  Vx_left(i)  = D2_dx2(i,i-1) * DGr
                      if (i/=Nx) Vx_right(i)  = D2_dx2(i,i+1) * DGr 
                                Vx_center(i)  = D2_dx2(i,i  ) * DGr 
    End do

!	call Vgeev (D2_dx2(1:Nx,1:Nx),ExVx(1:Nx,1:Nx),   &
!	&              Ex_invVx(1:Nx,1:Nx),LambxVx(1:Nx),Nx)

!	LambxVx = LambxVx * DGr
	
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
	 
	 LambyVx = LambyVx * DGr
          
! ================= Z - direction =================================
     
        D2_dz2 = 0.D0
  
    Do i=1,Nz1
                    P1 = 1.D0 / ( HPz(i-1) * Hz12(i-1) )
                    P2 = 1.D0 / ( HPz( i ) * Hz12(i-1) )

                    if (i/=1)	D2_dz2(i,i-1) = P1                
                    if (i/=Nz1)	D2_dz2(i,i+1) = P2
                
                    D2_dz2(i,i)   = -(P1+P2) 
 	End do

	Call Vgeev  (D2_dz2(1:Nz1,1:Nz1),EzVx(1:Nz1,1:Nz1),   &
	&               Ez_invVx(1:Nz1,1:Nz1),LambzVx(1:Nz1),Nz1)
	 
	 LambzVx = LambzVx * DGr

End Subroutine  EVDLapVx
  