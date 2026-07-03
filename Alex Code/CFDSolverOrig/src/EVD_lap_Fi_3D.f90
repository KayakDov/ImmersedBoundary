! ***************************************************
! *  Eigen Value decomposition of Lap(P)         *
! *                                                 *
! ***************************************************
Subroutine  EVD_Fi
 
   Use Numbers
   Use Numerica
   Use Grid
   Use Parameters
   Use EVD_Operators
   Use Thomas_coefficients

   Implicit Real(kind=8) (A-H,O-Z)

! ====================== X-direction ===============================
   
        D2_dx2 = 0.D0
        
        Write (*,*) 'EVD_Pot=', EVD_Pot

!$OMP Parallel Do Private(i, P1, P2)
	Do  i=1,Nx
                      P1 = 1.D0 /( Hx12(i-1) * HPx(i) )
                      P2 = 1.D0 /( Hx12(i  ) * HPx(i) )
                      
                      if (i/= 1) D2_dx2(i,i-1) =   P1
                      if (i/=Nx) D2_dx2(i,i+1) =        P2
                                 D2_dx2(i,i  ) = -(P1 + P2) 

                if (EVD_Pot_X == 1) then
                   if (i==1)  D2_dx2(i,i)   = -P2 
                   if (i==Nx) D2_dx2(i,i)   = -P1
                end if

                      if (i/= 1) Fi_left(i)   = D2_dx2(i,i-1) 
                      if (i/=Nx) Fi_right(i)  = D2_dx2(i,i+1) 
                                 Fi_center(i) = D2_dx2(i,i  )   
        End do
   
	Call Vgeev ( D2_dx2(1:Nx,1:Nx), ExxFi(1:Nx,1:Nx), Ex_invFi(1:Nx,1:Nx), LambxFi(1:Nx), Nx)
    	    
! ==================== Y-direction =============================
  
        D2_dy2 = 0.D0
  
!$OMP Parallel Do Private(i, P1, P2)
       Do i=1,Ny1
                      P1 = 1.D0 /( Hy12(i-1) * HPy(i-1) )
                      P2 = 1.D0 /( Hy12(i-1) * HPy( i ) )

                      if (i/=1  ) D2_dy2(i,i-1) =   P1                
                      if (i/=Ny1) D2_dy2(i,i+1) =        P2
                                  D2_dy2(i,i  ) = -(P1 + P2) 

                if (EVD_Pot_Y == 1) then
                  if (i==1)   D2_dy2(i,i)   = -P2
                  if (i==Ny1) D2_dy2(i,i)   = -P1   
                end if 
 	End do

	Call Vgeev (D2_dy2(1:Ny1,1:Ny1), EyFi(1:Ny1,1:Ny1), Ey_invFi(1:Ny1,1:Ny1), LambyFi(1:Ny1), Ny1)
    
!    Write (*,*) ' D2_Dy2=', Sum(D2_dy2)
!    Write (*,*) ' EyFi=', Sum(EyFi)
!    Write (*,*) ' Ey_invFi=', Sum(Ey_invFi)
!    Write (*,*) ' LambyFi=', Sum(LambyFi)
	 
! ==================== Z-direction =============================
  
        D2_dz2 = 0.D0
  
!$OMP Parallel Do Private(i, P1, P2)
       Do i=1,Nz
                      P1 = 1.D0 /( Hz12(i-1) * HPz(i) )
                      P2 = 1.D0 /( Hz12(i  ) * HPz(i) )

                      if (i/= 1) D2_dz2(i,i-1) =   P1                
                      if (i/=Nz) D2_dz2(i,i+1) =        P2
                                 D2_dz2(i,i  ) = -(P1 + P2) 

               if (EVD_Pot_Z == 1) then
                   if (i==1)   D2_dz2(i,i)   = -P2
                   if (i==Nz) D2_dz2(i,i)   = -P1   
               end if 
      End do

	Call Vgeev (D2_dz2(1:Nz,1:Nz), EzFi(1:Nz,1:Nz), Ez_invFi(1:Nz,1:Nz), LambzFi(1:Nz), Nz)
 !   Write (*,*) ' D2_Dz2=', Sum(D2_dz2)
 !   Write (*,*) ' EzFi=', Sum(EzFi)
 !   Write (*,*) ' Ez_invFi=', Sum(Ez_invFi)
 !   Write (*,*) ' LambzFi=', Sum(LambzFi)

!   Open(87, file='Eigenvalues.dat')

!    Do i=1,Nx
!      Write (87,*) LambxFi(i)
!    End Do

!    Do i=1,Ny1
!      Write (87,*) LambyFi(i)
!    End Do

!        Do i=1,Nz
!      Write (87,*) LambzFi(i)
!    End Do
    
!    Write (87,*) 'Ex=', Sum(ExxFi), Sum(Ex_invFi)
!    Write (87,*) 'Ey=', Sum(EyFi), Sum(Ey_invFi)
!    Write (87,*) 'Ez=', Sum(EzFi), Sum(Ez_invFi)
End Subroutine  EVD_Fi
