! ***************************************************
! *  Eigen Value decomposition of Lap(Temp)         *
! *                                                 *
! ***************************************************
    
Subroutine  EVDLapTmpr
  Use Numbers
  Use Grid
  Use EVD_Operators
  Use Thomas_coefficients

  Implicit Real(kind=8) (A-H,O-Z)

! ====================================================================

! ======================= X-direction ================================

            D2_d2 = 0.D0

	    Do  i=1,Nx1
            P1 = 1.D0 /( Hx12(i-1) * HPx(i-1) )
            P2 = 1.D0 /( Hx12(i-1) * HPx( i ) )
                   
            if (i/=1)	D2_d2(i,i-1) = P1
            if (i/=Nx1) D2_d2(i,i+1) = P2

      	    D2_d2(i,i)   = -(P1+P2) 

            if (EVD_BCx == 1) then
                   if (i==1)   D2_d2(i,i)   = -P2 
                   if (i==Nx1) D2_d2(i,i)   = -P1
            end if

            if (i/=1)   T_left(i)  = D2_d2(i,i-1) 
            if (i/=Nx1) T_right(i) = D2_d2(i,i+1) 
                       T_center(i) = D2_d2(i,i  ) 
        End Do
   
	call Vgeev (D2_d2(1:Nx1,1:Nx1),ExTemp(1:Nx1,1:Nx1),  &
	    &          Ex_invTemp(1:Nx1,1:Nx1),LambxTemp(1:Nx1),Nx1)
	 
!	 LambxTemp = LambxTemp /GrPr
	
! ===================== Y-direction ============================
     
            D2_d2 = 0.D0
  
        Do i=1,Ny1
            P1 = 1.D0 /( Hy12(i-1) * HPy(i-1) )
            P2 = 1.D0 /( Hy12(i-1) * HPy( i ) )

            if (i/=1)	D2_d2(i,i-1) = P1                
            if (i/=Ny1)	D2_d2(i,i+1) = P2
                
            D2_d2(i,i)   = -(P1+P2) 

            if (EVD_BCy == 1) then
                  if (i==1)   D2_d2(i,i)   = -P2
                  if (i==Ny1) D2_d2(i,i)   = -P1   
            end if 
 	    End Do

        Call Vgeev  (D2_d2(1:Ny1,1:Ny1),EyTemp(1:Ny1,1:Ny1),    &
	     &          Ey_invTemp(1:Ny1,1:Ny1),LambyTemp(1:Ny1),Ny1)
	 
!	    LambyTemp = LambyTemp /GrPr

! ===================== Z-direction ============================
     
            D2_d2 = 0.D0
  
        Do i=1,Nz1
            P1 = 1.D0 /( Hz12(i-1) * HPz(i-1) )
            P2 = 1.D0 /( Hz12(i-1) * HPz( i ) )

            if (i/=1)	  D2_d2(i,i-1) = P1                
            if (i/=Nz1) D2_d2(i,i+1) = P2
                
            D2_d2(i,i)   = -(P1+P2) 

            if (EVD_BCz == 1) then
                   if (i==1)   D2_d2(i,i)   = -P2
                   if (i==Nz1) D2_d2(i,i)   = -P1   
            end if 
 	
 	  End Do

	    Call Vgeev  (D2_d2(1:Nz1,1:Nz1),EzTemp(1:Nz1,1:Nz1),    &
	     &          Ez_invTemp(1:Nz1,1:Nz1),LambzTemp(1:Nz1),Nz1)
	 
!	    LambzTemp = LambzTemp /GrPr
 
End Subroutine  EVDLapTmpr
