! .......... Calculate electic potential .......................
!
!   This potential is defined in the points X, Yp, Z

Subroutine Get_Potential
    Use Grid
    Use Numbers
    Use Numerica
    Use Operators
    Use Variables
    Use EVD_Operators
    Use Thomas_coefficients

    Implicit Real(kind=8) (A-H,O-Z)
! ___________________________________________________
         

!$OMP Parallel Do Private(i,j,k,DVx_dz,Dvz_dx)
   Do i=1,Nx
     Do j=1,Ny1
      Do k=1,Nz
          DVx_dz = ( VMx(i,j,k+1) - VMx(i,j,k) ) / HPz(k)
          DVz_dx = ( VMz(i+1,j,k) - VMz(i,j,k) ) / HPx(i)
          
          FDRHP(i,j,k) = DVz_dx - DVx_dz 
      End Do
     End Do
   End Do
    
        Call  EVDmethod (Potential(1:Nx,1:Ny1,1:Nz), FDRHP(1:Nx,1:Ny1,1:Nz), &
 &                           ExxFi(1:Nx,1:Nx), Ex_invFi(1:Nx,1:Nx), &
 &                           EyFi(1:Ny1,1:Ny1),  Ey_invFi(1:Ny1,1:Ny1), &
 &                           EzFi(1:Nz,1:Nz),  Ez_invFi(1:Nz,1:Nz), &
 &                           LambxFi(1:Nx), LambyFi(1:Ny1), LambzFi(1:Nz), Nx, Ny1, Nz, 1.D0, 1.D0, 1.D0, 0.D0)

!      Call   EVD_Thomas (Potential(1:Nx,1:Ny1,1:Nz), FDRHP(1:Nx,1:Ny1,1:Nz),  &
! &                           EyFi(1:Ny1,1:Ny1),  Ey_invFi(1:Ny1,1:Ny1),       &
! &                           EzFi(1:Nz,1:Nz),  Ez_invFi(1:Nz,1:Nz),           &
! &                           LambyFi(1:Ny1), LambzFi(1:Nz),                   &
! &                           Fi_left(1:Nx), Fi_center(1:Nx), Fi_right(1:Nx),  &
! &                           Nx, Ny1, Nz, 0.D0)
      
        Potential = Potential - Potential(1,1,1)

    If(EVD_Pot_X == 1 ) then
        Potential(0,:,:) = Potential(1,:,:);  Potential(Nx1,:,:) = Potential(Nx, :,:)
     else
        Potential(0,:,:) = 0.d0;  Potential(Nx1,:,:) = 0.d0
    End If   

     If(EVD_Pot_Y == 1 ) then
       Potential(:,0,:) = Potential(:,1,:);  Potential(:,Ny2,:) = Potential(:,Ny1,:)
     else
        Potential(:,0,:) = 0.d0;  Potential(:,Ny2,:) = 0.d0
    End If   
       
     If(EVD_Pot_Z == 1 ) then
        Potential(:,:,0) = Potential(:,:,1);  Potential(:,:,Nz1) = Potential(:, :,Nz)
     else
        Potential(:,:,0) = 0.d0;  Potential(:,:,Nz1) = 0.d0
    End If   

 !   Write (*,*) ' EM: Potential=', Sum(Potential)
  Return
End Subroutine Get_Potential
    
! .......... Calculate electromagnetic force .......................

Subroutine EM_force
    Use Numbers
    Use Parameters
    Use Grid
    Use Operators
    Use Variables

    Implicit Real(kind=8) (A-H,O-Z)
! ___________________________________________________

         Coef = DGr * (Hartmann * WidRa)**2

!$OMP Parallel Do Private(i,j,k,DFi_dz)
   Do i=1,Nx
     Do j=1,Ny1
      Do k=1,Nz1
          DFi_dz = ( Potential(i,j,k) - Potential(i,j,k-1) ) / Hz12(k-1)
          
          RHSx(i,j,k) =  RHSx(i,j,k) + Coef * ( DFi_dz + VMx(i,j,k) )
      End Do
     End Do
   End Do
 !  Write (*,*) ' EM: RHSx=', Sum(RHSx)

!$OMP Parallel Do Private(i,j,k,DFi_dx)
   Do i=1,Nx1
     Do j=1,Ny1
      Do k=1,Nz
          DFi_dx = ( Potential(i,j,k) - Potential(i-1,j,k) ) / Hx12(i-1)

         RHSz(i,j,k) = RHSz(i,j,k) - Coef * ( DFi_dx -VMz(i,j,k) )
      End Do
     End Do
    End Do
!   Write (*,*) ' EM: RHSz=', Sum(RHSz)

! ........... Make div-free force .........................

!        Call EVDbounds_V(Work_flow) 
!        Call Make_divfree(Stream, 0)

!        Work_flow%P = 0.d0

End Subroutine EM_force
