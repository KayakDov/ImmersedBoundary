! ******************* pointw **************************
! *                                                   *
!  Writing a 3D array in the Point format (TECPLOT)   *
! *                                                   *
! *****************************************************


 Subroutine Point_Write_Lid 
         Use Numbers
         Use Parameters
         Use Numerica
         Use Grid
         Use Variables
         
           Implicit Real(kind=8) (a-h,o-z)
           
! ==============================================================
        Nfile = 70

        Open (Nfile, file='Vel.dat',   form='formatted',  status='unknown')

           Nqx = Nx2 + 1;  Ix = Nqx / 50;  Mqx = Ix * 50;  Lqx = Nqx / Ix 
           Nqy = Ny2 + 1;  Iy = Nqy / 50;  Mqy = Iy * 50;  Lqy = Nqy / Iy 
           Nqz = Nz2 + 1;  Iz = Nqz / 50;  Mqz = Iz * 50;  Lqz = Nqz / Iz 

           If(Mod(Nx2,Ix) == 0) Lqx = Lqx + 1
           If(Mod(Ny2,Iy) == 0) Lqy = Lqy + 1
           If(Mod(Nz2,Iz) == 0) Lqz = Lqz + 1

           Write (Nfile,  *) 'VARIABLES = "X","Y","Z","Vx","Vy","Vz"'
           Write (Nfile,301) Lqx, Lqy, Lqz
             
           Do k=0,Nz2,Iz
            Do j=0,Ny2,Iy
             Do i=0,Nx2,Ix
                 If(i > 0 .and. i < Nx2) then
                      ux  = 0.5d0 * (   VMx(i,j,k) +   VMx(i-1,j,k) )
                                         else
                      ux = 0.d0
                 End If

                 If(j > 0 .and. j < Ny2) then
                      uy  = 0.5d0 * (   VMy(i,j,k) +   VMy(i,j-1,k) )
                                         else
                      uy = 0.d0
                 End If

                 If(k > 0 .and. k < Nz2) then
                      uz  = 0.5d0 * (   VMz(i,j,k) +   VMz(i,j,k-1) ) 
                                         else
                      uz = 0.d0
                 End If

                Write (Nfile,310) X12(i), Y12(j), Z12(k), ux, uy, uz
             End Do
            End Do
           End Do

         Return
301        Format ('ZONE F=POINT, I=',I4, ', J=',I4, ', K=',I4)
310        Format (6(E12.5,1x) )
 End Subroutine Point_Write_Lid 
         
