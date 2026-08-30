! ******************* pointw **************************
! *                                                   *
!  Writing a 3D array in the Point format (TECPLOT)   *
! *                                                   *
! *****************************************************


         Subroutine Point_Write ( Nx, Ny, Nz, FFF, X, Y, Z, Nfile, Name)
         
           Implicit Real(kind=8) (a-h,o-z)
           
           Real(kind=8)  FFF(0:Nx,0:Ny,0:Nz), X(0:Nx), Y(0:Ny), Z(0:Nz)
           
           Character(Len=10)   Name
           
! ==============================================================

! ******* Write the header ***********************

           Nqx = Nx + 1
           Nqy = Ny + 1
           Nqz = Nz + 1

           Write (Nfile,  *) 'VARIABLES = "X","Y","Z","',Name,'"'
           Write (Nfile,301) Nqx, Nqy, Nqz
             
! ********* Write the result *******************

           Do k=0,Nz
            Do j=0,Ny
             Do i=0,Nx
                Write (Nfile,310) X(i), Y(j), Z(k), FFF(i,j,k)
             End Do
            End Do
           End Do


         Return
301        Format ('ZONE F=POINT, I=',I4, ', J=',I4, ', K=',I4)
310        Format ( 3(E12.5,1x) )
         End
         
