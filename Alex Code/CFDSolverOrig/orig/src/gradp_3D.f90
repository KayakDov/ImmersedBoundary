! ***************************************************
! *   Finite difference approximation of Grad(P)    *
! *                                                 *
! *   GradPx(Nx,Ny1) -    x-component of Grad(P)    *
! *   GradPy(Nx1,Ny) -    y-component of Grad(P)    *
! *   GradPz(Nx1,Ny) -    z-component of Grad(P)    *
! ***************************************************

        Subroutine  GradPx(ResX, Pressure)

         Use Numbers
         Use Grid
         Use Variables
         Use Operators

        Implicit Real(kind=8) (A-H,O-Z)

         Real(kind=8), Dimension(Nx ,Ny1,Nz1) :: ResX
         Real(kind=8), Dimension(Nx1,Ny1,Nz1) :: Pressure
! ===============================================================

!$OMP Parallel Do Private(i,j,k)
        Do i=1,Nx
         Do j=1,Ny1
          Do k=1,Nz1
            ResX(i,j,k) = ( Pressure(i+1,j,k) - Pressure(i,j,k) ) / HPx(i)
          End Do
         End Do
        End Do

        Return
        End


        Subroutine GradPy(ResY, Pressure) 

         Use Numbers
         Use Grid
         Use Variables
         Use Operators

        Implicit Real(kind=8) (A-H,O-Z)

         Real(kind=8), Dimension(Nx1,Ny ,Nz1) :: ResY
         Real(kind=8), Dimension(Nx1,Ny1,Nz1) :: Pressure
! =============================================================

 !$OMP Parallel Do Private(i,j,k)
       Do i=1,Nx1
         Do j=1,Ny
          Do k=1,Nz1
            ResY(i,j,k) = ( Pressure(i,j+1,k) - Pressure(i,j,k) ) / HPy(j)
          End Do
         End Do
        End Do

        Return
        End

        Subroutine GradPz(ResZ, Pressure) 

         Use Numbers
         Use Grid
         Use Variables
         Use Operators

        Implicit Real(kind=8) (A-H,O-Z)

         Real(kind=8), Dimension(Nx1,Ny1,Nz ) :: ResZ
         Real(kind=8), Dimension(Nx1,Ny1,Nz1) :: Pressure
! =============================================================

!$OMP Parallel Do Private(i,j,k)
        Do i=1,Nx1
         Do j=1,Ny1
          Do k=1,Nz
            ResZ(i,j,k) =( Pressure(i,j,k+1) - Pressure(i,j,k) ) / HPz(k)
          End Do
         End Do
        End Do

        Return
        End
