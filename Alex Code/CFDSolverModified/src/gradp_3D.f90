! ***************************************************
! *   Finite difference approximation of Grad(P)    *
! *   Updated for (Y, Z, X) Memory Layout           *
! ***************************************************

Subroutine GradPx(ResX, Pressure)
    Use Numbers
    Use Grid
    Use Variables
    Use Operators
    Implicit Real(kind=8) (A-H,O-Z)

    ! New Dimension Order: (Ny1, Nz1, Nx)
    Real(kind=8), Dimension(Ny1, Nz1, Nx)  :: ResX
    Real(kind=8), Dimension(Ny1, Nz1, Nx1) :: Pressure

    !$OMP Parallel Do Private(i,j,k)
    Do i=1,Nx
        Do k=1,Nz1
            Do j=1,Ny1
                ResX(j,k,i) = ( Pressure(j,k,i+1) - Pressure(j,k,i) ) / HPx(i)
            End Do
        End Do
    End Do
    Return
End Subroutine GradPx

Subroutine GradPy(ResY, Pressure)
    Use Numbers
    Use Grid
    Use Variables
    Use Operators
    Implicit Real(kind=8) (A-H,O-Z)

    ! New Dimension Order: (Ny, Nz1, Nx1)
    Real(kind=8), Dimension(Ny, Nz1, Nx1)  :: ResY
    Real(kind=8), Dimension(Ny1, Nz1, Nx1) :: Pressure

    !$OMP Parallel Do Private(i,j,k)
    Do i=1,Nx1
        Do k=1,Nz1
            Do j=1,Ny
                ResY(j,k,i) = ( Pressure(j+1,k,i) - Pressure(j,k,i) ) / HPy(j)
            End Do
        End Do
    End Do
    Return
End Subroutine GradPy

Subroutine GradPz(ResZ, Pressure)
    Use Numbers
    Use Grid
    Use Variables
    Use Operators
    Implicit Real(kind=8) (A-H,O-Z)

    ! New Dimension Order: (Ny1, Nz, Nx1)
    Real(kind=8), Dimension(Ny1, Nz, Nx1)  :: ResZ
    Real(kind=8), Dimension(Ny1, Nz1, Nx1) :: Pressure

    !$OMP Parallel Do Private(i,j,k)
    Do i=1,Nx1
        Do k=1,Nz
            Do j=1,Ny1
                ResZ(j,k,i) = ( Pressure(j,k+1,i) - Pressure(j,k,i) ) / HPz(k)
            End Do
        End Do
    End Do
    Return
End Subroutine GradPz