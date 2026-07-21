Subroutine Get_Potential
    Use Grid; Use Numbers; Use Numerica; Use Operators; Use Variables
    Use AlexCudaCompatibility, only : PotentialHandle
    Use eigenbcgsolver_eigen_mod, only : solve_eigen_decomp_d
    Implicit Real(kind=8) (A-H,O-Z)

    Real(kind=8), Dimension(Ny1, Nz, Nx) :: Potential_flat
    Real(kind=8), Dimension(Ny1, Nz, Nx) :: FDRHP_flat

    !$OMP Parallel Do Private(i,j,k,DVx_dz,DVz_dx)
    Do j=1,Ny1
        Do k=1,Nz
            Do i=1,Nx
                DVx_dz = ( VMx(j,k+1,i) - VMx(j,k,i) ) / HPz(k)
                DVz_dx = ( VMz(j,k,i+1) - VMz(j,k,i) ) / HPx(i)
                FDRHP(j,k,i) = DVz_dx - DVx_dz
            End Do
        End Do
    End Do

    FDRHP_flat     = FDRHP(1:Ny1, 1:Nz, 1:Nx)
    Potential_flat = Potential(1:Ny1, 1:Nz, 1:Nx)

    Call solve_eigen_decomp_d(PotentialHandle, Potential_flat, FDRHP_flat)

    Potential(1:Ny1, 1:Nz, 1:Nx) = Potential_flat

    Potential = Potential - Potential(1,1,1)

    If(EVD_Pot_X == 1 ) then
        Potential(:,:,0)   = Potential(:,:,1)
        Potential(:,:,Nx1) = Potential(:,:,Nx)
    else
        Potential(:,:,0)   = 0.d0
        Potential(:,:,Nx1) = 0.d0
    End If

    If(EVD_Pot_Y == 1 ) then
        Potential(0,:,:)   = Potential(1,:,:)
        Potential(Ny2,:,:) = Potential(Ny1,:,:)
    else
        Potential(0,:,:)   = 0.d0
        Potential(Ny2,:,:) = 0.d0
    End If

    If(EVD_Pot_Z == 1 ) then
        Potential(:,0,:)   = Potential(:,1,:)
        Potential(:,Nz1,:) = Potential(:,Nz,:)
    else
        Potential(:,0,:)   = 0.d0
        Potential(:,Nz1,:) = 0.d0
    End If

    Return
End Subroutine Get_Potential

! ===================================================================

Subroutine EM_force
    Use Numbers
    Use Parameters
    Use Grid
    Use Operators
    Use Variables
    Implicit Real(kind=8) (A-H,O-Z)

    Coef = DGr * (Hartmann * WidRa)**2

    !$OMP Parallel Do Private(i,j,k,DFi_dz)
    Do j=1,Ny1
        Do k=1,Nz1
            Do i=1,Nx
                DFi_dz = ( Potential(j,k,i) - Potential(j,k-1,i) ) / Hz12(k-1)
                RHSx(j,k,i) = RHSx(j,k,i) + Coef * ( DFi_dz + VMx(j,k,i) )
            End Do
        End Do
    End Do

    !$OMP Parallel Do Private(i,j,k,DFi_dx)
    Do j=1,Ny1
        Do k=1,Nz
            Do i=1,Nx1
                DFi_dx = ( Potential(j,k,i) - Potential(j,k,i-1) ) / Hx12(i-1)
                RHSz(j,k,i) = RHSz(j,k,i) - Coef * ( DFi_dx - VMz(j,k,i) )
            End Do
        End Do
    End Do

End Subroutine EM_force