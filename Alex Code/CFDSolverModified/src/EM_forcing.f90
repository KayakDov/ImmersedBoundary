Subroutine Get_Potential
    Use Grid; Use Numbers; Use Numerica; Use Operators; Use Variables
    ! Potential intentionally uses Alex's CPU solver until the library's
    ! per-axis finite-volume (cell-centred) mode is in place: the y-axis of
    ! the Fi operator is cell-centred, where the current GPU stencil is a
    ! different (valid but non-matching) discretization.  Unlike temperature
    ! and the velocities there is no Helmholtz shift to damp the difference,
    ! and the potential feeds the Lorentz force at coupling ~ DGr*Ha^2, so it
    ! must reproduce Alex's operator exactly.  Same reasoning and same
    ! transpose pattern as the pressure solve in time_step_Q2D.f90.
    Use EVD_Operators, only : ExxFi, Ex_invFi, EyFi, Ey_invFi, EzFi, Ez_invFi, &
            LambxFi, LambyFi, LambzFi
    Implicit Real(kind=8) (A-H,O-Z)

    ! Alex's solver expects (x,y,z) index order; the CFD arrays are (y,z,x).
    Real(kind=8), Dimension(1:Nx, 1:Ny1, 1:Nz) :: CPU_Fi
    Real(kind=8), Dimension(1:Nx, 1:Ny1, 1:Nz) :: CPU_RHS_Fi

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

    Do k = 1, Nz
        Do j = 1, Ny1
            Do i = 1, Nx
                CPU_RHS_Fi(i,j,k) = FDRHP(j,k,i)
            End Do
        End Do
    End Do

    CPU_Fi = 0.D0

    Call EVDmethod (CPU_Fi, CPU_RHS_Fi, &
            ExxFi(1:Nx,1:Nx),   Ex_invFi(1:Nx,1:Nx), &
            EyFi(1:Ny1,1:Ny1),  Ey_invFi(1:Ny1,1:Ny1), &
            EzFi(1:Nz,1:Nz),    Ez_invFi(1:Nz,1:Nz), &
            LambxFi(1:Nx), LambyFi(1:Ny1), LambzFi(1:Nz), &
            Nx, Ny1, Nz, 1.D0, 1.D0, 1.D0, 0.D0)

    ! Return to the CFD array layout.
    Do k = 1, Nz
        Do j = 1, Ny1
            Do i = 1, Nx
                Potential(j,k,i) = CPU_Fi(i,j,k)
            End Do
        End Do
    End Do

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