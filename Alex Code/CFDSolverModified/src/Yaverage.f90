Subroutine Average_flow
    Use Numbers
    Use Parameters
    Use Grid
    Use Variables

    Implicit Real(kind=8) (a-h,o-z)

    ! Note: Ux and Tr defined as (X, Z) -> (i, k)
    ! Psi and Tr are 2D
    Real(kind=8) :: Ux(0:Nx1,0:Nz2), Uz(0:Nx2,0:Nz1), Psi(0:Nx2,0:Nz2), Tr(0:Nx2,0:Nz2)

    ! =============================================================
    ! 1. Initialize accumulators
    Ux = 0.d0;  Uz = 0.d0;  Tr = 0.d0

    ! 2. Perform Y-averaging (Integration over j)
    ! VMx, VMz, Tmpr are (Y, Z, X) -> indices (j, k, i)
    Do j=1,Ny1
        ! Average for X-velocity (Ux)
        Do k=0,Nz2
            Do i=0,Nx1
                Ux(i,k) = Ux(i,k) + VMx(j,k,i) * Hy12(j-1)
            End Do
        End Do

        ! Average for Z-velocity (Uz)
        Do k=0,Nz1
            Do i=0,Nx2
                Uz(i,k) = Uz(i,k) + VMz(j,k,i) * Hy12(j-1)
            End Do
        End Do

        ! Average for Temperature (Tr)
        Do k=0,Nz2
            Do i=0,Nx2
                Tr(i,k) = Tr(i,k) + Tmpr(j,k,i) * Hy12(j-1)
            End Do
        End Do
    End Do

    ! 3. Normalize
    Ux = Ux / WidRa;   Uz = Uz / WidRa;  Tr = Tr / WidRa

    ! 4. Proceed to Psi calculations and Output
    Call PsiInt(Ux, Uz, Tr, Psi)

    Write (*,*) ' MaxPsi_average=', Maxval(abs( Psi ) ) / DGr

    Open(120, file='Psi_Yaverage.dat')
    Open(130, file='Tmpr_Yaverage.dat')

    Call Point_Write_2D ( Nx2, Nz2, Psi, X12, Z12, 120, 'Psi       ')
    Call Point_Write_2D ( Nx2, Nz2, Tr,  X12, Z12, 130, 'Tmpr      ')
    Return

    ! --- Internal Subroutine ---
Contains
    Subroutine PsiInt(Ux, Uz, Tr, Psi)
        ! Pass the calculated 2D arrays to PsiInt
        Real(kind=8), Intent(in) :: Ux(0:Nx1,0:Nz2), Uz(0:Nx2,0:Nz1), Tr(0:Nx2,0:Nz2)
        Real(kind=8), Intent(out) :: Psi(0:Nx2,0:Nz2)
        ! ... [Keep your existing Psi calculation logic here] ...

        ! Inside here, replace your Ux/Uz calculations with the passed arguments
        ! You can remove the internal loops inside PsiInt that recalculate averages.
    End Subroutine PsiInt
End Subroutine Average_flow