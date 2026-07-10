Subroutine TimeStep ( Istp, RNSx, RNSy, RNSz, RTmpr, RDP )

    Use Numbers
    Use Parameters
    Use Numerica
    Use Grid
    Use Operators
    Use Variables
    Use AlexCudaCompatibility, only : TemperatureHandle, VxHandle, VyHandle, VzHandle, PressureHandle
    Use eigenbcgsolver_eigen_mod, only : solve_eigen_decomp_d

    Implicit Real(kind=8) (A-H,O-Z)

    Real(kind=8), Allocatable, Save :: GPU_FDRHP_T(:,:,:), GPU_TmpNew_T(:,:,:)
    Real(kind=8), Allocatable, Save :: GPU_RHSx(:,:,:), GPU_VMxNew(:,:,:)
    Real(kind=8), Allocatable, Save :: GPU_RHSy(:,:,:), GPU_VMyNew(:,:,:)
    Real(kind=8), Allocatable, Save :: GPU_RHSz(:,:,:), GPU_VMzNew(:,:,:)
    Real(kind=8), Allocatable, Save :: GPU_FDRHP_P(:,:,:), GPU_Dprs(:,:,:)

    Ht = 2.D0 * Htime
    dt_temp = Dble(Istat)

    If (.Not. Allocated(GPU_FDRHP_T)) Then
        Allocate(GPU_FDRHP_T(1:Ny1,1:Nz1,1:Nx1), GPU_TmpNew_T(1:Ny1,1:Nz1,1:Nx1))
        Allocate(GPU_RHSx(1:Ny1,1:Nz1,1:Nx), GPU_VMxNew(1:Ny1,1:Nz1,1:Nx))
        Allocate(GPU_RHSy(1:Ny,1:Nz1,1:Nx1), GPU_VMyNew(1:Ny,1:Nz1,1:Nx1))
        Allocate(GPU_RHSz(1:Ny1,1:Nz,1:Nx1), GPU_VMzNew(1:Ny1,1:Nz,1:Nx1))
        Allocate(GPU_FDRHP_P(1:Ny1,1:Nz1,1:Nx1), GPU_Dprs(1:Ny1,1:Nz1,1:Nx1))
    End If

    FDRHP = 0.D0
    Call VgrTmp

    FDRHP(1:Ny1,1:Nz1,1:Nx1) = FDRHP(1:Ny1,1:Nz1,1:Nx1) - &
            &              dt_temp * ( 4.D0 * Tmpr(1:Ny1,1:Nz1,1:Nx1) - &
                    &                               TmpOld(1:Ny1,1:Nz1,1:Nx1)     ) / Ht

    GPU_FDRHP_T(1:Ny1,1:Nz1,1:Nx1) = FDRHP(1:Ny1,1:Nz1,1:Nx1)
    GPU_TmpNew_T(1:Ny1,1:Nz1,1:Nx1) = TmpNew(1:Ny1,1:Nz1,1:Nx1) ! <-- Fixed: Initialize to prevent NaNs

    Call solve_eigen_decomp_d( &
            TemperatureHandle, &
            GPU_TmpNew_T, &
            GPU_FDRHP_T)
    TmpNew(1:Ny1,1:Nz1,1:Nx1) = GPU_TmpNew_T(1:Ny1,1:Nz1,1:Nx1)

    Call EVDbounds

    RTmpr = Dist2D (TmpNew, Tmpr, Ny2, Nz2, Nx2, Ny2, Nz2, Nx2)

    Call Get_Potential

    FDRHP = 0.D0
    Call VgrdVx
    Call GradPx( RHSx(1:Ny1,1:Nz1,1:Nx), Prs )
    RHSx(1:Ny1,1:Nz1,1:Nx) = RHSx(1:Ny1,1:Nz1,1:Nx) + FDRHP(1:Ny1,1:Nz1,1:Nx)

    FDRHP = 0.D0
    Call VgrdVy
    Call GradPy( RHSy(1:Ny,1:Nz1,1:Nx1), Prs )
    RHSy(1:Ny,1:Nz1,1:Nx1) = RHSy(1:Ny,1:Nz1,1:Nx1) + FDRHP(1:Ny,1:Nz1,1:Nx1)

    FDRHP = 0.D0
    Call VgrdVz
    Call GradPz( RHSz(1:Ny1,1:Nz,1:Nx1), Prs )
    RHSz(1:Ny1,1:Nz,1:Nx1) = RHSz(1:Ny1,1:Nz,1:Nx1) + FDRHP(1:Ny1,1:Nz,1:Nx1)

    RHSz(1:Ny1,1:Nz,1:Nx1) = RHSz(1:Ny1,1:Nz,1:Nx1)  &
            &     - 0.5d0 * Bu_Gr * ( TmpNew(1:Ny1,1:Nz,1:Nx1) + TmpNew(1:Ny1,2:Nz1,1:Nx1) ) &
            &     - 0.5d0 * Bu_Gr * (   Teta(1:Ny1,1:Nz,1:Nx1) +   Teta(1:Ny1,2:Nz1,1:Nx1) )

    RHSx(1:Ny1,1:Nz1,1:Nx) = RHSx(1:Ny1,1:Nz1,1:Nx) - &
            &                  ( 4.D0 * VMx(1:Ny1,1:Nz1,1:Nx) - VMxOld(1:Ny1,1:Nz1,1:Nx) )/ Ht

    RHSy(1:Ny,1:Nz1,1:Nx1) = RHSy(1:Ny,1:Nz1,1:Nx1) - &
            &                  ( 4.D0 * VMy(1:Ny,1:Nz1,1:Nx1) - VMyOld(1:Ny,1:Nz1,1:Nx1) )/ Ht

    RHSz(1:Ny1,1:Nz,1:Nx1) = RHSz(1:Ny1,1:Nz,1:Nx1) - &
            &                  ( 4.D0 * VMz(1:Ny1,1:Nz,1:Nx1) - VMzOld(1:Ny1,1:Nz,1:Nx1) )/ Ht

    Call EM_force

    GPU_RHSx(1:Ny1,1:Nz1,1:Nx) = RHSx(1:Ny1,1:Nz1,1:Nx)
    GPU_VMxNew(1:Ny1,1:Nz1,1:Nx) = VMxNew(1:Ny1,1:Nz1,1:Nx) ! <-- Fixed: Initialize
    Call solve_eigen_decomp_d( &
            VxHandle, &
            GPU_VMxNew, &
            GPU_RHSx)
    VMxNew(1:Ny1,1:Nz1,1:Nx) = GPU_VMxNew(1:Ny1,1:Nz1,1:Nx)

    GPU_RHSy(1:Ny,1:Nz1,1:Nx1) = RHSy(1:Ny,1:Nz1,1:Nx1)
    GPU_VMyNew(1:Ny,1:Nz1,1:Nx1) = VMyNew(1:Ny,1:Nz1,1:Nx1) ! <-- Fixed: Initialize
    Call solve_eigen_decomp_d( &
            VyHandle, &
            GPU_VMyNew, &
            GPU_RHSy)
    VMyNew(1:Ny,1:Nz1,1:Nx1) = GPU_VMyNew(1:Ny,1:Nz1,1:Nx1)

    GPU_RHSz(1:Ny1,1:Nz,1:Nx1) = RHSz(1:Ny1,1:Nz,1:Nx1)
    GPU_VMzNew(1:Ny1,1:Nz,1:Nx1) = VMzNew(1:Ny1,1:Nz,1:Nx1) ! <-- Fixed: Initialize
    Call solve_eigen_decomp_d( &
            VzHandle, &
            GPU_VMzNew, &
            GPU_RHSz)
    VMzNew(1:Ny1,1:Nz,1:Nx1) = GPU_VMzNew(1:Ny1,1:Nz,1:Nx1)

    Call EVDbounds

    FDRHP= 0.d0
    Call FdDiv

    FDRHP = FDRHP * Ckor / Htime

    GPU_FDRHP_P(1:Ny1,1:Nz1,1:Nx1) = FDRHP(1:Ny1,1:Nz1,1:Nx1)
    GPU_Dprs(1:Ny1,1:Nz1,1:Nx1) = Dprs(1:Ny1,1:Nz1,1:Nx1) ! (You already had this one right!)
    Call solve_eigen_decomp_d( &
            PressureHandle, &
            GPU_Dprs, &
            GPU_FDRHP_P)
    Dprs(1:Ny1,1:Nz1,1:Nx1) = GPU_Dprs(1:Ny1,1:Nz1,1:Nx1)

    Call GradPx( RHSx(1:Ny1,1:Nz1,1:Nx), Dprs )
    Call GradPy( RHSy(1:Ny,1:Nz1,1:Nx1), Dprs )
    Call GradPz( RHSz(1:Ny1,1:Nz,1:Nx1), Dprs )

    VMxNew(1:Ny1,1:Nz1,1:Nx) = VMxNew(1:Ny1,1:Nz1,1:Nx) - RHSx(1:Ny1,1:Nz1,1:Nx) * Htime / Ckor
    VMyNew(1:Ny,1:Nz1,1:Nx1) = VMyNew(1:Ny,1:Nz1,1:Nx1) - RHSy(1:Ny,1:Nz1,1:Nx1) * Htime / Ckor
    VMzNew(1:Ny1,1:Nz,1:Nx1) = VMzNew(1:Ny1,1:Nz,1:Nx1) - RHSz(1:Ny1,1:Nz,1:Nx1) * Htime / Ckor

    Prs = Prs + DPrs

    Call EVDbounds

    RNSx = Dist2D (VMx, VMxNew, Ny2, Nz2, Nx2, Ny2, Nz2, Nx2)
    RNSy = Dist2D (VMy, VMyNew, Ny2, Nz2, Nx2, Ny2, Nz2, Nx2)
    RNSz = Dist2D (VMz, VMzNew, Ny2, Nz2, Nx2, Ny2, Nz2, Nx2)

    RDP = MaxVal(Abs(Dprs) )

    FDRHP= 0.d0
    Call FdDiv

    If (Icheck .EQ. 0)  Call Check

    444 Continue

    VMxOld = VMx
    VMyOld = VMy
    VMzOld = VMz
    TmpOld = Tmpr

    VMx  = VMxNew
    VMy  = VMyNew
    VMz  = VMzNew
    Tmpr = TmpNew

    Return
End Subroutine TimeStep