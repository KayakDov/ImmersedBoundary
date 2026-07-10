! ************************************************************
! *     Updated Variable and Operator Dimensions             *
! *     New Layout: (Y-axis, Z-axis, X-axis)                 *
! ************************************************************

Module Size
    Parameter (Nbig=50, Nxx=200, Nyy=1000, Nzz=200)
    Parameter (Nxx1=Nxx+1, Nxx2=Nxx+2)
    Parameter (Nyy1=Nyy+1, Nyy2=Nyy+2)
    Parameter (Nzz1=Nzz+1, Nzz2=Nzz+2)
    Parameter ( Nmax=max(Nxx2,Nyy2,Nzz2) )
End Module Size

Module Numbers
    Integer :: Nx, Nx1, Nx2, Ny, Ny1, Ny2, Nz, Nz1, Nz2
End Module Numbers

Module Parameters
    Real(kind=8) ::  AspRa, WidRa, Gr, Prandtl, Am, Bi, DGr, Hartmann, Bu_Gr
End Module Parameters

Module Numerica
    Real(kind=8) :: Eps, EpsCnv, Htime, Hmax, Time, TimCur, Tstart, Ckor
    Integer :: ItMax, Niter, Iprint, Icheck, I_matrix_C, inner
    Integer :: Iexcl, Istat, EVD_BCx, EVD_BCy, EVD_BCz
    Integer :: EVD_Pot_X, EVD_Pot_Y, EVD_Pot_Z
    Integer :: I_Fourier, N_Fourier
End Module Numerica

! ......... Arrays for definition of the mesh ............
! Note: These 1D arrays remain unchanged as they map to single coordinate axes.
Module Grid
    Use Size
    Real(kind=8) X(0:Nxx1), X12(0:Nxx2), Hx12(0:Nxx), HPx(0:Nxx1)
    Real(kind=8) Y(0:Nyy1), Y12(0:Nyy2), Hy12(0:Nyy), HPy(0:Nyy1)
    Real(kind=8) Z(0:Nzz1), Z12(0:Nzz2), Hz12(0:Nzz), HPz(0:Nzz1)
End Module Grid

! ......... Arrays for current values of functions (Now (Y, Z, X)) ........

Module Variables
    Use Size

    ! New dimension order: (Ny_dim, Nz_dim, Nx_dim)
    Real(kind=8), Allocatable, Dimension(:,:,:) :: VMxOld, VMx,  VMxNew, VMx_Av
    Real(kind=8), Allocatable, Dimension(:,:,:) :: VMyOld, VMy,  VMyNew, VMy_Av
    Real(kind=8), Allocatable, Dimension(:,:,:) :: VMzOld, VMz,  VMzNew, VMz_Av
    Real(kind=8), Allocatable, Dimension(:,:,:) :: TmpOld, Tmpr, TmpNew, Teta, Tmp_Av
    Real(kind=8), Allocatable, Dimension(:,:,:) :: Prs,    Dprs, Potential, Prs_Av

    Real(kind=8), Allocatable, Dimension(:,:,:,:) :: Tmp_Amplitude
    Real(kind=8), Allocatable, Dimension(:)       :: Omega

    Real(kind=8), Allocatable, Dimension(:,:,:) :: GPU_FDRHP
    Real(kind=8), Allocatable, Dimension(:,:,:) :: GPU_TmpNew

End Module Variables

! ........... Finite difference operators (Now (Y, Z, X)) ..........................

Module Operators
    Use Size

    ! New dimension order: (Ny_dim, Nz_dim, Nx_dim)
    Real(kind=8), Allocatable, Dimension(:,:,:) :: FDRHP, FDRHP1
    Real(kind=8), Allocatable, Dimension(:,:,:) :: RHSx
    Real(kind=8), Allocatable, Dimension(:,:,:) :: RHSy
    Real(kind=8), Allocatable, Dimension(:,:,:) :: RHSz

End Module Operators

! ... (EVD_Operators and Thomas_coefficients remain the same as they use fixed 1D/2D shapes) ...