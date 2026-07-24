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

         Module Grid
           Use Size

           Real(kind=8) X(0:Nxx1), X12(0:Nxx2), Hx12(0:Nxx), HPx(0:Nxx1)
           Real(kind=8) Y(0:Nyy1), Y12(0:Nyy2), Hy12(0:Nyy), HPy(0:Nyy1)
           Real(kind=8) Z(0:Nzz1), Z12(0:Nzz2), Hz12(0:Nzz), HPz(0:Nzz1)
         End Module Grid

! ......... Arrays for current values of functions ........

         Module Variables
           Use Size
   
          Real(kind=8), Allocatable, Dimension(:,:,:) :: VMxOld, VMx,  VMxNew, VMx_Av
          Real(kind=8), Allocatable, Dimension(:,:,:) :: VMyOld, VMy,  VMyNew, VMy_Av
          Real(kind=8), Allocatable, Dimension(:,:,:) :: VMzOld, VMz,  VMzNew, VMz_Av
          Real(kind=8), Allocatable, Dimension(:,:,:) :: TmpOld, Tmpr, TmpNew, Teta, Tmp_Av
          Real(kind=8), Allocatable, Dimension(:,:,:) :: Prs,    Dprs, Potential, Prs_Av
          
          Real(kind=8), Allocatable, Dimension(:,:,:,:) :: Tmp_Amplitude
          Real(kind=8), Allocatable, Dimension(:)       :: Omega
          
       End Module Variables

! ........... Finite difference operators ..........................

         Module Operators
           Use Size

! ......... Arrays for definition of FD equations ............

            Real(kind=8), Allocatable, Dimension(:,:,:) :: FDRHP, FDRHP1
            Real(kind=8), Allocatable, Dimension(:,:,:) :: RHSx
            Real(kind=8), Allocatable, Dimension(:,:,:) :: RHSy
            Real(kind=8), Allocatable, Dimension(:,:,:) :: RHSz  

         End Module Operators

! ........... Eigenvalues operators ..........................

         Module EVD_Operators
           Use Size
           
            Parameter (Nemax=max(Nxx2,Nyy2))
             
            Real(kind=16), dimension(1:Nxx1,1:Nxx1) :: D2_dx2
            Real(kind=16), dimension(1:Nyy1,1:Nyy1) :: D2_dy2
            Real(kind=16), dimension(1:Nzz1,1:Nzz1) :: D2_dz2
     
            Real(kind=8), dimension(1:Nyy1,1:Nyy1) :: EyTemp,Ey_invTemp
            Real(kind=8), dimension(1:Nyy1,1:Nyy1) :: EyVx,Ey_invVx
            Real(kind=8), dimension(1:Nyy,1:Nyy)   :: EyVy,Ey_invVy
            Real(kind=8), dimension(1:Nyy1,1:Nyy1) :: EyVz,Ey_invVz
            Real(kind=8), dimension(1:Nyy1,1:Nyy1) :: EyP,Ey_invP

            Real(kind=8), dimension(1:Nxx1,1:Nxx1) :: ExTemp,Ex_invTemp
            Real(kind=8), dimension(1:Nxx,1:Nxx)   :: ExVx,Ex_invVx
            Real(kind=8), dimension(1:Nxx1,1:Nxx1) :: ExVy,Ex_invVy
            Real(kind=8), dimension(1:Nxx1,1:Nxx1) :: ExVz,Ex_invVz
            Real(kind=8), dimension(1:Nxx1,1:Nxx1) :: ExxP,Ex_invP

            Real(kind=8), dimension(1:Nzz1,1:Nzz1) :: EzTemp,Ez_invTemp
            Real(kind=8), dimension(1:Nzz1,1:Nzz1) :: EzVx,Ez_invVx
            Real(kind=8), dimension(1:Nzz,1:Nzz)   :: EzVz,Ez_invVz
            Real(kind=8), dimension(1:Nzz1,1:Nzz1) :: EzVy,Ez_invVy
            Real(kind=8), dimension(1:Nzz1,1:Nzz1) :: EzP,Ez_invP
            
            Real(kind=8), dimension(1:Nxx1)        :: LambxTemp, LambxVz, LambxVy, LambxP
            Real(kind=8), dimension(1:Nyy1)        :: LambyTemp, LambyVx, LambyVz, LambyP
            Real(kind=8), dimension(1:Nzz1)        :: LambzTemp, LambzVx, LambzVy, LambzP
            Real(kind=8), dimension(1:Nyy)         :: LambyVy
            Real(kind=8), dimension(1:Nzz)         :: LambzVz
            Real(kind=8), dimension(1:Nxx)         :: LambxVx
                      
            Real(kind=8), dimension(1:Nxx,1:Nxx)   :: ExxFi, Ex_invFi
            Real(kind=8), dimension(1:Nxx)         :: LambxFi
            Real(kind=8), dimension(1:Nyy1,1:Nyy1) :: EyFi, Ey_invFi
            Real(kind=8), dimension(1:Nyy1)        :: LambyFi                 
            Real(kind=8), dimension(1:Nzz,1:Nzz)   :: EzFi, Ez_invFi
            Real(kind=8), dimension(1:Nzz)         :: LambzFi
        End Module EVD_Operators
         
         Module Thomas_coefficients
           Use Size

           Real(kind=8), Dimension(1:Nmax) :: T_left, T_center, T_right
           Real(kind=8), Dimension(1:Nmax) :: P_left, P_center, P_right
           Real(kind=8), Dimension(1:Nmax) :: Vx_left, Vx_center, Vx_right
           Real(kind=8), Dimension(1:Nmax) :: Vy_left, Vy_center, Vy_right
           Real(kind=8), Dimension(1:Nmax) :: Vz_left, Vz_center, Vz_right
           Real(kind=8), Dimension(1:Nmax) :: Fi_left, Fi_center, Fi_right
        
        End Module Thomas_coefficients
