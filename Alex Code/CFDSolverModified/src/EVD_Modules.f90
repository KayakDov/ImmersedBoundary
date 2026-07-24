! ============================================================
! EVD_Modules.f90 -- Alex's CPU-solver module data, restored
! VERBATIM from the original modfv_3D_Q2D.f90 so the original
! EVD_lap* / EVD_Thomas / EVDmethod files compile unchanged
! alongside the GPU code, for CPU-vs-GPU validation ONLY.
! Remove this file (and the EVD_* files) after validating --
! it is not part of production code.
! ============================================================

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
