         Module Size
           Parameter (Nxx=256, Nyy=256, Nzz=256)
           Parameter (Nxx1=Nxx+1, Nxx2=Nxx+2)
           Parameter (Nyy1=Nyy+1, Nyy2=Nyy+2)
           Parameter (Nzz1=Nzz+1, Nzz2=Nzz+2)
         End Module Size

         Module Numbers
             Integer :: Nx, Nx1, Nx2, Ny, Ny1, Ny2, Nz, Nz1, Nz2
             ! Boundary condition flags: 0 = Dirichlet, 1 = Neumann
             Integer :: EVD_BCx, EVD_BCy, EVD_BCz
         End Module Numbers

         Module Grid
           Use Size

           Real(kind=8) X(0:Nxx1), X12(0:Nxx2), Hx12(0:Nxx), HPx(0:Nxx1)
           Real(kind=8) Y(0:Nyy1), Y12(0:Nyy2), Hy12(0:Nyy), HPy(0:Nyy1)
           Real(kind=8) Z(0:Nzz1), Z12(0:Nzz2), Hz12(0:Nzz), HPz(0:Nzz1)
           ! Domain dimensions (aspect ratios)
           Real(kind=8) :: AspRa, WidRa
         End Module Grid

         Module Variables
           Use Size
             Real(kind=8), Dimension(0:Nxx2,0:Nyy2,0:Nzz2) :: TmpOld, Tmpr, TmpNew, Teta
             Real(kind=8), Dimension(0:Nxx2,0:Nyy2,0:Nzz2) :: FDRHP
       End Module Variables

         Module EVD_Operators
           Use Size
           
            Parameter (Nemax=max(Nxx2,Nyy2,Nzz2))
             
            Real(kind=8), dimension(1:Nemax,1:Nemax) :: D2_d2
     
            Real(kind=8), dimension(1:Nxx1,1:Nxx1) :: ExTemp,Ex_invTemp
            Real(kind=8), dimension(1:Nyy1,1:Nyy1) :: EyTemp,Ey_invTemp
            Real(kind=8), dimension(1:Nzz1,1:Nzz1) :: EzTemp,Ez_invTemp
            
            Real(kind=8), dimension(1:Nyy1)        :: LambyTemp
            Real(kind=8), dimension(1:Nzz1)        :: LambzTemp
            Real(kind=8), dimension(1:Nxx1)        :: LambxTemp
         End Module EVD_Operators
         
         Module Thomas_coefficients
           Use Size
           Real(kind=8), Dimension(1:Nxx2) :: T_left, T_center, T_right
        End Module Thomas_coefficients
