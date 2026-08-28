         Module Size
           Parameter (Nbig=100, Nxx=Nbig, Nyy=Nbig, Nzz=Nbig)
           Parameter (Nxx1=Nxx+1, Nxx2=Nxx+2)
           Parameter (Nyy1=Nyy+1, Nyy2=Nyy+2)
           Parameter (Nzz1=Nzz+1, Nzz2=Nzz+2)
         End Module Size

         Module Numbers
             Integer :: Nx, Nx1, Nx2, Ny, Ny1, Ny2, Nz, Nz1, Nz2
         End Module Numbers

         Module Parameters
             Real(kind=8) ::  AspRa, WidRa, Reynolds, angle
         End Module Parameters

         Module Numerica
             Real(kind=8) :: Eps, EpsCnv, Htime, Hmax, Time, TimCur, Tstart, Ckor
             Integer :: ItMax, Niter, Iprint, IprintIBM, Icheck, I_matrix_C, inner
             Integer :: Iexcl, Istat, EVD_BCx, EVD_BCy, EVD_BCz,SolverKind
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
          Real(kind=8), Target, Dimension(0:Nxx1,0:Nyy2,0:Nzz2) :: VMxOld, VMx,  VMxNew
          Real(kind=8), Target, Dimension(0:Nxx2,0:Nyy1,0:Nzz2) :: VMyOld, VMy,  VMyNew
          Real(kind=8), Target, Dimension(0:Nxx2,0:Nyy2,0:Nzz1) :: VMzOld, VMz,  VMzNew
          Real(kind=8), Target, Dimension(1:Nxx1,1:Nyy1,1:Nzz1) :: Prs,    Dprs
          
         
          Integer,  Dimension(1:Nxx1,1:Nyy1,1:Nzz1):: NumGlP
          
          Real(kind=8), POINTER:: Vel_X_Field(:,:,:),Vel_Y_Field(:,:,:),Vel_Z_Field(:,:,:)
          Real(kind=8), POINTER:: Thomas_f_New(:,:,:),Thomas_f_rhs(:,:,:), RHP(:,:,:), PresP(:,:,:)
          
          Real(kind=8), allocatable :: RHS_Precond(:)

    End Module Variables
    
     Module Matrices
          
          Real(kind=8), POINTER:: B(:),BT(:),BT_expanded(:,:,:),lambdaTemp(:)
          Integer,      POINTER:: B_R_C(:,:), BT_R_C(:,:)
                                  
         Real(kind=8), CONTIGUOUS, POINTER:: B_CSR_Prs(:), BT_CSR_Prs(:),RHS_F_tag(:)
         Real(kind=8), POINTER:: IGPrs(:)
         Integer, CONTIGUOUS, POINTER:: B_Row_CSR_Prs(:),B_Col_CSR_Prs(:), &
                                        BT_Row_CSR_Prs(:),BT_Col_CSR_Prs(:)
         
         Integer*8 Size_R_Ftag_Matrix
         
        End Module Matrices
    

! ........... Finite difference operators ..........................

         Module Operators
           Use Size

! ......... Arrays for definition of linear terms ...........

            Real(kind=8)  DivV(6,Nxx1,Nyy1,Nzz1)

! ......... Arrays for definition of FD equations ............

            Real(kind=8), Target,Dimension(0:Nxx2,0:Nyy2,0:Nzz2) :: FDRHP
            Real(kind=8), Target,Dimension(0:Nxx1,0:Nyy2,0:Nzz2) :: RHSx
            Real(kind=8), Target,Dimension(0:Nxx2,0:Nyy1,0:Nzz2) :: RHSy
            Real(kind=8), Target,Dimension(0:Nxx2,0:Nyy2,0:Nzz1) :: RHSz  

         End Module Operators

! ........... Eigenvalues operators ..........................

         Module EVD_Operators
           Use Size
           
            Parameter (Nemax=max(Nxx2,Nyy2))
             
            Real(kind=8), dimension(1:Nxx1,1:Nxx1) :: D2_dx2
            Real(kind=8), dimension(1:Nyy1,1:Nyy1) :: D2_dy2
            Real(kind=8), dimension(1:Nzz2,1:Nzz2) :: D2_dz2
     
            Real(kind=8), dimension(1:Nyy1,1:Nyy1) :: EyVx,Ey_invVx
            Real(kind=8), dimension(1:Nyy,1:Nyy)   :: EyVy,Ey_invVy
            Real(kind=8), dimension(1:Nyy1,1:Nyy1) :: EyVz,Ey_invVz
            Real(kind=8), dimension(1:Nyy1,1:Nyy1) :: EyP,Ey_invP

            Real(kind=8), dimension(1:Nxx1,1:Nxx1) :: ExVz,Ex_invVz
            Real(kind=8), dimension(1:Nxx,1:Nxx)   :: ExVx,Ex_invVx
            Real(kind=8), dimension(1:Nxx1,1:Nxx1) :: ExVy,Ex_invVy
            Real(kind=8), dimension(1:Nxx1,1:Nxx1) :: ExxP,Ex_invP
            
            Real(kind=8), dimension(1:Nyy1)        :: LambyVx, LambyVz, LambyP
            Real(kind=8), dimension(1:Nxx1)        :: LambxVz, LambxVy, LambxP
            Real(kind=8), dimension(1:Nyy)         :: LambyVy
            Real(kind=8), dimension(1:Nxx)         :: LambxVx
                      
         End Module EVD_Operators
         
         Module Thomas_coefficients
           Use Size

           Real(kind=8), Dimension(1:Nzz2) :: P_left,  P_center,  P_right
           Real(kind=8), Dimension(1:Nzz2) :: Vx_left, Vx_center, Vx_right
           Real(kind=8), Dimension(1:Nzz2) :: Vy_left, Vy_center, Vy_right
           Real(kind=8), Dimension(1:Nzz2) :: Vz_left, Vz_center, Vz_right
         
    End Module Thomas_coefficients
    
        Module Ibmethod 
            Use Size
            Real(kind=8), DIMENSION(:),POINTER::F_tag
            
            TYPE body_point_info
                INTEGER  RegNumberVx,RegNumberVy,RegNumberVz
              
                INTEGER, DIMENSION(:), POINTER::Vx_X_ind,Vx_Y_ind,Vx_Z_ind
                REAL(KIND(0.D0)), DIMENSION(:), POINTER ::Vx_Weight
                
                INTEGER, DIMENSION(:), POINTER::Vy_X_ind,Vy_Y_ind,Vy_Z_ind
                REAL(KIND(0.D0)), DIMENSION(:), POINTER ::Vy_Weight
                
                INTEGER, DIMENSION(:), POINTER::Vz_X_ind,Vz_Y_ind,Vz_Z_ind
                REAL(KIND(0.D0)), DIMENSION(:), POINTER ::Vz_Weight
            
            END TYPE body_point_info
            
          
      
            TYPE body
                LOGICAL   :: moving, Tparticip
                INTEGER   :: npts
                INTEGER*8 :: Number_Of_Matrix_B_Entries_X,Number_Of_Matrix_B_Entries_Y,Number_Of_Matrix_B_Entries_Z
                REAL(KIND(0.D0)), DIMENSION(:), POINTER ::     X,Y,Z,&
                                                               Vx_interp_New, Vy_interp_New, Vz_interp_New,  &
                                                               Vx_b,          Vy_b,          Vz_b,           &
                                                               fb_x,          fb_y,          fb_z,           &
                                                               fb_x_tag,      fb_y_tag,      fb_z_tag
                REAL(KIND(0.D0)):: X_C,Y_C, Z_C, Scl,Dv
                 
                REAL(KIND(0.D0)),  ALLOCATABLE :: Div_F_X_Val(:), Div_F_Y_Val(:), Div_F_Z_Val(:)
                INTEGER,           ALLOCATABLE :: Div_F_X_ROW(:), Div_F_X_COL(:), Div_F_Y_ROW(:), Div_F_Y_COL(:),Div_F_Z_ROW(:), Div_F_Z_COL(:)
                
                REAL(KIND=8), POINTER::R_Ftag_Matrix_Fx(:), R_Ftag_Matrix_Fy(:), R_Ftag_Matrix_Fz(:)
                INTEGER,      POINTER::R_Ftag_Matrix_Fx_Row(:), R_Ftag_Matrix_Fx_Col(:),&
                                       R_Ftag_Matrix_Fy_Row(:), R_Ftag_Matrix_Fy_Col(:),&
                                       R_Ftag_Matrix_Fz_Row(:), R_Ftag_Matrix_Fz_Col(:) 
                
                 TYPE(body_point_info), DIMENSION(:), POINTER ::point_info 
            END TYPE body
            
            TYPE interp_point
                REAL(KIND(0.D0))::   X,Y,Z
                TYPE(body_point_info):: interp_point_info    
            END TYPE interp_point
            
            
            INTEGER, PARAMETER :: maxbodies = 999 ! a large number
            INTEGER, PARAMETER :: maxpoints = 999 ! a large number
            TYPE(body), DIMENSION(maxbodies) :: bdy
            TYPE(interp_point), DIMENSION(maxpoints) :: interp_points
                
            REAL(KIND(0.0D0)) :: support = 3.0D0 ! support for smearing delta functions
            
            REAL(KIND(0.D0)):: d2X, supx,d2Y, supy,d2Z, supz, Delta
            
            
            INTEGER :: n_body          ! number of  bodies 
           
            LOGICAL :: TemprForcing    !At least one body exists with given temperature 
         
    End Module Ibmethod
    
  
