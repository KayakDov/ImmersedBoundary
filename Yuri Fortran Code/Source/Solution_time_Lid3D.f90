! ************************************************************
! *  Main subroutine for  analysis of convection             *
! *       in rectangular cavities                            *
! *                                                          *
! *   This is version with 3-level time integrator           *
! *                                                          *
! ************************************************************
        Subroutine    Solution_time

         Use Numbers
         Use Parameters
         Use Numerica
         Use Grid
         Use Variables
         Use Operators
         use EVD_Operators
         Use EvdProcedures
         Use MatrixFormAndOperate
         Use Matrices
         Use IBsetupInetrpRegular
         Use FVOperators

        Implicit Real(kind=8) (A-H,O-Z)
        Character*50  Head

        Real(kind=8) :: Nusselt

        Integer, Dimension(2) :: Loc
        Real*8, Dimension(4):: IntPoints

! =============================================================

        
! #######  Forming the mesh ###########################

         Call     Mesh
         Call   PrMesh 
 
        
 ! .........  Prepare divergence operator .............................

       Call   DivVel 
 
! .........  EienValDecomp .............................
       Call  EVDLapVx
       Call  EVDLapVy
       Call  EVDLapVz
       Call  EVDLapP
       Call  OrdVarPres
       
              
! ############ Introducing initial values #######################

        Call   Init

            VMxNew = VMx
            VMyNew = VMy
            VMzNew = VMz

! ######## Time integrating #############################

        Istp   = 0
        TimCur = Tstart
        Time   = Time + Tstart

        tim1 = timef()
        
        Call Setup_geometry  
        Call Allocate_Forces_And_Forces_RHS
        
       
        
        Call Build_B_And_BTranspose
         
       

10        Istp   = Istp + 1
          TimCur = TimCur + Htime
        
     
        Call   TimeStep (Istp, RNSx, RNSy,  RNSz, RDP)
        
         

! ............. Printing the current results ...................

        Itst = Iprint * ( Istp/Iprint )
        
          Vel_X_Field=>VMxNew
          Vel_Y_Field=>VMyNew
          Vel_Z_Field=>VMzNew
        
         !Do kk=1,4 
         !       Call interpolation_Velocity_postproc (Vel_X_Field,Vel_Y_Field,Vel_Z_Field, V_I_X,V_I_Y,V_I_Z,kk)
         !       IntPoints(kk)=V_I_Z
         !End Do 
         
        ! write(88,110) IntPoints(1:4)

        If (Itst .EQ. Istp) then
                            Write (2,210) Istp, Htime, TimCur, Time
                            Write (2,230) RNSx, RNSy, RNSz, RDP
                            Call    Outp (1) 

                            Write (*,210) Istp, Htime, TimCur, Time
                            Write (*,230) RNSx, RNSy, RNSz, RDP
                              
                            Vel_X_Field=>VMxNew
                            Vel_Y_Field=>VMyNew
                            Vel_Z_Field=>VMzNew
                            
                            Call interpolation(Vel_X_Field,Vel_Y_Field,Vel_Z_Field,bdy(1)%Vx_interp_New,bdy(1)%Vy_interp_New,bdy(1)%Vz_interp_New,1)
                            write(*,*)  maxval ((bdy(1)%VX_interp_New(:))), maxval ((bdy(1)%VY_interp_New(:))),maxval ((bdy(1)%Vz_interp_New(:)))
                            write(*,*)  minval ((bdy(1)%VX_interp_New(:))), minval ((bdy(1)%VY_interp_New(:))),minval ((bdy(1)%Vz_interp_New(:)))
                            
        End If     
  
        
! ............ Check the convergence ..........................

        Tst = max(RDP, RNSx, RNSy, RNSz)

        If (Tst < EpsCnv) Go to 880

        If (TimCur < Time) Go to 10
        
 880    Continue
        tim2 = timef()

       Write (*,*) ' Time integration lasted ', tim2-tim1,'   seconds'

          Call    Outp(4) 

! ******* Write the fields ***********************
 
        Call Point_Write ( Nx1, Ny2, Nz2, VMxNew, X  , Y12, Z12, 70, 'Vx        ')
        Call Point_Write ( Nx2, Ny1, Nz2, VMyNew, X12, Y  , Z12, 80, 'Vy        ')
        Call Point_Write ( Nx2, Ny2, Nz1, VMzNew, X12, Y12, Z  , 90, 'Vz        ')
        Call Point_Write ( Nx2, Ny2, Nz2, Prs,    X12, Y12, Z12, 60, 'Prs       ')
 
  !      Call Point_Write_Lid 

        Return
100     Format(G15.8)
110     Format(4G15.8)        
210        Format (//'  Istp=',I5, '   Time Step=',G11.4, ' Time=',G15.8,'   End Time=',G15.8)
211        Format (//,'Preparing FD approximations:')
220        Format (//,'  SolLid: Convergence is reached')
230        Format ('  RNSx=',  G11.4,  '  RNSy=', G11.4,  '  RNSz=', G11.4,  '  RDP=', G11.4 )
        End


