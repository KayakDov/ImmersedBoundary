! ************************************************************
! *   Subroutine for straight-forward solution  of           *     
! *       FD problem for convection in rectangulars          *
! *                                                          *
! *   This is version with 3-level time integrator           *
! *                                                          *
! ************************************************************

Subroutine  TimeStep ( Istp, RNSx, RNSy, RNSz, RTmpr, RDP )

    Use Numbers
    Use Parameters
    Use Numerica
    Use Grid
    Use Operators
    Use Variables
    Use EVD_Operators
    Use Thomas_coefficients
    Use AlexCudaCompatibility
    Use AlexCudaCompatibility, only : TemperatureHandle, VxHandle, VyHandle, VzHandle, PressureHandle
    Use eigenbcgsolver_eigen_mod, only : solve_eigen_decomp_d

    Implicit Real(kind=8) (A-H,O-Z)

    Ht = 2.D0 * Htime

    dt_temp = Dble(Istat)

    ! ########### Time step for temperature ##################

    !          Write (*,*) ' time step entered', dt_temp

    ! +++++++++++ Right hand side ++++++++++++++++++++++++++++++++

    FDRHP = 0.D0
    Call  VgrTmp
    !           Write (*,*) '1 FDRHP =', Sum(abs(FDRHP))

    ! +++++++++++ Time derivative +++++++++++++++++++++++++++++++

    FDRHP(1:Nx1,1:Ny1,1:Nz1) = FDRHP(1:Nx1,1:Ny1,1:Nz1) -          &
            &              dt_temp * ( 4.D0 * Tmpr(1:Nx1,1:Ny1,1:Nz1) -          &
                    &                               TmpOld(1:Nx1,1:Ny1,1:Nz1)     )  / Ht

    !          Write (*,*) '2 FDRHP =', Maxval(abs(FDRHP))

    ! ......... Solution ........................................................

    !           Call   EVD_Thomas (TmpNew(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1), &
    !     &                   EyTemp(1:Ny1,1:Ny1),Ey_invTemp(1:Ny1,1:Ny1),         &
    !     &                   EzTemp(1:Nz1,1:Nz1),Ez_invTemp(1:Nz1,1:Nz1),         &
    !     &                   LambyTemp(1:Ny1), LambzTemp(1:Nz1),                  &
    !     &                   T_left(1:Nx1), T_center(1:Nx1), T_right(1:Nx1),      &
    !     &                   Nx1, Ny1, Nz1, dt_temp)

    Call solve_eigen_decomp_d( &
            TemperatureHandle, &
            TmpNew(1:Nx1,1:Ny1,1:Nz1), &
            FDRHP(1:Nx1,1:Ny1,1:Nz1))

    !          Write (*,*) ' TmpNew=', Maxval(abs(TMpNew))

    Call EVDbounds

    RTmpr = Dist2D (TmpNew, Tmpr, Nx2, Ny2, Nz2, Nx2, Ny2, Nz2)
    !      Write (*,*) ' RTmpr=', RTmpr, Minval(TmpNew), Maxval(tmpnew)
    !        Write (*,*) 'DT=', Maxval(abs(TmpNew - Tmpr) ), Maxloc(abs(TmpNew - Tmpr) ), Maxval(abs(TmpNew(:,:,102)))
    !stop

    ! ########### Inversing the Stokes operator #################

    Call Get_Potential

    ! ======== Make r.h.s. of momentum equations ==================

    FDRHP = 0.D0
    Call   VgrdVx
    Call   GradPx( RHSx(1:Nx ,1:Ny1,1:Nz1), Prs )
    RHSx(1:Nx,1:Ny1,1:Nz1) = RHSx(1:Nx,1:Ny1,1:Nz1) + FDRHP(1:Nx,1:Ny1,1:Nz1)

    FDRHP = 0.D0
    Call   VgrdVy
    Call   GradPy( RHSy(1:Nx1,1:Ny ,1:Nz1), Prs )
    RHSy(1:Nx1,1:Ny,1:Nz1) = RHSy(1:Nx1,1:Ny,1:Nz1) + FDRHP(1:Nx1,1:Ny,1:Nz1)

    FDRHP = 0.D0
    Call   VgrdVz
    Call   GradPz( RHSz(1:Nx1,1:Ny1,1:Nz ), Prs )
    RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz) + FDRHP(1:Nx1,1:Ny1,1:Nz)

    ! .............. Add bouyancy force ......................

    RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz)  &
            &     - 0.5d0 * Bu_Gr * ( TmpNew(1:Nx1,1:Ny1,1:Nz) + TmpNew(1:Nx1,1:Ny1,2:Nz1) ) &
            &     - 0.5d0 * Bu_Gr *  (   Teta(1:Nx1,1:Ny1,1:Nz) +   Teta(1:Nx1,1:Ny1,2:Nz1) )

    ! +++++++++++ Straight-forward step +++++++++++++++++++++

    RHSx(1:Nx,1:Ny1,1:Nz1) = RHSx(1:Nx,1:Ny1,1:Nz1) - &
            &                  ( 4.D0 * VMx(1:Nx,1:Ny1,1:Nz1) - VMxOld(1:Nx,1:Ny1,1:Nz1) )/ Ht

    RHSy(1:Nx1,1:Ny,1:Nz1) = RHSy(1:Nx1,1:Ny,1:Nz1) - &
            &                  ( 4.D0 * VMy(1:Nx1,1:Ny,1:Nz1) - VMyOld(1:Nx1,1:Ny,1:Nz1) )/ Ht

    RHSz(1:Nx1,1:Ny1,1:Nz) = RHSz(1:Nx1,1:Ny1,1:Nz) - &
            &                  ( 4.D0 * VMz(1:Nx1,1:Ny1,1:Nz) - VMzOld(1:Nx1,1:Ny1,1:Nz) )/ Ht

    ! +++++++++ Electromagnetic force ++++++++++++++

    Call EM_force

    ! ++++++++++++ [Calculate Lap(u)^-1]*RHSx +++++++++++++++++++++++

    !      Call   EVD_Thomas (VMxNew(1:Nx,1:Ny1,1:Nz1), RHSx(1:Nx,1:Ny1,1:Nz1), &
    !     &                   EyVx(1:Ny1,1:Ny1),Ey_invVx(1:Ny1,1:Ny1),          &
    !     &                   EzVx(1:Nz1,1:Nz1),Ez_invVx(1:Nz1,1:Nz1),          &
    !     &                   LambyVx(1:Ny1), LambzVx(1:Nz1),                   &
    !     &                   Vx_left(1:Nx), Vx_center(1:Nx), Vx_right(1:Nx),   &
    !     &                   Nx, Ny1, Nz1, 1.D0)

    Call solve_eigen_decomp_d( &
            VxHandle, &
            VMxNew(1:Nx,1:Ny1,1:Nz1), &
            RHSx(1:Nx,1:Ny1,1:Nz1))

    ! ++++++++++++++ [Calculate Lap(v)^-1]*RHSy ++++++++++++++++++++++++++

    !      Call   EVD_Thomas (VMyNew(1:Nx1,1:Ny,1:Nz1), RHSy(1:Nx1,1:Ny,1:Nz1),   &
    !     &                   EyVy(1:Ny,1:Ny),  Ey_invVy(1:Ny,1:Ny),              &
    !     &                   EzVy(1:Nz1,1:Nz1),Ez_invVy(1:Nz1,1:Nz1),            &
    !     &                   LambyVy(1:Ny), LambzVy(1:Nz1),                      &
    !     &                   Vy_left(1:Nx1), Vy_center(1:Nx1), Vy_right(1:Nx1),  &
    !     &                   Nx1, Ny, Nz1, 1.D0)

    Call solve_eigen_decomp_d( &
            VyHandle, &
            VMyNew(1:Nx1,1:Ny,1:Nz1), &
            RHSy(1:Nx1,1:Ny,1:Nz1))

    ! ++++++++++++++ [Calculate Lap(w)^-1]*RHSz ++++++++++++++++++++++++++

    !      Call   EVD_Thomas (VMzNew(1:Nx1,1:Ny1,1:Nz), RHSz(1:Nx1,1:Ny1,1:Nz),   &
    !     &                   EyVz(1:Ny1,1:Ny1),Ey_invVz(1:Ny1,1:Ny1),            &
    !     &                   EzVz(1:Nz,1:Nz),  Ez_invVz(1:Nz,1:Nz),              &
    !     &                   LambyVz(1:Ny1), LambzVz(1:Nz),                      &
    !     &                   Vz_left(1:Nx1), Vz_center(1:Nx1), Vz_right(1:Nx1),  &
    !     &                   Nx1, Ny1, Nz, 1.D0)

    Call solve_eigen_decomp_d( &
            VzHandle, &
            VMzNew(1:Nx1,1:Ny1,1:Nz), &
            RHSz(1:Nx1,1:Ny1,1:Nz))

    Call EVDbounds

    !        RNSx = Dist2D (VMx, VMxNew, Nx1, Ny2, Nz2, Nx1, Ny2, Nz2)
    !        RNSy = Dist2D (VMy, VMyNew, Nx2, Ny1, Nz2, Nx2, Ny1, Nz2)
    !        RNSz = Dist2D (VMz, VMzNew, Nx2, Ny2, Nz1, Nx2, Ny2, Nz1)
    !
    !        Write (*,*) ' RNS=', RNSx, RNSy, RNSz
    !stop

    ! ++++++++ Calcualte pressure correction ++++++++++++

    FDRHP= 0.d0

    Call FdDiv

    FDRHP = FDRHP * Ckor / Htime

    !    Write (*,*) ' FDRHP=', Sum(abs(FDRHP))

    !      Call   EVD_Thomas (Dprs(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1),  &
    !     &                   EyP(1:Ny1,1:Ny1), Ey_invP(1:Ny1,1:Ny1),             &
    !     &                   EzP(1:Nz1,1:Nz1), Ez_invP(1:Nz1,1:Nz1),             &
    !     &                   LambyP(1:Ny1), LambzP(1:Nz1),                       &
    !     &                   P_left(1:Nx1), P_center(1:Nx1), P_right(1:Nx1),     &
    !     &                   Nx1, Ny1, Nz1, 0.D0)

    Call solve_eigen_decomp_d( &
            PressureHandle, &
            Dprs(1:Nx1,1:Ny1,1:Nz1), &
            FDRHP(1:Nx1,1:Ny1,1:Nz1))
    !   Write (*,*) ' Dprs=', Sum(abs(Dprs))

    ! ++++++++++++ Calculate velocities ++++++++++++++++++++++++++++++

    ! ........... calculate grad(Prs) .....................

    Call GradPx( RHSx(1:Nx ,1:Ny1,1:Nz1), Dprs )
    Call GradPy( RHSy(1:Nx1,1:Ny ,1:Nz1), Dprs )
    Call GradPz( RHSz(1:Nx1,1:Ny1,1:Nz ), Dprs )

    ! ........... Correct velocities .........................................

    VMxNew(1:Nx,1:Ny1,1:Nz1) = VMxNew(1:Nx,1:Ny1,1:Nz1) - RHSx(1:Nx ,1:Ny1,1:Nz1) * Htime / Ckor
    VMyNew(1:Nx1,1:Ny,1:Nz1) = VMyNew(1:Nx1,1:Ny,1:Nz1) - RHSy(1:Nx1,1:Ny ,1:Nz1) * Htime / Ckor
    VMzNew(1:Nx1,1:Ny1,1:Nz) = VMzNew(1:Nx1,1:Ny1,1:Nz) - RHSz(1:Nx1,1:Ny1,1:Nz ) * Htime / Ckor

    Prs = Prs + DPrs

    call EVDbounds

    RNSx = Dist2D (VMx, VMxNew, Nx1, Ny2, Nz2, Nx1, Ny2, Nz2)
    RNSy = Dist2D (VMy, VMyNew, Nx2, Ny1, Nz2, Nx2, Ny1, Nz2)
    RNSz = Dist2D (VMz, VMzNew, Nx2, Ny2, Nz1, Nx2, Ny2, Nz1)

    RDP = MaxVal(Abs(Dprs) )

    !     FDRHP= 0.d0
    !     Call FdDiv
    !    Write (*,*) ' DivV=', Sum(abs(FDRHP))
    !        Write (*,*) RNSx, RNSy, RNSz, RDP
    !        stop

    ! ######## Check of results #############################

    FDRHP= 0.d0
    Call FdDiv

    If (Icheck .EQ. 0)  Call    Check

    ! ######## Shift of the time step ########################

    444            Continue

    VMxOld = VMx
    VMyOld = VMy
    VMzOld = VMz
    TmpOld = Tmpr

    VMx  = VMxNew
    VMy  = VMyNew
    VMz  = VMzNew
    Tmpr = TmpNew

    Return
End