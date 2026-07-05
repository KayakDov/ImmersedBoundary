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
    use AlexCudaCompatibility

    Implicit Real(kind=8) (A-H,O-Z)
    Character*50  Head

    Real(kind=8) :: Nusselt

    Integer, Dimension(2) :: Loc

    ! =============================================================

    Nwrite = 1

    ! #######  Forming the mesh ###########################

    Call     Mesh
    Call   PrMesh

    k_z_write = Nz2 / 2
    j_y_write = Ny2 / 2

    Loc(1:1) = MinLoc( abs( X12(1:Nx1) - 0.05D0 ) );  i_x_write = Loc(1)

    ! .........  Prepare divergence operator .............................

    !        Call   DivVel

    ! .........  Background Temperature Initialization ....................

    Do i=0,Nx2
        Teta(i,0:Ny2,0:Nz2) = 1.D0 - X12(i) / AspRa
    End Do

    ! ############ Introducing initial values #######################

    Call   Init
    Write (*,*) ' Initial: Nu=', Nusselt(),'   Ekin=', Ekinem()

    TmpNew = Tmpr
    VMxNew = VMx
    VMyNew = VMy
    VMzNew = VMz

    ! ######## Time integrating #############################

    Istp   = 0
    TimCur = Tstart
    Time_Interval = Time
    Time   = Time + Tstart

    !           tim1 = timef()
    call cpu_time(tim1)

    10        Istp   = Istp + 1
    TimCur = TimCur + Htime

    Call   TimeStep (Istp, RNSx, RNSy,  RNSz, RTmpr, RDP)

    !          If( Nwrite*(Istp/Nwrite) == Istp ) then

    Ek = Ekinem(); Anu = Nusselt()
    Write (25,*) TimCur, Ek, Anu
    Write (26,*) TimCur, VMz(i_x_write,j_y_write,k_z_write), Tmpr(i_x_write,j_y_write,k_z_write)

    If(I_Fourier == 0) Call Fourier_sum
    !         End If

    ! ............. Printing the current results ...................

    Itst = Iprint * ( Istp/Iprint )

    If (Itst .EQ. Istp) then
        Write (2,210) Istp, Htime, TimCur, Time
        Write (2,230) RNSx, RNSy, RNSz, RTmpr, RDP
        Call    Outp

        Write (*,210) Istp, Htime, TimCur, Time
        Write (*,230) RNSx, RNSy, RNSz, RTmpr, RDP
        !   stop
    End If

    ! ............ Check the convergence ..........................

    Tst = max(RNSx, RNSy, RNSz, RTmpr)  ! max(RDP, RNSx, RNSy, RNSz, RTmpr)

    If (Tst < EpsCnv) Go to 880

    If (TimCur < Time) Go to 10

    880    Continue

    !      tim2 = timef()
    call cpu_time(tim2)
    Write (*,*) ' Time integration lasted ', tim2-tim1,'   seconds'

    Call    Outp

    ! ******* Write the fields ***********************
    !        Tmpr = Tmpr + Teta
    !        Call Point_Write ( Nx2, Ny2, Nz2, Tmpr,   X12, Y12, Z12, 50, 'Tmpr      ')
    !        Call Point_Write ( Nx1, Ny2, Nz2, VMxNew, X  , Y12, Z12, 70, 'Vx        ')
    !        Call Point_Write ( Nx2, Ny1, Nz2, VMyNew, X12, Y  , Z12, 80, 'Vy        ')
    !        Call Point_Write ( Nx2, Ny2, Nz1, VMzNew, X12, Y12, Z  , 90, 'Vz        ')
    !        Call Point_Write ( Nx, Ny, Nz, Prs,    X12(1:Nx1), Y12(1:Ny1), Z12(1:Nz1), 60, 'Prs       ')
    !        Call Point_Write ( Nx1, Ny2, Nz1, Potential,   X, Y12, Z, 100, 'Potential ')

    If(I_Fourier == 0) Call Plot_Amplitude

    Return
    100        Format(G15.8)
    210        Format (//'  Istp=',I5, '   Time Step=',G11.4, ' Time=',G15.8,'   End Time=',G15.8)
    211        Format (//,'Preparing FD approximations:')
    220        Format (//,'  SolLid: Convergence is reached')
    230        Format ('  RNSx=',  G11.4,  '  RNSy=', G11.4,  '  RNSz=', G11.4, '  RTmpr=', G11.4, '  RDP=', G11.4 )

Contains

    Subroutine Fourier_sum

        Tmp_Amplitude(0,:,:,:) = Tmp_Amplitude(0,:,:,:) + TmpNew(:,:,:) * Htime

        VMx_Av = VMx_Av + VMxNew * Htime
        VMy_Av = VMy_Av + VMyNew * Htime
        VMz_Av = VMz_Av + VMzNew * Htime
        Tmp_Av = Tmp_Av + TmpNew * Htime
        Prs_Av = Prs_Av + Prs    * Htime

        Do i=1,N_Fourier
            Tmp_Amplitude( i,:,:,:) = Tmp_Amplitude( i,:,:,:) + TmpNew(:,:,:) * Htime * cos( Omega(i) * TimCur )
            Tmp_Amplitude(-i,:,:,:) = Tmp_Amplitude(-i,:,:,:) + TmpNew(:,:,:) * Htime * sin( Omega(i) * TimCur )
        End Do
    End Subroutine Fourier_sum

    Subroutine Plot_Amplitude
        Character(len=10), Dimension(-3:3) :: Title

        Write (*,*) ' Plot_Amplitude'

        Open(123, file='Amplitudes.dat', status='unknown', form='formatted')
        Open(124, file='Average_velocity.ddd', status='unknown', form='unformatted')

        Tmp_Amplitude(0,0:Nx2,0:Ny2,0:Nz2) = Tmp_Amplitude(0,0:Nx2,0:Ny2,0:Nz2) / Time_Interval + Teta(0:Nx2,0:Ny2,0:Nz2)

        VMx_Av = VMx_Av / Time_Interval
        VMy_Av = VMy_Av / Time_Interval
        VMz_Av = VMz_Av / Time_Interval
        Tmp_Av = Tmp_Av / Time_Interval
        Prs_Av = Prs_Av / Time_Interval

        Do i=1,N_Fourier
            Tmp_Amplitude( i,:,:,:) = 2.d0 * Tmp_Amplitude( i,:,:,:) / Time_Interval
            Tmp_Amplitude(-i,:,:,:) = 2.d0 * Tmp_Amplitude(-i,:,:,:) / Time_Interval
        End Do

        Title(0) = 'Average   '
        Title(1) = 'Omega1_cos';  Title(-1) = 'Omega1_sin'
        Title(2) = 'Omega2_cos';  Title(-2) = 'Omega2_sin'
        Title(3) = 'Omega3_cos';  Title(-3) = 'Omega3_sin'

        Mx=(Nx2+1)/2
        Write (123, *) 'VARIABLES = "X","Y","Z"', (',"',Title(i),'"', i=-N_fourier, N_fourier)
        !              Write (123,301) Mx, Mx, Mx
        Write (123,301) Nx2+1, Ny2+1, Nz2+1

        ! ********* Write the result *******************

        Do k=0,Nz2
            Do j=0,Ny2
                Do i=0,Nx2
                    Write (123,310) X12(i), Y12(j), Z12(k), (Tmp_Amplitude(L,i,j,k), L=-N_fourier, N_fourier)
                End Do
            End Do
        End Do
        Rewind 124
        Write (124) ((( VMx_Av(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
        Write (124) ((( VMy_Av(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
        Write (124) ((( VMz_Av(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
        Write (124) ((( Tmp_Av(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)
        Write (124) ((( Prs_Av(i,j,k), i=1,Nx1), j=1,Ny1), k=1,Nz1)

        VMx = VMx_Av;  VMy = VMy_Av;  VMz = VMz_Av;  Tmpr = Tmp_Av; Prs = Prs_Av

        Tmpr = Tmpr + Teta
        Call Point_Write ( Nx2, Ny2, Nz2, Tmpr,   X12, Y12, Z12, 51, 'Tmpr      ')
        Call Point_Write ( Nx1, Ny2, Nz2, VMxNew, X  , Y12, Z12, 71, 'Vx        ')
        Call Point_Write ( Nx2, Ny1, Nz2, VMyNew, X12, Y  , Z12, 81, 'Vy        ')
        Call Point_Write ( Nx2, Ny2, Nz1, VMzNew, X12, Y12, Z  , 91, 'Vz        ')
        !      Call Point_Write ( Nx, Ny, Nz, Prs,    X12(1:Nx1), Y12(1:Ny1), Z12(1:Nz1), 61, 'Prs       ')

        Write (*,*) ' Plot_Amplitude'
        Return
        301        Format ('ZONE F=POINT, I=',I4, ', J=',I4, ', K=',I4)
        310        Format ( 12(E12.5,1x) )

    End Subroutine Plot_Amplitude


End Subroutine Solution_time