! ************************************************************
! *  Main program for unsteady convection                    *
! *            in rectangular cavities                       *
! *                                                          *
! ************************************************************

Use Numbers
Use Parameters
Use Numerica
Use Grid
Use Variables
Use Operators
use AlexCudaCompatibility
Use eigenbcgsolver_eigen_mod, only : finalize_eigen_decomp_d

Implicit Real(kind=8) (A-H,O-Z)

Character*50 sentns
Character*80 balabo

Integer ::  omp_get_max_threads
Real(kind=8) :: Nusselt, Nusselt_middle

! ======================================================================

Open (1, file='Conv.dat', form='formatted',   status='old')
Open (2, file='Conv.out', form='formatted',   status='unknown')
Open (3, file='Conv.ddd', form='unformatted', status='unknown', Err=881)
Open (4, file='SteadyState.ddd', form='unformatted', status='unknown', Err=881)

Open (50, file='Tmpr.dat', form='formatted',  status='unknown')
Open (60, file='Pres.dat', form='formatted',  status='unknown')
Open (70, file='Vx.dat',   form='formatted',  status='unknown')
Open (80, file='Vy.dat',   form='formatted',  status='unknown')
Open (90, file='Vz.dat',   form='formatted',  status='unknown')
Open (100, file='Potential.dat',   form='formatted',  status='unknown')

Open (51, file='Tmpr_av.dat', form='formatted',  status='unknown')
Open (61, file='Pres_av.dat', form='formatted',  status='unknown')
Open (71, file='Vx_av.dat',   form='formatted',  status='unknown')
Open (81, file='Vy_av.dat',   form='formatted',  status='unknown')
Open (91, file='Vz_av.dat',   form='formatted',  status='unknown')

Open (25, file='Serie.dat', form='formatted', status='unknown' , position='append')
Open (26, file='History.dat', form='formatted', status='unknown' , position='append')
Open (77, file='Cpu.dat', form='formatted', status='unknown' , position='append')

! #######  Input of parameters #########################

Am = 1.D0

Read (1,101) AspRa, WidRa, Gr, Bi, Prandtl, Hartmann
Read (1,100) Nx, Ny, Nz
Read (1,101) Eps, EpsCnv
Read (1,100) ItMax
Read (1,101) Htime, Time
Read (1,100) Iprint
Read (1,100) Icheck, Iexcl, Istat
Read (1,100) EVD_BCx, EVD_BCy, EVD_BCz
Read (1,100) EVD_Pot_X, EVD_Pot_Y, EVD_Pot_Z
Read (1,100) Ncpus

Read (1,100) I_Fourier, N_Fourier

Write (*,200)  Nx, Ny, Nz
Write (*,201)  AspRa, WidRa, Gr, Bi, Prandtl, Hartmann
Write (*,205)  Htime, Time
Write (*,203)  ItMax, Iprint , Eps

Write (2,200)  Nx, Ny, Nz
Write (2,201)  AspRa, WidRa, Gr, Bi, Prandtl, Hartmann
Write (2,205)  Htime, Time
Write (2,203)  ItMax, Iprint , Eps

DGr = 1.D0 / Sqrt(Gr)

!      DGr = DGr / Hartmann

Bu_Gr = Gr * DGr**2

! ******** OUTPUT OF THE COMPUTATIONAL STATUS **************

Write (*,208)

! .............................................

If (Icheck == 0) then
    balabo = ' Check of divergence and conservative properties is on'
Else
    balabo = ' Check of divergence and conservative properties is off'
End If
Write (*,210)  balabo
! ...............................................

If (Istat == 0) then
    Ismpl = 0
    balabo = ' Stationary temperature equation will be used '
    Write (*,210) balabo
End If


If (Iexcl /= 0) then
    If (Iexcl < 0) then
        balabo =' !!! Only temperature is computed !!! '
    Else
        balabo =' !!! Only fluid flow is computed !!!'
    End If
    Write (*,210)  balabo
End If
! ...............................................

If (Iprint < 0) then
    balabo = ' Initial values will be read from file'
Else
    balabo = ' Initial values will be crazily estimated'
End If
Write (*,210) balabo

balabo = ' This is version with 3-level time integrator'

Write (*,210) balabo

! ************ Some necessary integers *****************

Nx1 = Nx + 1
Nx2 = Nx + 2
Ny1 = Ny + 1
Ny2 = Ny + 2
Nz1 = Nz + 1
Nz2 = Nz + 2

! ************* allocate memory for large arrays **********************

Allocate( VMxOld(0:Ny2,0:Nz2,0:Nx1), VMx(0:Ny2,0:Nz2,0:Nx1),  VMxNew(0:Ny2,0:Nz2,0:Nx1), RHSx(0:Ny2,0:Nz2,0:Nx1) )
Allocate( VMyOld(0:Ny1,0:Nz2,0:Nx2), VMy(0:Ny1,0:Nz2,0:Nx2),  VMyNew(0:Ny1,0:Nz2,0:Nx2), RHSy(0:Ny1,0:Nz2,0:Nx2) )
Allocate( VMzOld(0:Ny2,0:Nz1,0:Nx2), VMz(0:Ny2,0:Nz1,0:Nx2),  VMzNew(0:Ny2,0:Nz1,0:Nx2), RHSz(0:Ny2,0:Nz1,0:Nx2) )

Allocate( TmpOld(0:Ny2,0:Nz2,0:Nx2), Tmpr(0:Ny2,0:Nz2,0:Nx2), TmpNew(0:Ny2,0:Nz2,0:Nx2), Teta(0:Ny2,0:Nz2,0:Nx2) )

Allocate( Prs(1:Ny1,1:Nz1,1:Nx1), Dprs(1:Ny1,1:Nz1,1:Nx1), FDRHP(0:Ny2,0:Nz2,0:Nx2), FDRHP1(0:Ny2,0:Nz2,0:Nx2) )

Allocate( Potential(0:Ny2,0:Nz1,0:Nx1) )

If(I_fourier == 0) then
    Allocate( Tmp_Amplitude(-N_fourier:N_fourier,0:Ny2,0:Nz2,0:Nx2), Omega(N_Fourier) )
    Allocate( VMx_Av(0:Ny2,0:Nz2,0:Nx1), VMy_Av(0:Ny1,0:Nz2,0:Nx2), VMz_Av(0:Ny2,0:Nz1,0:Nx2) )
    Allocate( Tmp_Av(0:Ny2,0:Nz2,0:Nx2) )
    Allocate( Prs_Av(1:Ny1,1:Nz1,1:Nx1) )

    Tmp_Amplitude = 0.d0
    VMx_Av = 0.d0;   VMy_Av = 0.d0;   VMz_Av = 0.d0;   Prs_Av = 0.d0

    If(I_Fourier ==0) then
        Do i=1,N_Fourier
            Read(1,101) Omega(i)
        End Do
    End If
End If

! ########  Call of the main subroutine ################


!Call omp_set_num_threads(Ncpus)
! Force standard OpenMP execution to a single thread
Ncpus = 1
Call omp_set_num_threads(Ncpus)
Write (*,*) 'FORCED: 1 CPU will be used'
Write (*,*) omp_get_max_threads(), '  CPUs will be used'

! ---------------------------------------------------------
! 1. GPU SETUP: External Call to your new file
! ---------------------------------------------------------

Call    Solution_time

!          Write (4) ((( VMx(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
!          Write (4) ((( VMy(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
!          Write (4) ((( VMz(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
!          Write (4) (((Tmpr(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)
!          Write (4) ((( Prs(i,j,k), i=1,Nx1), j=1,Ny1), k=1,Nz1)
! ******** Calculate Nusselt ********************

If(I_Fourier == 0) then
    ! NOTE: the original code read "Tmpr = Tmpr_Av; ... VMz = VMx_Az" -- both
    ! Tmpr_Av and VMx_Az are undeclared and, via implicit typing, were silently
    ! treated as uninitialized scalars (a pre-existing bug). Corrected to the
    ! evident intent:
    Tmpr = Tmp_Av;  VMx = VMx_Av;  VMy = VMy_Av;  VMz = VMz_Av
End If

Write (*,*) '          Nusselt number  = ', Nusselt()
Write (*,*) ' Midplane Nusselt number  = ', Nusselt_middle()
Write (*,*) '           kinetic energy = ', Ekinem()

Write (2,*) '          Nusselt number  = ', Nusselt()
Write (2,*) ' Midplane Nusselt number  = ', Nusselt_middle()
Write (2,*) '           kinetic energy = ', Ekinem()

Call Average_flow

! ---------------------------------------------------------
! 2. GPU TEARDOWN: Clear memory before the program exits
! ---------------------------------------------------------
Call finalize_eigen_decomp_d()
Stop

881        Write (*,8001)
Stop

8001       Format (' ConvMain:  Attempt to read initial values from file is not successfull' )
100        Format (I5)
101        Format (G15.8)
200        Format (1X,I3,' x-points  ',  /, 1x, I3,' y-points',/, 1x, I3,' z-points',//)
201        Format (  ' Aspect Ratio                  =',G15.8,/,  &
        &               ' Width  Ratio                  =',G15.8,/,  &
        &               ' Grashof number                =',G15.8,/,  &
        &               ' Biot number                   =',G15.8,/,  &
        &               ' Prandtl number                =',G15.8,/,  &
        &               ' Hartmann number               =',G15.8,//)

205        Format (  ' Initial time step             =', G15.8,/, &
        &               ' Time interval                 =', G15.8)

203        Format (  ' Maximal number of iterations  =', I5,/,    &
        &               ' Print step                    =', I5,/,    &
        &               ' Precision                     =', G15.8,/)
208        Format (//,'  Computational status:')
210        Format (/,A80, //)

End