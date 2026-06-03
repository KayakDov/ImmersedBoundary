! ************************************************************
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
         Use IBsetupInetrpRegular

        Implicit Real(kind=8) (A-H,O-Z)
 
        Character*50 sentns
        Character*80 balabo

        Integer ::  omp_get_max_threads
      
        Real(kind=8) :: Nusselt

! ======================================================================

        Open (1, file='Lid.dat', form='formatted',   status='old')
        Open (2, file='Lid.out', form='formatted',   status='unknown')
        Open (3, file='Lid.ddd', form='unformatted', status='unknown', Err=881)

        Open (60, file='Pres.dat', form='formatted',  status='unknown')
        Open (70, file='Vx.dat',   form='formatted',  status='unknown')
        Open (80, file='Vy.dat',   form='formatted',  status='unknown')
        Open (90, file='Vz.dat',   form='formatted',  status='unknown')

        Open (25, file='Serie.dat', form='formatted', status='unknown' , position='append')
        Open (26, file='History.dat', form='formatted', status='unknown' , position='append')
        Open (77, file='Cpu.dat', form='formatted', status='unknown' , position='append')
        Open (88, file='HistPoint.dat', form='formatted', status='unknown' , position='append')
        
        Open (777, file='AllData.dat', form='formatted',  status='unknown')

! #######  Input of parameters #########################

        Read (1,101) AspRa, WidRa, Reynolds, angle
        Read (1,100) Nx, Ny, Nz
        Read (1,101) Eps, EpsCnv
        Read (1,100) ItMax
        Read (1,101) Htime, Time
        Read (1,100) Iprint, IprintIBM
        Read (1,101) Delta
        Read (1,100) Icheck
        Read (1,100) SolverKind,Ncpus

        Write (*,200)  Nx, Ny, Nz
        Write (*,201)  AspRa, WidRa,  Reynolds, angle
        Write (*,205)  Htime, Time
        Write (*,203)  ItMax, Iprint , Eps

        Write (2,200)  Nx, Ny, Nz
        Write (2,201)  AspRa, WidRa, Reynolds, angle
        Write (2,205)  Htime, Time
        Write (2,203)  ItMax, Iprint , Eps

!        angle = atan(1.d0)

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

! ########  Call of the main subroutine ################

        Call omp_set_num_threads(Ncpus)
        Write (*,*) omp_get_max_threads(), '  CPUs will be used'

        Call    Solution_time

! ....... output for comparizon ...........................

          Write (*,*) ' U=', Sum(VMx)
          Write (*,*) ' V=', Sum(VMy)
          Write (*,*) ' W=', Sum(VMz)
          
          
          
        Close (1)
        Close (2)
        Close (3)

        Close (60)
        Close (70)
        Close (80)
        Close (90)

        Close (25)
        Close (26)
        Close (77)
        Close (88)
        Close (777) 
          
          

        Stop
881        Write (*,8001)  
        Stop

8001       Format (' ConvMain:  Attempt to read initial values from file is not successfull' )
100        Format (I5)
101        Format (G15.8)
200        Format (1X,I3,' x-points  ',  /, 1x, I3,' y-points',/, 1x, I3,' z-points',//)
201        Format (  ' Aspect Ratio                =',G15.8,/,  &
     &               ' Width  Ratio                =',G15.8,/,  &
     &               ' Reynolds number             =',G15.8,/,  &
     &               ' driving angle               =',G15.8,/,  //)
     
205        Format (  ' Initial time step             =', G15.8,/, &
     &               ' Time interval                 =', G15.8)

203        Format (  ' Maximal number of iterations  =', I5,/,    &
     &               ' Print step                    =', I5,/,    &
     &               ' Precision                     =', G15.8,/)
208        Format (//,'  Computational status:')
210        Format (/,A80, //)
        End

