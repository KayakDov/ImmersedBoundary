! ************************************************************
! *  Main program for unsteady convection                    *
! *            in rectangular cavities                       *
! *                                                          *
! ************************************************************

         Use Numbers
         Use Grid
         Use Variables
         Use EVD_Operators
         use Thomas_coefficients

        Implicit Real(kind=8) (A-H,O-Z)
 
        Integer ::  omp_get_max_threads
        
        Real(kind=4), Dimension(2) :: Time
        Real(kind=4)               :: t0, t1
      
! ======================================================================

        Nn = 256
        Nx = Nn;  Ny=Nn;  Nz=Nn
        Nx1 = Nx + 1
        Nx2 = Nx + 2
        Nx3 = Nx + 3
        Ny1 = Ny + 1
        Ny2 = Ny + 2
        Ny3 = Ny + 3
        Nz1 = Nz + 1
        Nz2 = Nz + 2
        Nz3 = Nz + 3
        
! ............ Make grid ...........................
        
        Call   Mesh

! .................. Make eigenvalues decompositions ...................

        Call    EVDLapTmpr 

! ..... Make r.h.s. .....

        FDRHP = 1.d0
        

! .......  Define number of CPUs ..........

        Write (*,*) ' How many CPUs?'
        Read (*,*) Ncpus
        
        Call omp_set_num_threads(Ncpus)
        Write (*,*) omp_get_max_threads(), '  CPUs will be used'

! ...........  Run TPF ......................................................
 
        t0 = etime(Time);  t0 = Time(1)
        
       Call  EVDmethod (TmpOld(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1), &
&                        ExTemp,Ex_invTemp,  EyTemp,Ey_invTemp,  EzTemp,Ez_invTemp, LambxTemp, LambyTemp, LambzTemp, Nx1, Ny1, Nz1, &
&                      1.d0, 1.d0, 1.d0, 0.d0)
        t1 = etime(Time); t1 = Time(1)
        Write (*,*) '  TPF worked ',t1-t0, ' secs', '   Tmp=',Sum(abs(TmpOld(1:Nx1,1:Ny1,1:Nz1)))        
        
! ...........  Run TPT ......................................................

        t0 = etime(Time);  t0 = Time(1)
        
       Call  EVD_Thomas (TmpNew(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1), &
&                         EyTemp,Ey_invTemp,  EzTemp,Ez_invTemp, &
&                         LambyTemp, LambzTemp, T_left, T_center, T_right, Nx1, Ny1, Nz1, 1.d0)

        t1 = etime(Time); t1 = Time(1)
        Write (*,*) '  TPT worked ',t1-t0, ' secs', '   Tmp=',Sum(abs(TmpNew(1:Nx1,1:Ny1,1:Nz1)))    
        Write (*,*) ' Diff=', maxval(abs( TmpNew(1:Nx1,1:Ny1,1:Nz1) - TmpOld(1:Nx1,1:Ny1,1:Nz1) ))

! ...........  Run modified TPF ......................................................
 
        t0 = etime(Time);  t0 = Time(1)
        
       Call  EVDmethod1 (TmpNew(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1), &
&                        ExTemp,Ex_invTemp,  EyTemp,Ey_invTemp,  EzTemp,Ez_invTemp, LambxTemp, LambyTemp, LambzTemp, Nx1, Ny1, Nz1, &
&                      1.d0, 1.d0, 1.d0, 0.d0)

        t1 = etime(Time); t1 = Time(1)
        Write (*,*) '  TPF1 worked ',t1-t0, ' secs', '   Tmp=',Sum(abs(TmpNew(1:Nx1,1:Ny1,1:Nz1)))    
        Write (*,*) ' Diff=', maxval(abs( TmpNew(1:Nx1,1:Ny1,1:Nz1) - TmpOld(1:Nx1,1:Ny1,1:Nz1) ))

! ...........  Run modified TPT ......................................................

        t0 = etime(Time);  t0 = Time(1)
        
       Call  EVD_Thomas1 (TmpNew(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1), &
&                         EyTemp,Ey_invTemp,  EzTemp,Ez_invTemp, &
&                         LambyTemp, LambzTemp, T_left, T_center, T_right, Nx1, Ny1, Nz1, 1.d0)

        t1 = etime(Time); t1 = Time(1)
        Write (*,*) '  TPT1 worked ',t1-t0, ' secs', '   Tmp=',Sum(abs(TmpNew(1:Nx1,1:Ny1,1:Nz1)))    
        Write (*,*) ' Diff=', maxval(abs( TmpNew(1:Nx1,1:Ny1,1:Nz1) - TmpOld(1:Nx1,1:Ny1,1:Nz1) ))

        Stop
        End

