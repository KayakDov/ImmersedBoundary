! *******************************************************
! *   Calculate the kinematic energy                    *
! *******************************************************

        Real(kind=8) Function Ekinem() 

        Use Numbers
        Use Grid
        Use variables

        Implicit Real(kind=8) (a-h,o-z)
        
! =============================================================

! +++++++++++ Volume Integral ++++++++++++++++++++++++++

        SS = 0.D0

!$OMP Parallel Do Private(i,j,k), Reduction(+:SS)
        Do i=1,Nx1
         Do j=1,Ny1
          Do k=1,Nz1

          SS = SS + 0.125D0 * ( ( VMx(i,j,k) + VMx(i-1,j,k) )**2 +       &
     &                          ( VMy(i,j,k) + VMy(i,j-1,k) )**2 +       &
     &                          ( VMz(i,j,k) + VMz(i,j,k-1) )**2  ) *    &
     &                              Hx12(i-1) * Hy12(j-1) * Hz12(k-1)
          End Do
         End Do
        End Do

        Ekinem = SS

        Return
        End   Function Ekinem 
        
 
        Real(kind=8) Function Nusselt() 

        Use Numbers
        Use Parameters
        Use Grid
        Use variables

        Implicit Real(kind=8) (a-h,o-z)
        
! =============================================================

        Snu = 0.D0

          div = HPx(0) * HPx(1) * ( HPx(0) + HPx(1) )
          f0  = HPx(1) * ( HPx(1) + 2.d0 * HPx(0) )
          f1  = - ( HPx(1) + HPx(0) )**2
          f2  = HPx(0)**2

 !$OMP Parallel Do Private(j,k,DT), Reduction(+:Snu)
        Do j=1,Ny1
          Do k=1,Nz1
  !          DT = ( Tmpr(0,j,k) - Tmpr(1,j,k) ) / HPx(0)
            DT = ( f0 * Tmpr(0,j,k) + f1 * Tmpr(1,j,k) + f2 * Tmpr(2,j,k) ) / div 

             Snu = Snu + DT*Hy12(j-1)*Hz12(k-1)
             
          End Do
!          pause
         End Do

         Nusselt = 1.d0/AspRa + Snu  / WidRa 

        Return
    End Function Nusselt
    
    Real(kind=8) Function Nusselt_middle() 

        Use Numbers
        Use Parameters
        Use Grid
        Use variables

        Implicit Real(kind=8) (a-h,o-z)
        
! =============================================================

        Snu = 0.D0

 !$OMP Parallel Do Private(j,k,DT), Reduction(+:Snu)
        Do k=1,Nz1
            DT = ( Tmpr(0,Ny2/2,k) - Tmpr(1,Ny2/2,k) ) / HPx(0)

             Snu = Snu + DT*Hz12(k-1)
         End Do

         Nusselt_middle = 1.d0/AspRa + Snu

        Return
        End Function Nusselt_middle
