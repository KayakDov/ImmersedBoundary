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
     &                          ( VMz(i,j,k) + VMz(i,j-1,k) )**2  ) *    &
     &                              Hx12(i-1) * Hy12(j-1) * Hz12(k-1)
          End Do
         End Do
        End Do

        Ekinem = SS

        Return
        End   Function Ekinem 
        
