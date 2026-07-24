! ****************************************************************
! *     Computing FD approximation of DIV(V)                     *
! ****************************************************************

        Subroutine FDdiv

         Use Numbers
         Use Grid
         Use Variables
         Use Operators

        Implicit Real(kind=8) (A-H,O-Z)

! ==================================================================

!$OMP Parallel Do Private(i,j,k)
        Do i=1,Nx1
         Do j=1,Ny1
          Do k=1,Nz1
           FDRHP(i,j,k) = ( VMxNew( i ,j, k ) - VMxNew(i-1,j, k ) ) / Hx12(i-1) &
     &                  + ( VMyNew(i, j , k ) - VMyNew(i,j-1, k ) ) / Hy12(j-1) &
     &                  + ( VMzNew(i, j , k ) - VMzNew(i, j ,k-1) ) / Hz12(k-1)
          End Do
         End Do
        End Do

        Return
        End
