! *******************************************************
! *   Finite difference approximation of Div(V)         *
! *                                                     *
! *   DivV(6,Nx1,Ny1)     - FD SCHEME FOR DIV(V)        *
! *   X, X12              - NODES IN X-DIRECTION        *
! *   Y, Y12              - NODES IN Y-DIRECTION        *
! *   Z, Z12              - NODES IN Z-DIRECTION        *
! *   Hx12, Hy12, Hz12    - STEPS OF THE MESH           *
! *******************************************************

        Subroutine DivVel 

         Use Numbers
         Use Grid
         Use Operators

        Implicit Real(kind=8) (A-H,O-Z)

! ================================================================
 
! ######## Forming FD scheme for Div(V) ##################

        Do i=1,Nx1
         Do j=1,Ny1
          Do k=1,Nz1
                DivV(1,i,j,k) =  1.D0 / Hx12(i-1)
                DivV(2,i,j,k) = -DivV(1,i,j,k)

                DivV(3,i,j,k) =  1.D0 / Hy12(j-1)
                DivV(4,i,j,k) = -DivV(3,i,j,k)

                DivV(5,i,j,k) =  1.D0 / Hz12(k-1)
                DivV(6,i,j,k) = -DivV(5,i,j,k)
          End Do
         End Do
        End Do

        Return
        End

! ****************************************************************
! *     Computing FD approximation of DIV(V)                     *
! ****************************************************************

        Subroutine FDdiv

         Use Numbers
         Use Variables
         Use Operators

        Implicit Real(kind=8) (A-H,O-Z)

! ==================================================================

!$OMP Parallel Do Private(i,j,k)
        Do i=1,Nx1
         Do j=1,Ny1
          Do k=1,Nz1
           FDRHP(i,j,k) = DivV(1,i,j,k) * VMxNew( i ,j, k ) + &
     &                    DivV(2,i,j,k) * VMxNew(i-1,j, k ) + &
     &                    DivV(3,i,j,k) * VMyNew(i, j , k ) + &
     &                    DivV(4,i,j,k) * VMyNew(i,j-1, k ) + &
     &                    DivV(5,i,j,k) * VMzNew(i, j , k ) + &
     &                    DivV(6,i,j,k) * VMzNew(i, j ,k-1)
          End Do
         End Do
        End Do

        Return
        End
