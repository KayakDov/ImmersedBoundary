! ************************************************************
! *   Subroutines for computation of the convective terms    *
! ************************************************************

! +++++++++++++ (Vgrad)Tmpr ++++++++++++++++++++++++++++++++++++

        Subroutine  VgrTmp 

         Use Numbers
         Use Grid
         Use Variables
         Use Operators

        Implicit Real(kind=8) (A-H,O-Z)

! =============================================================

!$OMP Parallel Do Private(i,j,k, px,py,pz, A1,A2,A3,A4,A5,A6,A7)
        Do i=1,Nx1
                       px = 0.5D0 / Hx12(i-1)
           Do j=1,Ny1
                       py = 0.5D0 / Hy12(j-1) 
            Do k=1,Nz1
                       pz = 0.5D0 / Hz12(k-1) 

                       A1 = - VMy(i,j-1, k ) * py
                       A4 =   VMy(i, j , k ) * py

                       A2 = - VMx(i-1,j, k ) * px
                       A5 =   VMx( i ,j, k ) * px

                       A6 = - VMz( i ,j,k-1) * pz
                       A7 =   VMz( i ,j, k ) * pz
                       
                       A3 = - ( A1 + A2 + A4 + A5 +A6 + A7 )

		       FDRHP(i,j,k) = FDRHP(i,j,k) &
     &                                    + A1 * ( Tmpr(i  ,j-1,k  ) + Teta(i  ,j-1,k  ) ) &
     &                                    + A2 * ( Tmpr(i-1,j  ,k  ) + Teta(i-1,j  ,k  ) ) &
     &                                    + A3 * ( Tmpr(i  ,j  ,k  ) + Teta(i  ,j  ,k  ) ) &
     &                                    + A4 * ( Tmpr(i  ,j+1,k  ) + Teta(i  ,j+1,k  ) ) &
     &                                    + A5 * ( Tmpr(i+1,j  ,k  ) + Teta(i+1,j  ,k  ) ) &
     &                                    + A6 * ( Tmpr(i  ,j  ,k-1) + Teta(i  ,j  ,k-1) ) &
     &                                    + A7 * ( Tmpr(i  ,j  ,k+1) + Teta(i  ,j  ,k+1) )
  
          End Do
         End Do
        End Do

        Return
        End


! +++++++++++++ (Vgrad)Vr ++++++++++++++++++++++++++++++++++++

        Subroutine  VgrdVx 

         Use Numbers
         Use Grid
         Use Variables
         Use Operators

        Implicit Real(kind=8) (A-H,O-Z)

! ================================================================


!$OMP Parallel Do Private(i,j,k, px,py,pz, py1,py2,pz1,pz2, A1,A2,A3,A4,A5,A6,A7)
        Do i=1,Nx
                         px  = 0.25D0 / HPx(i)
         Do j=1,Ny1
                         py  = px / Hy12(j-1) 
                         py1 = Hx12(i-1) * py
                         py2 = Hx12( i ) * py
          Do k=1,Nz1
                         pz  = px / Hz12(k-1) 
                         pz1 = Hx12(i-1) * pz
                         pz2 = Hx12( i ) * pz

                A1 = - py2*VMy(i+1,j-1,k) - py1*VMy(i,j-1,k)
                A4 =   py2*VMy(i+1, j ,k) + py1*VMy(i, j ,k)

                A6 = - pz2*VMz(i+1,j,k-1) - pz1*VMz(i,j,k-1)
                A7 =   pz2*VMz(i+1,j,k  ) + pz1*VMz(i,j,k)

                A2 = - VMx(i-1,j,k) * px
                A5 =   VMx(i+1,j,k) * px

                A3 = - ( A1 + A4 + A6 + A7 )

                FDRHP(i,j,k) = FDRHP(i,j,k)+ A1*VMx(i  ,j-1,k  ) &
     &                                     + A2*VMx(i-1,j  ,k  ) &
     &                                     + A3*VMx(i  ,j  ,k  ) &
     &                                     + A4*VMx(i  ,j+1,k  ) &
     &                                     + A5*VMx(i+1,j  ,k  ) &
     &                                     + A6*VMx(i  ,j  ,k-1) &
     &                                     + A7*VMx(i  ,j  ,k+1)

          End Do
         End Do
        End Do

        Return
        End

! +++++++++++++ (Vgrad)Vy ++++++++++++++++++++++++++++++++++++

        Subroutine  VgrdVy 

         Use Numbers
         Use Grid
         Use Variables
         Use Operators

        Implicit Real(kind=8) (A-H,O-Z)

! ====================================================================

!$OMP Parallel Do Private(i,j,k, px,py,pz, px1,px2,pz1,pz2, A1,A2,A3,A4,A5,A6,A7)
        Do j=1,Ny
                          py = 0.25D0 / HPy(j)
          Do i=1,Nx1
                          px  = py / Hx12(i-1)
                          px1 = Hy12(j-1) * px
                          px2 = Hy12( j ) * px
           Do k=1,Nz1
                          pz  = py / Hz12(k-1)
                          pz1 = Hy12(j-1) * pz
                          pz2 = Hy12( j ) * pz

                A1 = -VMy(i,j-1,k) * py
                A4 =  VMy(i,j+1,k) * py

                A2 = - px2*VMx(i-1,j+1,k) - px1*VMx(i-1,j,k) 
                A5 =   px2*VMx( i ,j+1,k) + px1*VMx( i ,j,k)

                A6 = - pz2*VMz(i,j+1,k-1) - pz1*VMz(i,j,k-1) 
                A7 =   pz2*VMz(i,j+1,k  ) + pz1*VMz(i,j,k  )

                A3 = - (  A2 + A5 +A6 + A7 )

                FDRHP(i,j,k) = FDRHP(i,j,k)+ A1*VMy(i  ,j-1,k  ) &
     &                                     + A2*VMy(i-1,j  ,k  ) &
     &                                     + A3*VMy(i  ,j  ,k  ) &
     &                                     + A4*VMy(i  ,j+1,k  ) &
     &                                     + A5*VMy(i+1,j  ,k  ) &
     &                                     + A6*VMy(i  ,j  ,k-1) &
     &                                     + A7*VMy(i  ,j  ,k+1)     
          End Do
         End Do
        End Do

        Return
        End

! +++++++++++++ (Vgrad)Vz ++++++++++++++++++++++++++++++++++++

        Subroutine  VgrdVz 

         Use Numbers
         Use Grid
         Use Variables
         Use Operators

        Implicit Real(kind=8) (A-H,O-Z)

! ====================================================================

!$OMP Parallel Do Private(i,j,k, px,py,pz, px1,px2,py1,py2, A1,A2,A3,A4,A5,A6,A7)
        Do k=1,Nz
                          pz = 0.25D0 / HPz(k)
          Do i=1,Nx1
                          px  = pz / Hx12(i-1)
                          px1 = Hz12(k-1) * px
                          px2 = Hz12( k ) * px
           Do j=1,Ny1
                          py  = pz / Hy12(j-1)
                          py1 = Hz12(k-1) * py
                          py2 = Hz12( k ) * py

                A6 = -VMz(i,j,k-1) * pz
                A7 =  VMz(i,j,k+1) * pz

                A2 = - px2*VMx(i-1,j,k+1) - px1*VMx(i-1,j,k) 
                A5 =   px2*VMx( i ,j,k+1) + px1*VMx( i ,j,k)

                A1 = - py2*VMy(i,j-1,k+1) - py1*VMy(i,j-1,k) 
                A4 =   py2*VMy(i,j  ,k+1) + py1*VMy(i,j  ,k)

                A3 = - (  A2 + A5 +A1 + A4 )

                FDRHP(i,j,k) = FDRHP(i,j,k) + A1*VMz(i  ,j-1,k  ) &
     &                                      + A2*VMz(i-1,j  ,k  ) &
     &                                      + A3*VMz(i  ,j  ,k  ) &
     &                                      + A4*VMz(i  ,j+1,k  ) &
     &                                      + A5*VMz(i+1,j  ,k  ) &
     &                                      + A6*VMz(i  ,j  ,k-1) &
     &                                      + A7*VMz(i  ,j  ,k+1)     
          End Do
         End Do
        End Do

        Return
        End
