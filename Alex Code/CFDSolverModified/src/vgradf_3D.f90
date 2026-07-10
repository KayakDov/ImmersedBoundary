! ************************************************************
! *   Subroutines for computation of the convective terms    *
! ************************************************************

! +++++++++++++ (Vgrad)Tmpr ++++++++++++++++++++++++++++++++++++

    Subroutine VgrTmp
        Use Numbers; Use Grid; Use Variables; Use Operators
        Implicit Real(kind=8) (A-H,O-Z)

        !$OMP Parallel Do Private(i,j,k, px,py,pz, A1,A2,A3,A4,A5,A6,A7)
        Do i=1,Nx1
            px = 0.5D0 / Hx12(i-1)
            Do k=1,Nz1
                pz = 0.5D0 / Hz12(k-1)
                Do j=1,Ny1
                    py = 0.5D0 / Hy12(j-1)

                    A1 = - VMy(j-1,k,i) * py
                    A4 =   VMy(j,k,i)   * py
                    A2 = - VMx(j,k,i-1) * px
                    A5 =   VMx(j,k,i)   * px
                    A6 = - VMz(j,k-1,i) * pz
                    A7 =   VMz(j,k,i)   * pz
                    A3 = - ( A1 + A2 + A4 + A5 + A6 + A7 )

                    FDRHP(j,k,i) = FDRHP(j,k,i) &
                            + A1 * ( Tmpr(j-1,k,i) + Teta(j-1,k,i) ) &
                            + A2 * ( Tmpr(j,k,i-1) + Teta(j,k,i-1) ) &
                            + A3 * ( Tmpr(j,k,i)   + Teta(j,k,i)   ) &
                            + A4 * ( Tmpr(j+1,k,i) + Teta(j+1,k,i) ) &
                            + A5 * ( Tmpr(j,k,i+1) + Teta(j,k,i+1) ) &
                            + A6 * ( Tmpr(j,k-1,i) + Teta(j,k-1,i) ) &
                            + A7 * ( Tmpr(j,k+1,i) + Teta(j,k+1,i) )
                End Do
            End Do
        End Do
        Return
    End Subroutine VgrTmp

! +++++++++++++ (Vgrad)Vr ++++++++++++++++++++++++++++++++++++

    Subroutine VgrdVx
        Use Numbers; Use Grid; Use Variables; Use Operators
        Implicit Real(kind=8) (A-H,O-Z)

        !$OMP Parallel Do Private(i,j,k, px,py,pz, py1,py2,pz1,pz2, A1,A2,A3,A4,A5,A6,A7)
        Do i=1,Nx
            px = 0.25D0 / HPx(i)
            Do k=1,Nz1
                pz  = px / Hz12(k-1)
                pz1 = Hx12(i-1) * pz
                pz2 = Hx12( i ) * pz
                Do j=1,Ny1
                    py  = px / Hy12(j-1)
                    py1 = Hx12(i-1) * py
                    py2 = Hx12( i ) * py

                    A1 = - py2*VMy(j-1,k,i+1) - py1*VMy(j-1,k,i)
                    A4 =   py2*VMy(j,k,i+1)   + py1*VMy(j,k,i)
                    A6 = - pz2*VMz(j,k-1,i+1) - pz1*VMz(j,k-1,i)
                    A7 =   pz2*VMz(j,k,i+1)   + pz1*VMz(j,k,i)
                    A2 = - VMx(j,k,i-1) * px
                    A5 =   VMx(j,k,i+1) * px
                    A3 = - ( A1 + A4 + A6 + A7 )

                    FDRHP(j,k,i) = FDRHP(j,k,i) + A1*VMx(j-1,k,i) + A2*VMx(j,k,i-1) &
                            + A3*VMx(j,k,i)   + A4*VMx(j+1,k,i) &
                            + A5*VMx(j,k,i+1) + A6*VMx(j,k-1,i) &
                            + A7*VMx(j,k+1,i)
                End Do
            End Do
        End Do
        Return
    End Subroutine VgrdVx
! +++++++++++++ (Vgrad)Vy ++++++++++++++++++++++++++++++++++++

    Subroutine VgrdVy
        Use Numbers; Use Grid; Use Variables; Use Operators
        Implicit Real(kind=8) (A-H,O-Z)

        !$OMP Parallel Do Private(i,j,k, px,py,pz, px1,px2,pz1,pz2, A1,A2,A3,A4,A5,A6,A7)
        Do i=1,Nx1
            Do k=1,Nz1
                Do j=1,Ny
                    py = 0.25D0 / HPy(j)
                    px  = py / Hx12(i-1)
                    px1 = Hy12(j-1) * px
                    px2 = Hy12( j ) * px
                    pz  = py / Hz12(k-1)
                    pz1 = Hy12(j-1) * pz
                    pz2 = Hy12( j ) * pz

                    A1 = -VMy(j-1,k,i) * py
                    A4 =  VMy(j+1,k,i) * py
                    A2 = - px2*VMx(j+1,k,i-1) - px1*VMx(j,k,i-1)
                    A5 =   px2*VMx(j+1,k,i)   + px1*VMx(j,k,i)
                    A6 = - pz2*VMz(j+1,k-1,i) - pz1*VMz(j,k-1,i)
                    A7 =   pz2*VMz(j+1,k,i)   + pz1*VMz(j,k,i)
                    A3 = - ( A2 + A5 + A6 + A7 )

                    FDRHP(j,k,i) = FDRHP(j,k,i) + A1*VMy(j-1,k,i) + A2*VMy(j,k,i-1) &
                            + A3*VMy(j,k,i)   + A4*VMy(j+1,k,i) &
                            + A5*VMy(j,k,i+1) + A6*VMy(j,k-1,i) &
                            + A7*VMy(j,k+1,i)
                End Do
            End Do
        End Do
        Return
    End Subroutine VgrdVy

! +++++++++++++ (Vgrad)Vz ++++++++++++++++++++++++++++++++++++

    Subroutine VgrdVz
        Use Numbers; Use Grid; Use Variables; Use Operators
        Implicit Real(kind=8) (A-H,O-Z)

        !$OMP Parallel Do Private(i,j,k, px,py,pz, px1,px2,py1,py2, A1,A2,A3,A4,A5,A6,A7)
        Do i=1,Nx1
            Do k=1,Nz
                pz = 0.25D0 / HPz(k)
                Do j=1,Ny1
                    px  = pz / Hx12(i-1)
                    px1 = Hz12(k-1) * px
                    px2 = Hz12( k ) * px
                    py  = pz / Hy12(j-1)
                    py1 = Hz12(k-1) * py
                    py2 = Hz12( k ) * py

                    A6 = -VMz(j,k-1,i) * pz
                    A7 =  VMz(j,k+1,i) * pz
                    A2 = - px2*VMx(j,k+1,i-1) - px1*VMx(j,k,i-1)
                    A5 =   px2*VMx(j,k+1,i)   + px1*VMx(j,k,i)
                    A1 = - py2*VMy(j-1,k+1,i) - py1*VMy(j-1,k,i)
                    A4 =   py2*VMy(j,k+1,i)   + py1*VMy(j,k,i)
                    A3 = - ( A2 + A5 + A1 + A4 )

                    FDRHP(j,k,i) = FDRHP(j,k,i) + A1*VMz(j-1,k,i) + A2*VMz(j,k,i-1) &
                            + A3*VMz(j,k,i)   + A4*VMz(j+1,k,i) &
                            + A5*VMz(j,k,i+1) + A6*VMz(j,k-1,i) &
                            + A7*VMz(j,k+1,i)
                End Do
            End Do
        End Do
        Return
    End Subroutine VgrdVz