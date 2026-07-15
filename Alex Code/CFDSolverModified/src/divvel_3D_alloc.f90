! ****************************************************************
! *     Computing FD approximation of DIV(V)                     *
! ****************************************************************

    Subroutine FDdiv
        Use Numbers; Use Grid; Use Variables; Use Operators
        Implicit Real(kind=8) (A-H,O-Z)

        !$OMP Parallel Do Private(i,j,k)
        Do i=1,Nx1
            Do k=1,Nz1
                Do j=1,Ny1
                    FDRHP(j,k,i) = ( VMxNew( j,k,i ) - VMxNew(j,k,i-1) ) / Hx12(i-1) &
                            + ( VMyNew( j,k,i ) - VMyNew(j-1,k,i) ) / Hy12(j-1) &
                            + ( VMzNew( j,k,i ) - VMzNew(j,k-1,i) ) / Hz12(k-1)
                End Do
            End Do
        End Do
        Return
    End Subroutine FDdiv