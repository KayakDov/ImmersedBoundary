! ************************************************************
! *             Printing the mesh                            *
! *                                                          *
! ************************************************************

       Subroutine  PrMesh 

         Use Numbers
         Use Grid

        Implicit Real(kind=8) (A-H,O-Z)

! ================================================================

        Write (2,120)
        Write (2,121)  (i, X(i), Hx12(i), X12(i), HPx(i), i=0,Nx)
        Write (2,122)  Nx1, X(Nx1), X12(Nx1), HPx(Nx1)
        Write (2,123)  Nx2, X12(Nx2)

        Write (2,130)
        Write (2,131)  (i, Y(i), Hy12(I), Y12(i),HPy(i), i=0,Ny)
        Write (2,132)  Ny1, Y(Ny1), Y12(Ny1), HPy(Ny1)
        Write (2,133)  Ny2, Y12(Ny2)

        Write (2,140)
        Write (2,131)  (i, Z(i), Hz12(I), Z12(i),HPz(i), i=0,Nz)
        Write (2,132)  Nz1, Z(Nz1), Z12(Nz1), HPz(Nz1)
        Write (2,133)  Nz2, Z12(Nz2)

        Return
120        Format (//,10X,'X mesh',/)
130        Format (//,10X,'Y mesh')
140        Format (//,10X,'Z mesh')
121        Format (' i=',I3, '  X=',  G11.4,'  Hx12=',G11.4, '  X12=',G11.4,  '  HPx=',G11.4)
122        Format (' i=',I3, '  X=', G11.4,    18X, '  X12=',G11.4,  '  HPx=',G11.4)
123        Format (' i=',I3, 33X, '  X12=',G11.4)
131        Format (' i=',I3, '  Y=',  G11.4,'  Hy12=',G11.4, '  Y12=',G11.4,  '  HPy=',G11.4)
132        Format (' i=',I3, '  Y=', G11.4,   18X,  '  Y12=',G11.4,  '  HPy=',G11.4)
133        Format (' i=',I3, 33X, '  Y12=',G11.4)
        End
