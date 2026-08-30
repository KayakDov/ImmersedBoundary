! ************************************************************
! *  Forming the stret!hed mesh in !ylindri!al !oordinates   *
! *         for axisymmetri! problems in a !ylinder          *
! *                                                          *
! ************************************************************

        Subroutine    Mesh

         Use Numbers
         Use Parameters
         Use Grid

        Implicit Real(kind=8) (a-h,o-z)

! ==========================================================

        Pi = 4.D0 * Atan(1.D0)
         a = 0.d0 !   0.12D0
         b = 0.d0 !0.12D0
         c = 0.d0 !0.12D0

! ######## Forming X mesh ########################

        h    = 1.D0 / Nx1

        X(0) = 0.D0

        Do i=1,Nx1
            X(i) = X(i-1) + h
        End Do

            X = X - a * Sin( 2.D0 * Pi * X )
            X = X * AspRa

!   ........... Forming i+1/2 points ...........

            X12(0)   = 0.D0
            X12(Nx2) = AspRa 

        Do i=1,Nx1
            X12(i) = ( X(i) + X(i-1) ) / 2.D0
        End Do

!   ............. X steps ...........

        Do i=0,Nx
            Hx12(i) = X(i+1) - X(i)
        End Do

        Do i=0,Nx1
            HPx(i) = X12(i+1) - X12(i)
        End Do

! ######### Forming Y mesh #################

        H    = 1.D0 / Ny1
        Y(0) = 0.D0

        Do i=1,Ny1
            Y(i) = Y(i-1) + H
        End Do

            Y = Y - b * Sin( 2.D0 * Pi * Y )
            Y = Y * WidRa

!   ........... Forming i+1/2 points ...........

        Y12(0)   = 0.D0
        Y12(Ny2) = WidRa 

        Do i=1,Ny1
            Y12(i)  = 0.5D0 * ( Y(i) + Y(i-1) )
        End Do

!    ........ Y steps .............

        Do i=0,Ny
              Hy12(i) = Y(i+1) - Y(i)
        End Do

        Do i=0,Ny1
            HPy(i) = Y12(i+1) - Y12(i)
        End Do

! ######### Forming Z mesh #################

        h    = 1.D0 / Nz1
        Z(0) = 0.D0

        Do i=1,Nz1
            Z(i) = Z(i-1) + h
        End Do

            Z = Z - c * Sin( 2.D0 * Pi * Z )

!   ........... Forming i+1/2 points ...........

        Z12(0)   = 0.D0
        Z12(Ny2) = 1.D0 

        Do i=1,Nz1
            Z12(i)  = 0.5D0 * ( Z(i) + Z(i-1) )
        End Do

!    ........ Y steps .............

        Do i=0,Nz
              Hz12(i) = Z(i+1) - Z(i)
        End Do

        Do i=0,Nz1
            HPz(i) = Z12(i+1) - Z12(i)
        End Do

        Return
        End




