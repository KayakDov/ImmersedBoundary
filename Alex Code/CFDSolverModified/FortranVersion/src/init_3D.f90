! ************************************************************
! *     POSING INITIAL VALUES FOR UNKNOWN FUNCTIONS          *
! ************************************************************

        Subroutine  Init 

         Use Numbers
         Use Parameters
         Use Numerica
         Use Grid
         Use Variables

        Implicit Real(kind=8) (A-H,O-Z)

! ======================================================================

        If (Iprint < 0) then
                 Rewind 3

                  Read (3) (((  VMx(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
                  Read (3) (((  VMy(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
                  Read (3) (((  VMz(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
                  Read (3) ((( Tmpr(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)
                  Read (3) (((  Prs(i,j,k), i=1,Nx1), j=1,Ny1), k=1,Nz1)
                Iprint = - Iprint
 !               Write (*,*) '0'; go to 55

        Read (3, Err=55) ((( VMxOld(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
        Read (3, Err=55) ((( VMyOld(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
        Read (3, Err=55) ((( VMzOld(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
        Read (3, Err=55) ((( TmpOld(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)

        Read(3) Tstart

                 Ckor = 1.5D0; !  Tstart = 0.d0
               Return
        Else

! ........ Meridional flow ..........................

            VMx = 10.D-06
            VMy = 1.D-06
            VMz = 1.D-06
            Prs = 1.D-06
            Tmpr = 1.D-06
        End If

55      Continue

           VMxOld = VMx
           VMyOld = VMy
           VMzOld = VMz
           TmpOld = Tmpr

                Ckor = 1.5D0
          Tstart = 0.d0

        Return
        End


