 
 !  Writing the current result 
 
 
        Subroutine Outp

         Use Numbers
         Use Numerica
         Use Grid
         Use Variables

        Implicit real(kind=8) (A-H,O-Z)

! ##########  Saving the current result ###################

        Rewind 3

        Write (3) ((( VMx(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
        Write (3) ((( VMy(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
        Write (3) ((( VMz(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
        Write (3) (((Tmpr(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)
        Write (3) ((( Prs(i,j,k), i=1,Nx1), j=1,Ny1), k=1,Nz1)

        Write (3) ((( VMxOld(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
        Write (3) ((( VMyOld(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
        Write (3) ((( VMzOld(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
        Write (3) ((( TmpOld(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)

        Write (3) TimCur

!       Write (*,200)

        Return
200        Format (' Outp: current results are written')
        End


