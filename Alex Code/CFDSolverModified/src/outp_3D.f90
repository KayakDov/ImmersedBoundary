! ****************************************************
! *   Writing of the current results                 *
! ****************************************************

 Subroutine Outp

     Use Numbers
     Use Numerica
     Use Variables

    Implicit Real(kind=8) (A-H,O-Z)

! ====================================================

     ! Arrays are stored in the new (Y, Z, X) layout and accessed as
     ! (j, k, i), but the restart file Conv.ddd keeps the ORIGINAL byte
     ! order (x fastest, then y, then z) so that:
     !   (a) legacy restart files written by the unmodified code remain
     !       readable, and
     !   (b) the buffered reader in init_3D.f90, which reads legacy order
     !       and transposes, round-trips correctly.

     Rewind 3

     Write (3) ((( VMx(j,k,i), i=0,Nx1), j=0,Ny2), k=0,Nz2)
     Write (3) ((( VMy(j,k,i), i=0,Nx2), j=0,Ny1), k=0,Nz2)
     Write (3) ((( VMz(j,k,i), i=0,Nx2), j=0,Ny2), k=0,Nz1)
     Write (3) (((Tmpr(j,k,i), i=0,Nx2), j=0,Ny2), k=0,Nz2)
     Write (3) ((( Prs(j,k,i), i=1,Nx1), j=1,Ny1), k=1,Nz1)

     Write (3) ((( VMxOld(j,k,i), i=0,Nx1), j=0,Ny2), k=0,Nz2)
     Write (3) ((( VMyOld(j,k,i), i=0,Nx2), j=0,Ny1), k=0,Nz2)
     Write (3) ((( VMzOld(j,k,i), i=0,Nx2), j=0,Ny2), k=0,Nz1)
     Write (3) ((( TmpOld(j,k,i), i=0,Nx2), j=0,Ny2), k=0,Nz2)

     Write (3) TimCur

    Return
 End Subroutine Outp
