 
 !  Writing the current result 


 Subroutine Outp

     Use Numbers
     Use Numerica
     Use Grid
     Use Variables

     Implicit real(kind=8) (A-H,O-Z)

     ! ##########  Saving the current result ###################
     ! Memory layout is now (Y, Z, X), so loops must match:
     ! Fastest index is j (Y), then k (Z), then i (X).

     Rewind 3

     ! Note: Index bounds are maintained from original definition but
     ! variables are accessed as (j, k, i)

     Write (3) ((( VMx(j,k,i), j=0,Ny2), k=0,Nz2), i=0,Nx1)
     Write (3) ((( VMy(j,k,i), j=0,Ny1), k=0,Nz2), i=0,Nx2)
     Write (3) ((( VMz(j,k,i), j=0,Ny2), k=0,Nz1), i=0,Nx2)
     Write (3) (((Tmpr(j,k,i), j=0,Ny2), k=0,Nz2), i=0,Nx2)
     Write (3) ((( Prs(j,k,i), j=1,Ny1), k=1,Nz1), i=1,Nx1)

     Write (3) ((( VMxOld(j,k,i), j=0,Ny2), k=0,Nz2), i=0,Nx1)
     Write (3) ((( VMyOld(j,k,i), j=0,Ny1), k=0,Nz2), i=0,Nx2)
     Write (3) ((( VMzOld(j,k,i), j=0,Ny2), k=0,Nz1), i=0,Nx2)
     Write (3) ((( TmpOld(j,k,i), j=0,Ny2), k=0,Nz2), i=0,Nx2)

     Write (3) TimCur

     Return

 End Subroutine Outp