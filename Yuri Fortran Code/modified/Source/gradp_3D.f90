! ***************************************************
! *   Finite difference approximation of Grad(P)    *
! *                                                 *
! *   GradPx(Nx,Ny1) -    x-component of Grad(P)    *
! *   GradPy(Nx1,Ny) -    y-component of Grad(P)    *
! *   GradPz(Nx1,Ny) -    z-component of Grad(P)    *
! ***************************************************

        Subroutine  GradPx(ResX, Pressure)
        Use Grid
        Implicit Real(kind=8) (A-H,O-Z)
        
        Real(kind=8), POINTER:: ResX(:,:,:),Pressure(:,:,:)

! ===============================================================

!$OMP Parallel Do Private(i,j,k)
        Do i=1,Nx
         Do j=1,Ny1
          Do k=1,Nz1
            ResX(i,j,k) = ( Pressure(i+1,j,k) - Pressure(i,j,k) ) / HPx(i) 
          End Do
         End Do
        End Do

       
    End Subroutine  GradPx
    
   
   Subroutine GradPy(ResY, Pressure) 
    Use Grid
        Implicit Real(kind=8) (A-H,O-Z)

        Real(kind=8), POINTER:: ResY(:,:,:),Pressure(:,:,:)
! =============================================================

 !$OMP Parallel Do Private(i,j,k)
       Do i=1,Nx1
         Do j=1,Ny
          Do k=1,Nz1
            ResY(i,j,k) = ( Pressure(i,j+1,k) - Pressure(i,j,k) ) / HPy(j) 
          End Do
         End Do
        End Do

       
    End Subroutine GradPy
     
     
    Subroutine GradPz(ResZ, Pressure)    
    Use Grid
        Implicit Real(kind=8) (A-H,O-Z)

        Real(kind=8), POINTER:: ResZ(:,:,:),Pressure(:,:,:)
! =============================================================

!$OMP Parallel Do Private(i,j,k)
        Do i=1,Nx1
         Do j=1,Ny1
          Do k=1,Nz
            ResZ(i,j,k) = ( Pressure(i,j,k+1) - Pressure(i,j,k) ) / HPz(k) 
          End Do
         End Do
        End Do
       
        End   Subroutine GradPz