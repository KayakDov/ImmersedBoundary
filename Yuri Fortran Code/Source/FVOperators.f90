MODULE FVOperators
  Use Numbers
  Use Grid
  Use Variables
  Use Operators
 CONTAINS
  
! *******************************************************
! *   Finite difference approximation of Div(V)         *
! *                                                     *
! *   DivV(6,Nx1,Ny1)     - FD SCHEME FOR DIV(V)        *
! *   X, X12              - NODES IN X-DIRECTION        *
! *   Y, Y12              - NODES IN Y-DIRECTION        *
! *   Z, Z12              - NODES IN Z-DIRECTION        *
! *   Hx12, Hy12, Hz12    - STEPS OF THE MESH           *
! *******************************************************

        Subroutine DivVel 

        

        Implicit Real(kind=8) (A-H,O-Z)

! ================================================================
 
! ######## Forming FD scheme for Div(V) ##################

        Do i=1,Nx1
         Do j=1,Ny1
          Do k=1,Nz1
                DivV(1,i,j,k) =  1.D0 / Hx12(i-1)
                DivV(2,i,j,k) = -DivV(1,i,j,k)

                DivV(3,i,j,k) =  1.D0 / Hy12(j-1)
                DivV(4,i,j,k) = -DivV(3,i,j,k)

                DivV(5,i,j,k) =  1.D0 / Hz12(k-1)
                DivV(6,i,j,k) = -DivV(5,i,j,k)
          End Do
         End Do
        End Do


        End  Subroutine DivVel 

! ****************************************************************
! *     Computing FD approximation of DIV(V)                     *
! ****************************************************************

        Subroutine FDdiv


        Implicit Real(kind=8) (A-H,O-Z)

! ==================================================================

!$OMP Parallel Do Private(i,j,k)
        Do i=1,Nx1
         Do j=1,Ny1
          Do k=1,Nz1
           FDRHP(i,j,k) = DivV(1,i,j,k) * VMxNew( i ,j, k ) + &
     &                    DivV(2,i,j,k) * VMxNew(i-1,j, k ) + &
     &                    DivV(3,i,j,k) * VMyNew(i, j , k ) + &
     &                    DivV(4,i,j,k) * VMyNew(i,j-1, k ) + &
     &                    DivV(5,i,j,k) * VMzNew(i, j , k ) + &
     &                    DivV(6,i,j,k) * VMzNew(i, j ,k-1)
          End Do
         End Do
        End Do

        End   Subroutine FDdiv
    
! ***************************************************
! *   Finite difference approximation of Grad(P)    *
! *                                                 *
! *   GradPx(Nx,Ny1) -    x-component of Grad(P)    *
! *   GradPy(Nx1,Ny) -    y-component of Grad(P)    *
! *   GradPz(Nx1,Ny) -    z-component of Grad(P)    *
! ***************************************************

        Subroutine  GradPx(ResX, Pressure)

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
        
        
! ************************************************************
! *   Subroutines for computation of the convective terms    *
! ************************************************************


    
    
    
    
! +++++++++++++ (Vgrad)Vx ++++++++++++++++++++++++++++++++++++

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

                RHSx(i,j,k) = RHSx(i,j,k)  + A1*VMx(i  ,j-1,k  ) &
                                           + A2*VMx(i-1,j  ,k  ) &
                                           + A3*VMx(i  ,j  ,k  ) &
                                           + A4*VMx(i  ,j+1,k  ) &
                                           + A5*VMx(i+1,j  ,k  ) &
                                           + A6*VMx(i  ,j  ,k-1) &
                                           + A7*VMx(i  ,j  ,k+1)

          End Do
         End Do
        End Do

    End  Subroutine  VgrdVx

    
! +++++++++++++ (Vgrad)Vy ++++++++++++++++++++++++++++++++++++

        Subroutine  VgrdVy 


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

                RHSy(i,j,k) = RHSy(i,j,k)  + A1*VMy(i  ,j-1,k  ) &
                                           + A2*VMy(i-1,j  ,k  ) &
                                           + A3*VMy(i  ,j  ,k  ) &
                                           + A4*VMy(i  ,j+1,k  ) &
                                           + A5*VMy(i+1,j  ,k  ) &
                                           + A6*VMy(i  ,j  ,k-1) &
                                           + A7*VMy(i  ,j  ,k+1)     
          End Do
         End Do
        End Do

    End  Subroutine  VgrdVy 

    
  

! +++++++++++++ (Vgrad)Vz ++++++++++++++++++++++++++++++++++++

        Subroutine  VgrdVz 

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

                RHSz(i,j,k) = RHSz(i,j,k)   + A1*VMz(i  ,j-1,k  ) &
                                            + A2*VMz(i-1,j  ,k  ) &
                                            + A3*VMz(i  ,j  ,k  ) &
                                            + A4*VMz(i  ,j+1,k  ) &
                                            + A5*VMz(i+1,j  ,k  ) &
                                            + A6*VMz(i  ,j  ,k-1) &
                                            + A7*VMz(i  ,j  ,k+1)     
          End Do
         End Do
        End Do
        End Subroutine  VgrdVz   

! +++++++++++++ (Vgrad)VxPart ++++++++++++++++++++++++++++++++++++
        
    
END MODULE FVOperators
