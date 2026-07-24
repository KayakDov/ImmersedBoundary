Subroutine Average_flow

         Use Numbers
         Use Parameters
         Use Grid
         Use Variables

        Implicit Real(kind=8) (a-h,o-z)
        
        Real(kind=8) :: Ux(0:Nx1,0:Nz2), Uz(0:Nx2,0:Nz1), Psi(0:Nx2,0:Nz2), Tr(0:Nx2,0:Nz2)
        
! =============================================================
               
!.................. Average velocities ................................
                
                   Ux = 0.d0;  Uz = 0.d0;  Tr= 0.d0
    Do  j=1,Ny1
        Ux(0:Nx1,0:Nz2) = Ux(0:Nx1,0:Nz2) +  VMx(0:Nx1,j,0:Nz2) * Hy12(j-1)
        Uz(0:Nx2,0:Nz1) = Uz(0:Nx2,0:Nz1) +  VMz(0:Nx2,j,0:Nz1) * Hy12(j-1)
        Tr(0:Nx2,0:Nz2) = Tr(0:Nx2,0:Nz2) + Tmpr(0:Nx2,j,0:Nz2) * Hy12(j-1)
    End Do
    
        Ux = Ux / WidRa;   Uz = Uz / WidRa;  Tr = Tr / WidRa
        
 !      Do k=0,Nz2
 !         Tr(0:Nx2,k) = Tr(0:Nx2,k) + Teta(0:Nx2, Ny2/2, k)
 !      End Do

       Call   PsiInt
       
       Write (*,*) ' MaxPsi_average=', Maxval(abs( Psi ) ) / DGr
       
       Open(120, file='Psi_Yaverage.dat')
       Open(130, file='Tmpr_Yaverage.dat')
       
       Call Point_Write_2D ( Nx2, Nz2, Psi, X12, Z12, 120, 'Psi       ')
       Call Point_Write_2D ( Nx2, Nz2, Tr,  X12, Z12, 130, 'Tmpr      ')
     Return
 Contains     
        Subroutine PsiInt

! ============================================================

         Do i=0,Nx2
                         Psi(i,0) = 0.D0
                         Psi(i,Nz2) = 0.D0
         End Do

         Do k=1,Nz1

            Psi( 0 ,k) = 0.D0 
            Psi(Nx2,k) = 0.D0 

            Do i=1,Nx1
                vz = ( Uz(i, k ) + Uz(i-1, k ) +Uz(i,k-1) + Uz(i-1,k-1)  ) /4.D0

                Psi(i,k) = Psi(i-1,k) + vz *HPx(i-1)
           End Do
         End Do

           Write (*,*) '      2D Nusselt number  =', Nusselt_2D() 
           Write (*,*) '      2D kinetic energy  =', Ekinem_2D() 
           Write (2,*) '      2D Nusselt number  =', Nusselt_2D() 
           Write (2,*) '      2D kinetic energy  =', Ekinem_2D() 
        Return
        End Subroutine PsiInt

   Real(kind=8) Function Nusselt_2D() 

        Use Numbers
        Use Parameters
        Use Grid
        Use variables

        Implicit Real(kind=8) (a-h,o-z)
        
! =============================================================

        Snu = 0.D0

 !$OMP Parallel Do Private(j,k,DT), Reduction(+:Snu)
        Do j=1,Nz1
            DT = ( Tr(0,j) - Tr(1,j) ) / HPx(0)

             Snu = Snu + DT*Hz12(j-1)
         End Do

         Nusselt_2D = 1.d0/AspRa + Snu

        Return
        End Function Nusselt_2D


   Real(kind=8) Function Ekinem_2D() 

        Use Numbers
        Use Parameters
        Use Grid
        Use variables

        Implicit Real(kind=8) (a-h,o-z)
        
! =============================================================

        Ekinem_2D = 0.D0

        Do i=1,Nx1
         Do k=1,Nz1
             Ekinem_2D = Ekinem_2D + 0.25D0 * ( ( Ux(i,k) + Ux(i-1,k) )**2 +       &
     &                                          ( Uz(i,k) + Uz(i,k-1) )**2  ) *    &
     &                              Hx12(i-1) * Hz12(k-1)
          End Do
         End Do

        Return
   End Function Ekinem_2D
   
End Subroutine Average_flow
