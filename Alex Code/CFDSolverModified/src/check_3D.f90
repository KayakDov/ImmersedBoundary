! *******************************************************
! *   Check of the results                              *
! *   ((Y, Z, X) memory layout: arrays indexed (j,k,i)) *
! *******************************************************

        Subroutine Check

         Use Numbers
         Use Grid
         Use Variables
         Use Operators

        Implicit Real(kind=8) (a-h,o-z)

! =============================================================

         FDRHP = 0.D0

! ########## Check of the Div V  #####################

        Call   FDdiv

        Write (*,*)  ' Div=', Maxval(Abs(FDRHP))

        SSS = 0.D0

        Do i=1,Nx1
         Do j=1,Ny1
          Do k=1,Nz1

              pp = FDRHP(j,k,i)**2

              vx = 0.5*( VMx(j,k,i) + VMx(j,k,i-1) )
              vy = 0.5*( VMy(j,k,i) + VMy(j-1,k,i) )
              vz = 0.5*( VMz(j,k,i) + VMz(j,k-1,i) )
              vv = vx*vx + vy*vy + vz*vz

               If (vv > 1.D0) pp = pp / vv

            If (pp > SSS) then
                                SSS = pp
                                imax = i
                                jmax = j
                                kmax = k
            End If

          End Do
         End Do
        End Do

        SSS = Sqrt(SSS)

        Write (*,540) SSS, imax, jmax, kmax

! ########### Check of the <Grad(P), V> ##################

! +++++++++++ Volumes Integrals ++++++++++++++++++++++++++

        Sx = 0.D0
        Sy = 0.D0
        Sz = 0.D0

! ......... X - component ...................

        Call   GradPx( RHSx(1:Ny1,1:Nz1,1:Nx), Prs )

!$OMP Parallel Do Private(i,j,k), Reduction(+:Sx)
        Do i=1,Nx
         Do j=1,Ny1
          Do  k=1,Nz1
            Sx = Sx + VMx(j,k,i) * RHSx(j,k,i) * HPx(i)*Hy12(j-1) * Hz12(k-1)
          End Do
         End Do
        End Do

! ........... Y - component ..............

        Call    GradPy( RHSy(1:Ny,1:Nz1,1:Nx1), Prs )

!$OMP Parallel Do Private(i,j,k), Reduction(+:Sy)
        Do i=1,Nx1
         Do j=1,Ny
          Do  k=1,Nz1
            Sy = Sy + VMy(j,k,i) * RHSy(j,k,i) * Hx12(i-1) * HPy(j) * Hz12(k-1)
          End Do
         End Do
        End Do

! ........... Z - component ..............

        Call    GradPz( RHSz(1:Ny1,1:Nz,1:Nx1), Prs )

!$OMP Parallel Do Private(i,j,k), Reduction(+:Sz)
        Do i=1,Nx1
         Do j=1,Ny1
          Do  k=1,Nz
            Sz = Sz + VMz(j,k,i) * RHSz(j,k,i) * Hx12(i-1) * Hy12(j-1) * HPz(k)
          End Do
         End Do
        End Do

         SS = Sx + Sy + Sz
         Write (*,550) SS, Sx, Sy, Sz

! ########### Check of the < (Vgrad)Tmpr, Tmpr> ################

! +++++++++++ Volume Integral ++++++++++++++++++++++++++

         FDRHP = 0.D0

         Call  VgrTmp

        Stmpr = 0.D0

!$OMP Parallel Do Private(i,j,k), Reduction(+:Stmpr)
        Do i=1,Nx1
         Do j=1,Ny1
          Do  k=1,Nz1
             Stmpr = Stmpr + FDRHP(j,k,i) * ( Tmpr(j,k,i) + Teta(j,k,i) ) &
    &            * Hx12(i-1) * Hy12(j-1) * Hz12(k-1)
          End Do
         End Do
        End Do

! ########### Check of the < (Vdrad)V, V> ##################

! +++++++++++ Volumes Integrals ++++++++++++++++++++++++++

! ........... x - component ..............................

         FDRHP = 0.D0
         Call   VgrdVx

        Sx   = 0.D0
        Px   = 0.D0

!$OMP Parallel Do Private(i,j,k), Reduction(+:Sx), Reduction(max:Px)
        Do i=1,Nx
         Do j=1,Ny1
          Do k=1,Nz1
            expr = FDRHP(j,k,i) * VMx(j,k,i)
            Sx   = Sx   + expr * HPx(i) * Hy12(j-1) * Hz12(k-1)

            Px = Max(Px,expr)
          End Do
         End Do
        End Do

! ........... y - component ..............................

         FDRHP = 0.D0

         Call  VgrdVy

         Sy = 0.D0
         Py = 0.D0

!$OMP Parallel Do Private(i,j,k), Reduction(+:Sy), Reduction(max:Py)
        Do j=1,Ny
         Do i=1,Nx1
          Do k=1,Nz1
             expr = FDRHP(j,k,i) * VMy(j,k,i)
             Sy = Sy + expr * Hx12(i-1) * HPy(j) * Hz12(k-1)

             Py = Max(Py, expr)
          End Do
         End Do
        End Do

! ........... z - component ..............................

         FDRHP = 0.D0

         Call  VgrdVz

         Sz = 0.D0
         Pz = 0.D0

!$OMP Parallel Do Private(i,j,k), Reduction(+:Sz), Reduction(max:Pz)
        Do j=1,Ny1
         Do i=1,Nx1
          Do k=1,Nz
             expr = FDRHP(j,k,i) * VMz(j,k,i)
             Sz = Sz + expr * Hx12(i-1) * Hy12(j-1) * HPz(k)

             Pz = Max(Pz, expr)
          End Do
         End Do
        End Do

        SS  = Sx + Sy + Sz

        Write (*,510) Stmpr

        Write (*,520) SS, Sx, Sy, Sz
        Write (*,530) Px, Py, Pz

! ************** Calculation of <grad(V*V),V> ****************

!$OMP Parallel Do Private(i,j,k)
         Do i=1,Nx1
          Do j=1,Ny1
           Do k=1,Nz1
            DPrs(j,k,i) = 0.5*( VMx(j,k,i)**2 + VMx(j,k,i-1)**2 + &
     &                          VMy(j,k,i)**2 + VMy(j-1,k,i)**2 + &
     &                          VMz(j,k,i)**2 + VMz(j,k-1,i)**2  )
           End Do
          End Do
         End Do

         SS = 0.D0

! ......... X - component ...................

        Call   GradPx( RHSx(1:Ny1,1:Nz1,1:Nx), Dprs )

!$OMP Parallel Do Private(i,j,k), Reduction(+:SS)
        Do i=1,Nx
         Do j=1,Ny1
          Do k=1,Nz1
            SS = SS + VMx(j,k,i) * RHSx(j,k,i) * HPx(i)*Hy12(j-1) * Hz12(k-1)
          End Do
         End Do
        End Do

! ........... Y - component ..............

        Call    GradPy( RHSy(1:Ny,1:Nz1,1:Nx1), Dprs )

!$OMP Parallel Do Private(i,j,k), Reduction(+:SS)
        Do i=1,Nx1
         Do j=1,Ny
          Do k=1,Nz1
            SS = SS + VMy(j,k,i) * RHSy(j,k,i) * Hx12(i-1) * HPy(j) * Hz12(k-1)
          End Do
         End Do
        End Do

! ........... Z - component ..............

        Call    GradPz( RHSz(1:Ny1,1:Nz,1:Nx1), Dprs )

!$OMP Parallel Do Private(i,j,k), Reduction(+:SS)
        Do i=1,Nx1
         Do j=1,Ny
          Do k=1,Nz1
            SS = SS + VMz(j,k,i) * RHSz(j,k,i) * Hx12(i-1) * Hy12(j-1) * HPz(k)
          End Do
         End Do
        End Do

            SS = SS / 2.D0

        Write (*,560) SS

! _________________________________________________________

        Return
540        Format (/,'  Maximum of Div(V): ',  G15.8, ' in point', 3I4)
550        Format (/,'  Check of <grad(p),V>: =' ,  G15.8, '   Sx=',G15.8, '  Sy=',G15.8, '  Sz=',G15.8)
510        Format (/,' Check of < (Vgrad)Tmpr,Tmpr>: =', G15.8)
520        Format (/,' Check of < (Vgrad)V, V>: =', G15.8, '   Sx=',G15.8, '  Sy=',G15.8, '  Sz=',G15.8)
530        Format (/,' Maximal values of convective terms:',  '   Px =',G15.8, '   Py=',G15.8, '   Pz=',G15.8)
560        Format (/,'  Check of <grad(V*V),V>: =' ,  G15.8)
        End
