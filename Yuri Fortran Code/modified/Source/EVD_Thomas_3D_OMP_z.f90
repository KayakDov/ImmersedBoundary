Subroutine  EVD_Thomas_z (f_New, f_rhs, Ex,Ex_inv, Ey,Ey_inv, &
     &                                Lambx,Lamby,P_a, P_bb, P_c,Nxsol,Nysol,Nzsol,Dtm)
        
   Use Numbers
   Use Numerica
   Use Grid
   Use Parameters
   Use EVD_Operators
   Use Thomas_coefficients

   Implicit Real(kind=8) (A-H,O-Z)
         
   real(kind=8), dimension(1:Nysol,1:Nysol):: Ey,Ey_inv
   real(kind=8), dimension(1:Nysol)        :: Lamby
   real(kind=8), dimension(1:Nxsol,1:Nxsol):: Ex,Ex_inv
   real(kind=8), dimension(1:Nxsol)        :: Lambx
   
   real(kind=8), POINTER:: f_new(:,:,:), f_rhs(:,:,:)

   real(kind=8), dimension(1:Nzsol)        :: X_Thom
   real(kind=8), dimension(1:Nzsol)        :: P_a, P_b, P_c, P_bb
   real(kind=8), dimension(1:Nzsol)        :: d_RHS, c_tag, d_tag

! ===================================================================

!$OMP Parallel do Private(j,k)
     Do k=1,Nzsol
      Do j=1,Nysol
        f_New(1:NxSol,j,k) = matmul( Ex_Inv(1:Nxsol,1:Nxsol), f_rhs(1:NxSol,j,k) )
      End Do
     End Do
 !    Write (*,*) '1 fnew =' , Sum( abs(F_new(1:Nxsol,1:Nysol,1:Nzsol) ))

!$OMP Parallel do Private(i,k)
     Do i=1,Nxsol
      Do k=1,Nzsol
         f_New(i,1:Nysol,k) = matmul( Ey_Inv(1:Nysol,1:Nysol), f_New(i,1:Nysol,k) )
      End Do
     End Do
 !    Write (*,*) '2 fnew =' , Sum( abs(F_new(1:Nxsol,1:Nysol,1:Nzsol) ))

!$OMP Parallel Do Private(j,k,i,pdum,zdum,c_tag, d_tag, X_thom, d_RHS, P_b) 
	 do j=1,Nysol
	  do k=1,Nxsol
	            pdum =( -(Ckor/Htime)*Dtm +  Lamby(j) + Lambx(k) )
	            if (abs(pdum) <= 1.D-8) pdum = 1.D0

             P_b(1:Nzsol) = P_bb(1:Nzsol) 
           d_RHS(1:Nzsol) = f_New(k,j,1:Nzsol)

           If(Dtm == 0.d0) then
                                P_b(1:Nzsol) = P_b(1:Nzsol) + pdum
                           else
                                P_b(1:Nzsol-1) = P_b(1:Nzsol-1) + pdum
                          !      P_b(Nxsol)     = P_b(Nxsol) + Lamby(j) + Lambz(k)
           End If

! ============ Embedded Thomas algorithm ====================

       ! /* Modify the coefficients */             
                  
         c_tag(1) = P_c(1)/P_b(1)
         d_tag(1) = d_RHS(1)/P_b(1)
         
         do i=2,Nzsol
            zdum = P_b(i) - c_tag(i-1)*P_a(i)
            c_tag(i) = P_c(i) / zdum
            d_tag(i) =( d_RHS(i) - d_tag(i-1)*P_a(i) ) / zdum
         end do
 	 
	  !  /* Now back substitute */

	      X_Thom(Nzsol) = d_tag(Nzsol)
	     
	     do i=Nzsol-1,1,-1
	        X_Thom(i) = d_tag(i) - c_tag(i) * X_Thom(i+1)
	     end do

           f_new(k,j,1:Nzsol) = X_Thom(1:Nzsol)
      end do
     end do

 !    Write (*,*) '3 fnew =' , Sum( abs(F_new(1:Nxsol,1:Nysol,1:Nzsol) ))

!$OMP Parallel do Private(i,k)
     Do i=1,Nxsol
      Do k=1,Nzsol
         f_New(i,1:Nysol,k) = matmul( Ey(1:Nysol,1:Nysol), f_New(i,1:Nysol,k) )
      End Do
     End Do
 !    Write (*,*) '4 fnew =' , Sum( abs(F_new(1:Nxsol,1:Nysol,1:Nzsol) ))

!$OMP Parallel do Private(k,j)
     Do k=1,Nzsol
      Do j=1,Nysol
         f_New(1:NxSol,j,k) = matmul( Ex(1:Nxsol,1:Nxsol), f_New(1:NxSol,j,k) )
      End Do
     End Do
!    Write (*,*) '5 fnew =' , Sum( abs(F_new(1:Nxsol,1:Nysol,1:Nzsol) ))

   Return
 
End Subroutine  EVD_Thomas_z
