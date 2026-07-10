Subroutine  EVD_Thomas (f_New, f_rhs, Ey, Ey_inv, Ez, Ez_inv, &
     &                  Lamby, Lambz, P_a, P_bb, P_c, Nxsol, Nysol, Nzsol, Dtm)

   Implicit Real(kind=8) (A-H,O-Z)
         
   Real(kind=8), dimension(1:Nysol,1:Nysol):: Ey,Ey_inv
   Real(kind=8), dimension(1:Nysol)        :: Lamby
   Real(kind=8), dimension(1:Nzsol,1:Nzsol):: Ez,Ez_inv
   Real(kind=8), dimension(1:Nzsol)        :: Lambz
   
   Real(kind=8), dimension(1:Nxsol,1:Nysol,1:Nzsol):: f_New, f_rhs
   Real(kind=8),  dimension(1:Nxsol,1:Nysol,1:Nzsol):: f_hat

   Real(kind=8), dimension(1:Nxsol)        :: X_Thom
   Real(kind=8), dimension(1:Nxsol)        :: P_a,  P_b,  P_c, P_bb
   Real(kind=8), dimension(1:Nxsol)        :: d_RHS, c_tag, d_tag

   Real(kind=8),  dimension(1:Nzsol,1:Nxsol):: Amat1, Amat2
   Real(kind=8),  dimension(1:Nysol,1:Nxsol):: Bmat1, Bmat2
! ===================================================================
   
!   Write (*,*) ' f_rhs=', sum(abs(f_rhs))
   
!$OMP Parallel Do Schedule(dynamic) Private(j, Amat1, Amat2) FirstPrivate(NxSol, NzSol, Ez_inv) 
       Do j=1,Nysol
           
           Amat1(1:Nzsol,1:Nxsol) = Transpose( f_rhs(1:Nxsol,j,1:Nzsol) )
             Call DGEMM('N', 'N', Nzsol, Nxsol, Nzsol, 1.d0, Ez_inv, Nzsol, Amat1, Nzsol, 0.d0, Amat2, Nzsol)
            f_hat(1:Nxsol,j,1:Nzsol) = Transpose( Amat2(1:Nzsol,1:Nxsol) )
       End Do
!   Write (*,*) ' f_hat=', sum(abs(f_hat))
       
!$OMP Parallel Do  Schedule(dynamic) Private(k, Bmat1, Bmat2)  FirstPrivate(NySol, NxSol, Ey_inv) 
       Do k=1,Nzsol

              Bmat1(1:Nysol,1:Nxsol) = Transpose( f_hat(1:Nxsol,1:Nysol,k) )

             Call DGEMM('N', 'N', Nysol, Nxsol, Nysol, 1.d0, Ey_inv, Nysol, Bmat1, Nysol, 0.d0, Bmat2, Nysol)
           f_New(1:Nxsol,1:Nysol,k) = Transpose( Bmat2(1:Nysol,1:Nxsol) )
       End Do
!   Write (*,*) ' f_new=', sum(abs(f_new)) !, Ckor, Htime, Dtm, Sum(Lamby), Sum(Lambz)
!         pause

!$OMP Parallel Do Schedule(dynamic)  Private(j,k,i,pdum,zdum,c_tag, d_tag, X_thom, d_RHS, P_b) FirstPrivate(beta, alphaY,Lamby,alphaZ,Lambz,P_a, P_bb,P_c) Collapse(2)
	 Do j=1,Nysol
	  Do k=1,Nzsol
	            pdum = Lamby(j) + Lambz(k) 
	            if (abs(pdum) <= 1.D-8) pdum = 1.D-0

                  P_b(1:Nxsol) =  P_bb(1:Nxsol) + pdum
                d_RHS(1:Nxsol) = f_New(1:Nxsol,j,k)

  !....... embedded Thomas algorithm ...........................

       ! /* Modify the coefficients */             
                  
         c_tag(1) = P_c(1)/P_b(1)
         d_tag(1) = d_RHS(1)/P_b(1)

         do i=2,Nxsol
            zdum = P_b(i) - c_tag(i-1)*P_a(i)
            c_tag(i) = P_c(i)/zdum
            d_tag(i) =(d_RHS(i) - d_tag(i-1)*P_a(i))/zdum
         end do
 	 
	  !  /* Now back substitute */

	      X_Thom(Nxsol) = d_tag(Nxsol)
	     
	     do i=Nxsol-1,1,-1
               X_Thom(i) = d_tag(i) - c_tag(i) * X_Thom(i+1)
	     end do

  !....... end of embedded Thomas algorithm ...........................
           
                f_New(1:Nxsol,j,k) = X_thom(1:Nxsol)
	  End do
     End do
!   Write (*,*) ' f_new=', sum(abs(f_new))
   
!$OMP Parallel Do Schedule(dynamic)   Private(k, Bmat1, Bmat2),  FirstPrivate(NySol, NxSol, Ey) 
       Do k=1,Nzsol
           Bmat1(1:Nysol,1:Nxsol) = Transpose( f_New(1:Nxsol,1:Nysol,k) )
             Call DGEMM('N', 'N', Nysol, Nxsol, Nysol, 1.d0, Ey, Nysol, Bmat1, Nysol, 0.d0, Bmat2, Nysol)
           f_hat(1:Nxsol,1:Nysol,k) = Transpose( Bmat2(1:Nysol,1:Nxsol) )
       End Do
!   Write (*,*) ' f_hat=', sum(abs(f_hat))

!$OMP Parallel Do Schedule(dynamic) Private(j, Amat1, Amat2) FirstPrivate(NySol, NzSol, Ez) 
       Do j=1,Nysol
           
            Amat1(1:Nzsol,1:Nxsol) = Transpose( f_hat(1:Nxsol,j,1:Nzsol) )
             Call DGEMM('N', 'N', Nzsol, Nxsol, Nzsol, 1.d0, Ez, Nzsol, Amat1, Nzsol, 0.d0, Amat2, Nzsol)
            f_new(1:Nxsol,j,1:Nzsol) = Transpose( Amat2(1:Nzsol,1:Nxsol) )
       End Do
 !  Write (*,*) ' f_new=', sum(abs(f_new))

 Return
end Subroutine  EVD_Thomas
	
