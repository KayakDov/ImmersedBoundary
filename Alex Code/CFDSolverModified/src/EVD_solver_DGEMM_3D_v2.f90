Subroutine  EVDmethod (f_new, f_rhs, Ex, Ex_inv, Ey, Ey_inv, Ez, Ez_inv, Lambx, Lamby, Lambz, Nxsol, Nysol, NzSol, &
&                      alphaX, alphaY, alphaZ, beta)

        Implicit Real*8 (A-H,O-Z)
         
        Real(kind=8), dimension(1:Nxsol,1:Nxsol):: Ex,Ex_inv
        Real(kind=8), dimension(1:Nxsol)        :: Lambx
        Real(kind=8), dimension(1:Nysol,1:Nysol):: Ey,Ey_inv
        Real(kind=8), dimension(1:Nysol)        :: Lamby
        Real(kind=8), dimension(1:Nzsol,1:Nzsol):: Ez,Ez_inv
        Real(kind=8), dimension(1:Nzsol)        :: Lambz
        Real(kind=8), dimension(1:Nxsol,1:Nysol,1:Nzsol):: f_new, f_rhs

        Real(kind=8),  dimension(1:Nzsol,1:Nxsol):: Amat1, Amat2
        Real(kind=8),  dimension(1:Nysol,1:Nxsol):: Bmat1, Bmat2
        Real(kind=8),  dimension(1:Nxsol,1:Nysol):: Bmat3

        Integer ::   clock_rate, clock_start, clock_stop1, clock_stop2, clock_stop12
! ===================================================================

!$OMP Parallel do   Private(j, Amat1, Amat2) FirstPrivate(NxSol, NzSol, Ez_inv) 
  Do j=1,Nysol
!        f_new (i,1:Nysol,1:Nzsol) = &
!       &    transpose(matmul(Ez_Inv(1:Nzsol,1:Nzsol),transpose(f_rhs(i,1:Nysol,1:Nzsol))))

     Amat1(1:Nzsol,1:Nxsol) = Transpose( f_rhs(1:Nxsol,j,1:Nzsol) )
     Call DGEMM('N', 'N', Nzsol, Nxsol, Nzsol, 1.d0, Ez_inv, Nzsol, Amat1, Nzsol, 0.d0, Amat2, Nzsol)
    f_new(1:Nxsol,j,1:Nzsol)  = Transpose( Amat2(1:Nzsol,1:Nxsol) )
  End Do
 !  Write (*,*) ' 1 fnew=', Sum(f_new)

!$OMP Parallel do  Schedule(dynamic)  Private(k, Bmat1, Bmat2) FirstPrivate(NxSol, NzSol, Nysol, Ey_inv) 
   Do k=1,Nzsol
 !      f_new (1:Nxsol,1:Nysol,k) = &
 !       &   transpose(matmul(Ey_Inv(1:Nysol,1:Nysol),transpose(f_new(1:Nxsol,1:Nysol,k))))

     Bmat1(1:Nysol,1:NxSol) = Transpose( f_new(1:Nxsol,1:Nysol,k) )
      Call DGEMM('N', 'N', Nysol, Nxsol, Nysol, 1.d0, Ey_inv, Nysol, Bmat1, Nysol, 0.d0, Bmat2, Nysol)
     f_New(1:Nxsol,1:Nysol,k) = Transpose( Bmat2(1:Nysol,1:Nxsol) )
   End Do
!   Write (*,*) ' 2 fnew=', Sum(f_new)

!$OMP Parallel do Schedule(dynamic)  Private(k, Bmat1, Bmat3) FirstPrivate(NxSol, Nysol, Ex_inv) 
   Do k=1,Nzsol

       Bmat1(1:Nysol,1:Nxsol) = Transpose( f_new(1:Nxsol,1:Nysol,k) )
        Call DGEMM('N', 'T', Nxsol, Nysol, Nxsol, 1.d0, Ex_inv, Nxsol, Bmat1, Nysol, 0.d0, Bmat3, Nxsol)
       f_new(1:Nxsol,1:Nysol,k) = Bmat3(1:Nxsol,1:Nysol)
   End Do
 !  Write (*,*) ' 3 fnew=', Sum(f_new)

!$OMP Parallel do Private(i,j,k,pdum)
     do i=1,Nxsol
      do j=1,Nysol
       do k=1,Nzsol
	            pdum = alphaX * Lambx(i) + alphaY * Lamby(j) + alphaZ * Lambz(k) + beta
	            if (abs(pdum) <= 1.D-8) pdum = 1.D0

	            f_new(i,j,k) = f_new(i,j,k) / pdum
       end do
      end do
     end do
 !  Write (*,*) ' 4 fnew=', Sum(f_new)

!$OMP Parallel do  Schedule(dynamic)  Private(k, Bmat1, Bmat3) FirstPrivate(NxSol, Nysol, Ex) 
   Do k=1,Nzsol
!       f_new(1:Nxsol,1:Nysol,k) = &
!       &   matmul(Ex(1:Nxsol,1:Nxsol),f_new(1:Nxsol,1:Nysol,k))

       Bmat1(1:Nysol,1:Nxsol) = Transpose( f_new(1:Nxsol,1:Nysol,k) )
        Call DGEMM('N', 'T', Nxsol, Nysol, Nxsol, 1.d0, Ex, Nxsol, Bmat1, Nysol, 0.d0, Bmat3, Nxsol)
       f_new(1:Nxsol,1:Nysol,k) = Bmat3(1:Nxsol,1:Nysol)
   End Do
 !  Write (*,*) ' 5 fnew=', Sum(f_new)

!$OMP Parallel do  Schedule(dynamic)  Private(k, Bmat1, Bmat2) FirstPrivate(NxSol, Nysol, Ey) 
   Do k=1,Nzsol
!       f_new(1:Nxsol,1:Nysol,k) =  &
!      &    transpose(matmul(Ey(1:Nysol,1:Nysol),transpose(f_new(1:Nxsol,1:Nysol,k))))

    Bmat1(1:Nysol,1:NxSol) = Transpose( f_new(1:Nxsol,1:Nysol,k) )
      Call DGEMM('N', 'N', Nysol, Nxsol, Nysol, 1.d0, Ey, Nysol, Bmat1, Nysol, 0.d0, Bmat2, Nysol)
     f_new(1:Nxsol,1:Nysol,k) = Transpose( Bmat2(1:Nysol,1:Nxsol) )
   End Do
 !  Write (*,*) ' 6 fnew=', Sum(f_new)

!$OMP Parallel do Schedule(dynamic)  Private(j, Amat1, Amat2) FirstPrivate(NzSol, Nxsol, Ez)
   Do j=1,Nysol
!      f_new(i,1:Nysol,1:Nzsol) =  &
!      &     transpose(matmul(Ez(1:Nzsol,1:Nzsol),transpose(f_new(i,1:Nysol,1:Nzsol))))

    Amat1(1:Nzsol,1:Nxsol) = Transpose( f_new(1:Nxsol,j,1:Nzsol) )
     Call DGEMM('N', 'N', Nzsol, Nxsol, Nzsol, 1.d0, Ez, Nzsol, Amat1, Nzsol, 0.d0, Amat2, Nzsol)
    f_new(1:Nxsol,j,1:Nzsol) = Transpose( Amat2(1:Nzsol,1:Nxsol) )   ! was 1:Nysol - pre-existing typo, correct data by accident without bounds checks
   End Do
!   Write (*,*) ' 7 fnew=', Sum(f_new)
    
End Subroutine  EVDmethod
