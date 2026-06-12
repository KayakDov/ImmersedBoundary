! @@@@@@@@@@@@@ With sorting of the eigenvalues from largest to smallest @@@@@@@@@@@@@@@@@@@@@

Subroutine Vgeev (Mat,E,E_inv,Lamb,Ne)

   implicit real(kind=8)(a-h,o-z)
	
   real(kind=8), dimension(1:Ne,1:Ne):: E, E_inv, VL, Mat, Check
   real(kind=8), dimension(1:Ne)     :: Lamb, WI

   real(kind=8), dimension(6*(Ne+1)) :: WORK
   integer,      dimension(Ne+1)     :: IPIV
   Integer, Dimension(2)             :: Loc

! ==================================================================================
      LWORKevd=4*Ne
      E = 0.D0;   E_inv = 0.D0;   VL=0.D0; Lamb = 0.D0 
      Check(1:Ne,1:Ne)= Mat(1:Ne,1:Ne)

	  Call DGEEV('N','V',Ne, Mat(1:Ne,1:Ne),Ne,Lamb,WI,VL(1:Ne,1:Ne),Ne,E(1:Ne,1:Ne),Ne,WORK,LWORKevd,INFO) 
		       If(info /= 0)  Write (*,*) 'ZGEEV: info=', info
		       If(MaxVal(Abs(WI(1:NE))) > 1.D-06) Write (*,*) ' Complex eigenvalue !!!'

! .......... Normalize eigenvectors ........................................
               
        Do i=1,Ne
                 ps = Sum(E(1:Ne,i))
                 
                 If(ps >= 0.d0) then
                                     pm = 1.d0
                                else
                                     pm = -1.d0
                 End If
        
                 E(1:Ne,i) = pm * E(1:Ne,i) / Dot_Product ( E(1:Ne,i), E(1:Ne,i) )
        End Do
               
! ....... Sort of the eigenvalues from largest to smallest ........
               
        Do i=1,Ne
                 Loc(1:1) = MaxLoc(Lamb);  imax = Loc(1)
                 
                 WI(i) = Lamb(imax);  E_inv(1:Ne,i) = E(1:Ne,imax)
                 Lamb(imax) = -1.d+30
        End Do
        
        E = E_inv;  Lamb = WI

! .................... Inverse eigenvalue matrix .......................
        
      Call DGETF2 (Ne,Ne,E_Inv(1:Ne,1:Ne),Ne,IPIV, INFO)
		If(info /= 0)  Write (*,*) 'VR_inv (DGETF2): info=', info
 
      Call DGETRI (Ne,E_Inv(1:Ne,1:Ne), Ne,IPIV, WORK, Ne, INFO)
		If(info /= 0)  Write (*,*) 'VR_inv (DGETRI): info=', info

!=======  Checking Inverse matrix ===============================

   flag=0
   if (flag==1) then
			VL = 0.D0
			do i=1,Ne
			    VL(i,i)= 1.D0
		    end do
			
            Write(*,*) "max (inverse matrix)=", maxval(abs(matmul(E_inv,E)-VL))
 
 !=======  Checking Eigenvalue decomposition  =====================
            VL = 0.D0
		    do i=1,Ne
			   VL(i,i)= Lamb(i)
		    end do

			VL(1:Ne,1:Ne)=matmul(E(1:Ne,1:Ne),VL(1:Ne,1:Ne))
			VL(1:Ne,1:Ne)=matmul(VL(1:Ne,1:Ne),E_inv(1:Ne,1:Ne))
			
            Write(*,*) "max (eigenvalue decomposition)=", maxval(abs(Check-VL))
   end if

End Subroutine Vgeev 

