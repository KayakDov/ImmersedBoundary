! *********************************************************************
! *                                                                   *
! *   Bi-CGSTAB(2)        *
! *********************************************************************

        Subroutine BICG5D ( Sol,Sg,IterMax, Epsilon, IG, prs_tag)

        Use Numbers
        Use Grid
	    Use Variables 
        Use Matrices
        Use Operators
        Use MatrixFormAndOperate

        Implicit Real*8 (a-h,o-z)
        Integer  Sg
       
        Integer IterMax
     
        Real*8,Dimension (1:Sg):: R, W,V,P,R1,C, S, T,Sol, IG
        Real(kind=8),Dimension(1:Nxx1,1:Nyy1,1:Nzz1) ::   prs_tag
        Character*50  HEAD

! =======================================================================
           
        
         Epsilon2 = Epsilon*Epsilon

! ......... Initial scalars ...................

         ro     = 1.D0
         omeg1  = 1.D0
         omeg2  = 1.D0
         alfa   = 1.D0


! .......... r0 = b - Ac0,  W = V = P = 0 ...........................

         
          !Call mkl_dcsrgemv('N',Sg,Amat,IRN, ICN, IG,C)
          Call Precond_Matrix_Vector_Product_For_Krylov_Space (IG,C, Sg)
          
          R = 0.D0
          W = 0.D0
          V = 0.D0
          P = 0.D0
          R = Sol- C
          R1=R
          C =   IG

! ================ START ITERATIONS =================================

          Do 1111 k=0,IterMax
                    ro1 = - omeg2 * ro

! +++++++++++++ Even Bi-CG step +++++++++++++++++++++++++++++++++++++

! ........... ro = <R1,R> ..........................................

             ro  = ddot(Sg, R, 1, R1, 1) 
             beta = ro * alfa  / ro1
             ro1  = ro

! ............ P = R - beta*(P - omeg1 V - omeg2 W) ..................

             P = R - beta * ( P - omeg1 * V - omeg2 * W )

! .......... V = A P ................................................

             !Call mkl_dcsrgemv('N',Sg,Amat,IRN, ICN, P,V) 
              Call Precond_Matrix_Vector_Product_For_Krylov_Space (P,V, Sg)
! ........... gamma = <V,R1> ........................................ 

             gamma = ddot(Sg, V, 1, R1, 1)
             alfa  = ro / gamma
! ........... R = R - alfa*V ..........................................

               R = R - alfa * V
! .............. S = A R .............................................
             
              !Call mkl_dcsrgemv('N',Sg,Amat,IRN, ICN, R,S)
              Call Precond_Matrix_Vector_Product_For_Krylov_Space (R,S, Sg)
              
! ............ C = C + alfa P  .......................................

               C = C + alfa * P

! ================= Odd  Bi-CG step ====================================

! ........... ro = <R1,S> .............................................

              ro   = ddot(Sg, S, 1, R1, 1) 
              beta = ro * alfa / ro1
              ro1  = ro
                           
      
! ............. V = S - beta V .........................................

               V = S - beta * V
 
! .............. W = A V ...............................................
             
              !Call mkl_dcsrgemv('N',Sg,Amat,IRN, ICN, V,W)
              Call Precond_Matrix_Vector_Product_For_Krylov_Space (V,W, Sg)

! ........... gamma = <W,R1> ...........................................
            
              gamma = ddot(Sg, W, 1, R1, 1) 
              alfa  = ro / gamma

! ........... P = R - beta P ,  R = R - alfa V, S = S - alfa W .........

              P = R - beta * P
              R = R - alfa * V
              S = S - alfa * W

! .............. T = A S ...............................................
            
              !Call mkl_dcsrgemv('N',Sg,Amat,IRN, ICN, S,T)
               Call Precond_Matrix_Vector_Product_For_Krylov_Space (S,T, Sg)

! ============= GMRES(2) - part =========================================

! ......... omeg1 = <R,S>,  amu = <S,S>,  anu = <S,T> ..................
! ......... omeg2 = <R,T>,  tau = <T,T> ................................

              omeg1 = ddot(Sg, S, 1, R, 1) 
              omeg2 = ddot(Sg, T, 1, R, 1) 
              amu   = ddot(Sg, S, 1, S, 1) 
              anu   = ddot(Sg, S, 1, T, 1) 
              tau   = ddot(Sg, T, 1, T, 1) 
              tau   = tau - anu*anu / amu
              omeg2 = ( omeg2 - anu * omeg1 / amu ) / tau
              omeg1 = ( omeg1 - anu * omeg2       ) / amu
! ............ C = C + alfa P + omeg1 R + omeg2 S ....................
! ............ R = R - omeg1 S - omeg2 T

               C = C +  alfa * P + &
                                omeg1 * R + &
                                omeg2 * S

               R = R - omeg1 * S - &
                       omeg2 * T

! ............... Convergence ??????? ...............................

               cnorm = Sqrt( ddot(Sg, C, 1, C, 1))
  
               tstsol = Sqrt(sum(( Sol - C )**2))

               If (cnorm .GT. 1.D0) tstsol =tstsol/cnorm

               test = tstsol
        
                If (test   .LT.  Epsilon) go to 2222

! ............ Copying current iteration to the previous one .........
               
                 Sol = C
                 
1111       Continue

! .................. Convergence is not reached .................

            Write (*,200) IterMax
            Stop


! .................. Convergence is reached .................
2222        Continue
             Sol = C

!           Write (8,*) '  Convergence is reached: iter=',k
             
!$OMP PARALLEL DO DEFAULT(Shared) Private(i,j,k)  
         DO i=1,Nx1
            DO j=1,Ny1
                DO k=1,Nz1
                   prs_tag(i,j,k)=Sol(NumGlP(i,j,k))
            END DO
           END DO
         END DO

             

          Return
200        Format ('  BiCG5d:   No convergence in ', I6, '  iterations')
210        Format ('  BiCG5d:   Division by zero happened')
220        Format ('  iter=', I6, '   test=',G15.8)
          End

! =======================================================================

    
