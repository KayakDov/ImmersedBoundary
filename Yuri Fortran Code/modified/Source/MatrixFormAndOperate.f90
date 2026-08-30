MODULE  MatrixFormAndOperate
 USE Grid
 USE Variables
 Use Numbers
 Use Numerica
 Use Operators
 Use Ibmethod 
 Use Parameters
 Use EVD_Operators
 Use FVOperators
 Use Thomas_coefficients
 Use EvdProcedures
 Use Matrices
 Use, Intrinsic :: ISO_C_BINDING, Only: C_INT
 Use MKL_SPBLAS, Only: SPARSE_MATRIX_T, MATRIX_DESCR, &
                       SPARSE_STATUS_SUCCESS, SPARSE_INDEX_BASE_ONE, &
                       SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_LOWER, &
                       SPARSE_DIAG_NON_UNIT, SPARSE_OPERATION_NON_TRANSPOSE, &
                       mkl_sparse_d_create_csr, mkl_sparse_set_mv_hint, &
                       mkl_sparse_optimize, mkl_sparse_d_mv, mkl_sparse_destroy
 Use IBsetupInetrpRegular
 
 IMPLICIT NONE
 Integer*8 NGLPrs_Temp, NGLPrs
 Real*8 , Parameter:: MinThresh= 1.e-6
 Real factorT, factorQ
 INTEGER :: TotalUnknownsT, TotalUnknownsP,length_row_plus_one,kkk
 INTEGER*8::counterEntriesBAndBtransposed_Prs
 TYPE(SPARSE_MATRIX_T) :: B_MKL_Prs, BT_MKL_Prs
 TYPE(MATRIX_DESCR) :: MKL_General_Descr
 LOGICAL :: MKL_Sparse_Handles_Initialized = .FALSE.
 CONTAINS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! oneMKL Inspector-Executor sparse-matrix handles.  The handles retain
! references to the CSR arrays, so those arrays must remain allocated and
! unchanged until Destroy_MKL_Sparse_Handles is called.
SUBROUTINE Check_MKL_Sparse_Status(status, operation_name)
    IMPLICIT NONE

    INTEGER(C_INT), INTENT(IN) :: status
    CHARACTER(LEN=*), INTENT(IN) :: operation_name

    IF (status /= SPARSE_STATUS_SUCCESS) THEN
        WRITE(*,*) 'oneMKL sparse operation failed: ', TRIM(operation_name), &
                   ', status = ', status
        ERROR STOP 1
    END IF
END SUBROUTINE Check_MKL_Sparse_Status


SUBROUTINE Initialize_MKL_Sparse_Handles
    IMPLICIT NONE

    INTEGER(C_INT) :: status, expected_calls

    IF (MKL_Sparse_Handles_Initialized) THEN
        CALL Destroy_MKL_Sparse_Handles
    END IF

    MKL_General_Descr%type = SPARSE_MATRIX_TYPE_GENERAL
    MKL_General_Descr%mode = SPARSE_FILL_MODE_LOWER
    MKL_General_Descr%diag = SPARSE_DIAG_NON_UNIT

    status = mkl_sparse_d_create_csr( &
        B_MKL_Prs, SPARSE_INDEX_BASE_ONE, &
        INT(3*TotalUnknownsP, C_INT), INT(Nx1*Ny1*Nz1, C_INT), &
        B_Row_CSR_Prs(1:3*TotalUnknownsP), &
        B_Row_CSR_Prs(2:3*TotalUnknownsP+1), &
        B_Col_CSR_Prs, B_CSR_Prs)
    CALL Check_MKL_Sparse_Status(status, 'mkl_sparse_d_create_csr(B)')

    status = mkl_sparse_d_create_csr( &
        BT_MKL_Prs, SPARSE_INDEX_BASE_ONE, &
        INT(Nx1*Ny1*Nz1, C_INT), INT(3*TotalUnknownsP, C_INT), &
        BT_Row_CSR_Prs(1:Nx1*Ny1*Nz1), &
        BT_Row_CSR_Prs(2:Nx1*Ny1*Nz1+1), &
        BT_Col_CSR_Prs, BT_CSR_Prs)
    CALL Check_MKL_Sparse_Status(status, 'mkl_sparse_d_create_csr(BT)')

    ! Each BiCG solve performs one initial product, four products per
    ! iteration, and the surrounding time-step code performs one more.
    expected_calls = INT(MAX(1, 4*(ItMax+1)+2), C_INT)

    status = mkl_sparse_set_mv_hint( &
        B_MKL_Prs, SPARSE_OPERATION_NON_TRANSPOSE, &
        MKL_General_Descr, expected_calls)
    CALL Check_MKL_Sparse_Status(status, 'mkl_sparse_set_mv_hint(B)')

    status = mkl_sparse_set_mv_hint( &
        BT_MKL_Prs, SPARSE_OPERATION_NON_TRANSPOSE, &
        MKL_General_Descr, expected_calls)
    CALL Check_MKL_Sparse_Status(status, 'mkl_sparse_set_mv_hint(BT)')

    status = mkl_sparse_optimize(B_MKL_Prs)
    CALL Check_MKL_Sparse_Status(status, 'mkl_sparse_optimize(B)')

    status = mkl_sparse_optimize(BT_MKL_Prs)
    CALL Check_MKL_Sparse_Status(status, 'mkl_sparse_optimize(BT)')

    MKL_Sparse_Handles_Initialized = .TRUE.
END SUBROUTINE Initialize_MKL_Sparse_Handles


SUBROUTINE MKL_B_MatVec(x, y)
    IMPLICIT NONE

    REAL(KIND=8), CONTIGUOUS, INTENT(IN) :: x(:)
    REAL(KIND=8), CONTIGUOUS, INTENT(INOUT) :: y(:)
    INTEGER(C_INT) :: status

    IF (.NOT. MKL_Sparse_Handles_Initialized) THEN
        ERROR STOP 'oneMKL sparse handles have not been initialized'
    END IF

    status = mkl_sparse_d_mv( &
        SPARSE_OPERATION_NON_TRANSPOSE, 1.D0, B_MKL_Prs, &
        MKL_General_Descr, x, 0.D0, y)
    CALL Check_MKL_Sparse_Status(status, 'mkl_sparse_d_mv(B)')
END SUBROUTINE MKL_B_MatVec


SUBROUTINE MKL_BT_MatVec(x, y)
    IMPLICIT NONE

    REAL(KIND=8), CONTIGUOUS, INTENT(IN) :: x(:)
    REAL(KIND=8), CONTIGUOUS, INTENT(INOUT) :: y(:)
    INTEGER(C_INT) :: status

    IF (.NOT. MKL_Sparse_Handles_Initialized) THEN
        ERROR STOP 'oneMKL sparse handles have not been initialized'
    END IF

    status = mkl_sparse_d_mv( &
        SPARSE_OPERATION_NON_TRANSPOSE, 1.D0, BT_MKL_Prs, &
        MKL_General_Descr, x, 0.D0, y)
    CALL Check_MKL_Sparse_Status(status, 'mkl_sparse_d_mv(BT)')
END SUBROUTINE MKL_BT_MatVec


SUBROUTINE Destroy_MKL_Sparse_Handles
    IMPLICIT NONE

    INTEGER(C_INT) :: status

    IF (.NOT. MKL_Sparse_Handles_Initialized) RETURN

    status = mkl_sparse_destroy(B_MKL_Prs)
    CALL Check_MKL_Sparse_Status(status, 'mkl_sparse_destroy(B)')

    status = mkl_sparse_destroy(BT_MKL_Prs)
    CALL Check_MKL_Sparse_Status(status, 'mkl_sparse_destroy(BT)')

    MKL_Sparse_Handles_Initialized = .FALSE.
END SUBROUTINE Destroy_MKL_Sparse_Handles

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


SUBROUTINE Sparse_To_CSR_Format (Amat,row,col,sz, n_row_full,  Amat_new, row_new, col_new)
Integer m,n, i,j,n_row_full
Integer*8 sz
Real(kind=8) Amat(1:sz)
Integer row(1:sz), col(1:sz)
Real(kind=8), POINTER:: Amat_new(:)
Integer, POINTER::      row_new(:),col_new(:)  

ALLOCATE( Amat_new(1:sz),row_new(1:n_row_full+1), col_new(1:sz))

Amat_new=0.d0
row_new=0
col_new=0

Do i=1,sz
    row_new(row(i)+1)= row_new(row(i)+1)+1
End Do 

j=2
Do while (j<=n_row_full+1)
    row_new(j)=row_new(j)+row_new(j-1)
    j=j+1
End Do

Do i=1,sz
    Amat_new(row_new(row(i))+1)=Amat(i)
    col_new (row_new(row(i))+1)=col(i)
    row_new(row(i))= row_new(row(i))+1
End Do    
  
Do i=n_row_full,1,-1
    row_new(i+1)= row_new(i)
End Do
row_new(1)=0
row_new=row_new+1
END SUBROUTINE Sparse_To_CSR_Format



SUBROUTINE Allocate_Forces_And_Forces_RHS
INTEGER :: running_index,runningIndexB,n,m,i,j,k, Szx, Szy, Szz

DO i=1,n_body
    TotalUnknownsP=TotalUnknownsP+bdy(i)%npts
END DO 

ALLOCATE (F_tag (1:3*TotalUnknownsP), RHS_F_tag(1:3*TotalUnknownsP)) ! Do not forget to deallocate  F_tag
ALLOCATE( IGPrs(1:(Nx1)*(Ny1)*(Nz1)) )
IGPrs=0.d0

END SUBROUTINE Allocate_Forces_And_Forces_RHS

SUBROUTINE Counter_Entries_B_And_B_Transposed_Prs
INTEGER :: n,m,i,j,k
counterEntriesBAndBtransposed_Prs=0
DO n=1, n_body
     
        counterEntriesBAndBtransposed_Prs= counterEntriesBAndBtransposed_Prs+size(bdy(n)%Div_F_X_Val)+&
                                                                             size(bdy(n)%Div_F_Y_Val)+&
                                                                             size(bdy(n)%Div_F_Z_Val)
     
END DO
END SUBROUTINE Counter_Entries_B_And_B_Transposed_Prs 


SUBROUTINE Build_B_And_BTranspose
INTEGER    :: running_index,runningIndexB,n,m,i,j,k
INTEGER*8 ::Szx, Szy, Szz
REAL :: start_time, end_time, elapsed_time

!Call OrdVarPres
TotalUnknownsP=0
Open  (123,   file='Div_Reg_Fx_Tag.ddd', form='unformatted',access='stream',status='unknown')
Open  (1234,  file='Div_Reg_Fy_Tag.ddd', form='unformatted',access='stream',status='unknown')
Open  (12345, file='Div_Reg_Fz_Tag.ddd', form='unformatted',access='stream',status='unknown')


DO i=1,n_body
    TotalUnknownsP=TotalUnknownsP+bdy(i)%npts
    
    CALL CPU_TIME(start_time)
    IF (IprintIBM > 0) THEN 
         Call Build_R_Ftag_Matrix_P_Attached(i) 
        
 !$OMP PARALLEL SECTIONS DEFAULT(shared) 
 !$OMP SECTION       
         Call Div_Reg_Fx_Tag(bdy(i)%Div_F_X_Val,bdy(i)%Div_F_X_ROW, bdy(i)%Div_F_X_COL, i, HX12(1), Szx) 
 !$OMP SECTION    
         Call Div_Reg_Fy_Tag(bdy(i)%Div_F_Y_Val,bdy(i)%Div_F_Y_ROW, bdy(i)%Div_F_Y_COL, i, HY12(1), Szy)
  !$OMP SECTION          
         Call Div_Reg_Fz_Tag(bdy(i)%Div_F_Z_Val,bdy(i)%Div_F_Z_ROW, bdy(i)%Div_F_Z_COL, i, HZ12(1), Szz)
  !$OMP END PARALLEL SECTIONS       
        CALL CPU_TIME(end_time)
        elapsed_time = end_time - start_time
        PRINT *, "Elapsed time in seconds: ", elapsed_time 
        
         DEALLOCATE( bdy(i)%R_Ftag_Matrix_Fx,bdy(i)%R_Ftag_Matrix_Fx_Row,bdy(i)%R_Ftag_Matrix_Fx_Col,&
                     bdy(i)%R_Ftag_Matrix_Fy,bdy(i)%R_Ftag_Matrix_Fy_Row,bdy(i)%R_Ftag_Matrix_Fy_Col,&
                     bdy(i)%R_Ftag_Matrix_Fz,bdy(i)%R_Ftag_Matrix_Fz_Row,bdy(i)%R_Ftag_Matrix_Fz_Col )      
        
        Write (123),  Szx
        Write (123)   bdy(i)%Div_F_X_Val
        Write (123)   bdy(i)%Div_F_X_ROW
        Write (123)   bdy(i)%Div_F_X_COL
        
        Write (1234)   Szy
        Write (1234)   bdy(i)%Div_F_Y_Val
        Write (1234)   bdy(i)%Div_F_Y_ROW
        Write (1234)   bdy(i)%Div_F_Y_COL
        
        Write (12345)   Szz
        Write (12345)   bdy(i)%Div_F_Z_Val
        Write (12345)   bdy(i)%Div_F_Z_ROW
        Write (12345)   bdy(i)%Div_F_Z_COL
        
    ELSE
         Read (123)  Szx
         Read(1234)  Szy
         Read(12345) Szz
        
         ALLOCATE( bdy(i)%Div_F_X_Val(Szx), bdy(i)%Div_F_X_ROW(Szx), bdy(i)%Div_F_X_COL(Szx))
         ALLOCATE( bdy(i)%Div_F_Y_Val(Szy), bdy(i)%Div_F_Y_ROW(Szy), bdy(i)%Div_F_Y_COL(Szy))
         ALLOCATE( bdy(i)%Div_F_Z_Val(Szz), bdy(i)%Div_F_Z_ROW(Szz), bdy(i)%Div_F_Z_COL(Szz))
         
        Read (123)   bdy(i)%Div_F_X_Val
        Read (123)   bdy(i)%Div_F_X_ROW
        Read (123)   bdy(i)%Div_F_X_COL
    
        Read (1234)   bdy(i)%Div_F_Y_Val
        Read (1234)   bdy(i)%Div_F_Y_ROW
        Read (1234)   bdy(i)%Div_F_Y_COL
        
        Read (12345)   bdy(i)%Div_F_Z_Val
        Read (12345)   bdy(i)%Div_F_Z_ROW
        Read (12345)   bdy(i)%Div_F_Z_COL
    END IF
END DO 

close  (123)
close  (1234)
close  (12345)

  
ALLOCATE( BT_expanded(1:Nx1,1:Ny1,1:Nz1),lambdaTemp(1:3*TotalUnknownsP))
  
BT_expanded=0.d0

Call Counter_Entries_B_And_B_Transposed_Prs
  
ALLOCATE (B(counterEntriesBAndBtransposed_Prs),BT(counterEntriesBAndBtransposed_Prs) )
ALLOCATE (B_R_C(counterEntriesBAndBtransposed_Prs,2),BT_R_C(counterEntriesBAndBtransposed_Prs,2) )
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
B=0.d0
BT=0.d0
  
runningIndexB=1
running_index=0

DO n=1, n_body 
        DO i=1,size(bdy(n)%Div_F_X_Val)
            
            BT(runningIndexB)=      -bdy(n)%Div_F_X_Val(i)
            BT_R_C(runningIndexB,1)= bdy(n)%Div_F_X_ROW(i)                 ! Rows of BT
            BT_R_C(runningIndexB,2)= bdy(n)%Div_F_X_COL(i)+running_index   ! Columns of BT 
            
            B(runningIndexB)=      bdy(n)%Div_F_X_Val(i)
            B_R_C(runningIndexB,1)=bdy(n)%Div_F_X_COL(i)+running_index      ! Rows of B
            B_R_C(runningIndexB,2)=bdy(n)%Div_F_X_ROW(i)                    ! Columns of B 
            
            runningIndexB=runningIndexB+1
        END DO
        running_index=running_index+bdy(n)%npts
    
        DO i=1,size(bdy(n)%Div_F_Y_Val)
            
            BT(runningIndexB)=      -bdy(n)%Div_F_Y_Val(i)
            BT_R_C(runningIndexB,1)= bdy(n)%Div_F_Y_ROW(i)                 ! Rows of BT
            BT_R_C(runningIndexB,2)= bdy(n)%Div_F_Y_COL(i)+ running_index  ! Columns of BT 
            
            B(runningIndexB)=      bdy(n)%Div_F_Y_Val(i)
            B_R_C(runningIndexB,1)=bdy(n)%Div_F_Y_COL(i)+ running_index      ! Rows of B
            B_R_C(runningIndexB,2)=bdy(n)%Div_F_Y_ROW(i)                     ! Columns of B 
            
            runningIndexB=runningIndexB+1
        END DO
        running_index=running_index+bdy(n)%npts
    
        DO i=1,size(bdy(n)%Div_F_Z_Val)
            
            BT(runningIndexB)=      -bdy(n)%Div_F_Z_Val(i)
            BT_R_C(runningIndexB,1)= bdy(n)%Div_F_Z_ROW(i)                 ! Rows of BT
            BT_R_C(runningIndexB,2)= bdy(n)%Div_F_Z_COL(i)+ running_index  ! Columns of BT 
            
            B(runningIndexB)=      bdy(n)%Div_F_Z_Val(i)
            B_R_C(runningIndexB,1)=bdy(n)%Div_F_Z_COL(i)+ running_index      ! Rows of B
            B_R_C(runningIndexB,2)=bdy(n)%Div_F_Z_ROW(i)                     ! Columns of B 
            
            runningIndexB=runningIndexB+1
        END DO
        running_index=running_index+bdy(n)%npts
END DO
 
Call  Sparse_To_CSR_Format (  B(:), B_R_C(:,1), B_R_C(:,2), counterEntriesBAndBtransposed_Prs, 3*TotalUnknownsP,B_CSR_Prs, B_Row_CSR_Prs, B_Col_CSR_Prs)  
Call  Sparse_To_CSR_Format ( BT(:),BT_R_C(:,1),BT_R_C(:,2), counterEntriesBAndBtransposed_Prs, Nx1*Ny1*Nz1,    BT_CSR_Prs,BT_Row_CSR_Prs,BT_Col_CSR_Prs) 
Call Initialize_MKL_Sparse_Handles
DEALLOCATE( BT_expanded,lambdaTemp, B, BT,B_R_C, BT_R_C)!, R_Ftag_Matrix,R_Ftag_Matrix_Row,R_Ftag_Matrix_Col)

END  SUBROUTINE Build_B_And_BTranspose



SUBROUTINE Precond_Matrix_Vector_Product_For_Krylov_Space (vector, reslt, sz)

    Real(kind=8),CONTIGUOUS,POINTER::  res(:),  res1(:)
    Real(kind=8),POINTER::  RHS(:,:,:),precond(:,:,:)
    Integer sz, sz_B, sz_BT, i, j, k
    Real*8,Dimension (1:sz):: vector,reslt
   
    ALLOCATE(res(3*TotalUnknownsP), res1(sz))
    
    CALL MKL_B_MatVec(vector, res)  
    CALL MKL_BT_MatVec(res, res1)
     
    DEALLOCATE (res) 
   
    res1=2.d0*res1
    
    ALLOCATE(RHS(Nx1,Ny1,Nz1), precond(Nx1,Ny1,Nz1))
   
    !$OMP PARALLEL DO DEFAULT(Shared) Private(i,j,k)  
    DO i=1, Nx1
        DO j=1, Ny1
             DO k=1, Nz1
                         RHS(i,j,k)=res1(NumGlP(i,j,k))
                  END DO
        END DO
    END DO
    
    DEALLOCATE (res1) 
    Thomas_f_New=>precond
    Thomas_f_rhs=>RHS
    Call   EVD_Thomas_z (Thomas_f_New, Thomas_f_rhs,                         &
     &                   ExxP(1:Nx1,1:Nx1), Ex_invP(1:Nx1,1:Nx1),            &
     &                   EyP(1:Ny1,1:Ny1),  Ey_invP(1:Ny1,1:Ny1),            &
     &                   LambxP(1:Nx1), LambyP(1:Ny1),                       &
     &                   P_left(1:Nz1), P_center(1:Nz1), P_right(1:Nz1),     &
     &                   Nx1, Ny1, Nz1, 0.D0)

    
   !$OMP PARALLEL DO DEFAULT(Shared) Private(i,j,k)  
    DO i=1, Nx1
        DO j=1, Ny1
             DO k=1, Nz1
                     reslt(NumGlP(i,j,k))=precond(i,j,k)
              END DO
        END DO
    END DO
    
    DEALLOCATE(RHS, precond)
    reslt= reslt+vector
      
    
END  SUBROUTINE Precond_Matrix_Vector_Product_For_Krylov_Space


SUBROUTINE Precond_RHS_P (RHS_P_prime, RHS_F_prime, RHS_Precond)
Real(kind=8),CONTIGUOUS,POINTER:: RHS_F_prime(:)
Real(kind=8),allocatable :: RHS_Precond(:)
Real(kind=8),CONTIGUOUS,POINTER:: temp(:)
Real(kind=8),POINTER:: temp1(:,:,:), temp2(:,:,:)
Real(kind=8),Dimension(0:Nxx2,0:Nyy2,0:Nzz2) :: RHS_P_prime

Integer i, j, k
ALLOCATE(temp(Nx1*Ny1*Nz1),temp1(Nx1,Ny1,Nz1))
IF (.NOT. ALLOCATED (RHS_Precond))  ALLOCATE(RHS_Precond(Nx1*Ny1*Nz1)) !Do not forget to deallocate in the end of the program 
CALL MKL_BT_MatVec(RHS_F_prime, temp)

!$OMP PARALLEL DO DEFAULT(Shared) Private(i,j,k)  
    DO i=1, Nx1
        DO j=1, Ny1
             DO k=1, Nz1
                     temp1(i,j,k)=temp(NumGlP(i,j,k))
              END DO
        END DO
    END DO
    
   DEALLOCATE(temp)
   ALLOCATE(temp2(Nx1,Ny1,Nz1))

   temp2=RHS_P_prime(1:Nx1,1:Ny1,1:Nz1)+2.d0*temp1
   
    Thomas_f_New=> temp1
    Thomas_f_rhs=>temp2
    Call   EVD_Thomas_z (Thomas_f_New, Thomas_f_rhs,                         &
     &                   ExxP(1:Nx1,1:Nx1), Ex_invP(1:Nx1,1:Nx1),            &
     &                   EyP(1:Ny1,1:Ny1),  Ey_invP(1:Ny1,1:Ny1),            &
     &                   LambxP(1:Nx1), LambyP(1:Ny1),                       &
     &                   P_left(1:Nz1), P_center(1:Nz1), P_right(1:Nz1),     &
     &                   Nx1, Ny1, Nz1, 0.D0)

    
 !$OMP PARALLEL DO DEFAULT(Shared) Private(i,j,k)  
    DO i=1, Nx1
        DO j=1, Ny1
             DO k=1, Nz1
                    RHS_Precond(NumGlP(i,j,k))= temp1(i,j,k)
              END DO
        END DO
    END DO
    
    DEALLOCATE(temp1,temp2)   
END  SUBROUTINE Precond_RHS_P


END MODULE MatrixFormAndOperate
