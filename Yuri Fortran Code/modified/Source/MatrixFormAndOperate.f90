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
 Use, Intrinsic :: ISO_C_BINDING, Only: C_DOUBLE, C_SIZE_T, C_INT32_T
 Use eigenbcgsolver_imeq_mod
 Use IBsetupInetrpRegular
 
 IMPLICIT NONE
 Integer*8 NGLPrs_Temp, NGLPrs
 Real*8 , Parameter:: MinThresh= 1.e-6
 Real factorT, factorQ
 INTEGER :: TotalUnknownsT, TotalUnknownsP,length_row_plus_one,kkk
 INTEGER*8::counterEntriesBAndBtransposed_Prs
 CONTAINS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! GPU (CudaBandedLib) coupled pressure/force solve. init_immersed_eq_d_i32
! pre-factorizes the pressure Laplacian on the GPU once; solve_immersed_eq_primes_d_i32
! then runs the whole BiCGStab loop on the device every time step. This replaces
! Build_B_And_BTranspose's old MKL handles, Precond_RHS_P, BICGf90.f90, and the
! B_P_prime/F_tag block in time_step_Lid3D_z.f90 -- see Init_GPU_IBM_Solver below
! and the call site in time_step_Lid3D_z.f90.
!
! Everything below is now grounded in ImerssedEquation.h/.cu and
! FortranBindings.hpp, not guessed:
!
! (1) Axis permutation (FortranBindings.hpp's initImmersedEq_d_i32): the
!     wrapper unconditionally calls the C++ constructor as
!     initImmersedEq(dim1Length, dim3Length, dim2Length, ...) i.e.
!     height=dim1Length, width=dim3Length, depth=dim2Length, and
!     EigenDecompForFortran.cpp's Doxygen comment says explicitly "rows
!     (Y-dimension) ... cols (X-dimension) ... layers (Z-dimension)". So:
!       dim1Length/dim1Delta/dim1*IsNeumann/dim1*Val -> Y  -> Ny1, Hy12(1)
!       dim2Length/dim2Delta/dim2*IsNeumann/dim2*Val -> Z  -> Nz1, Hz12(1)
!       dim3Length/dim3Delta/dim3*IsNeumann/dim3*Val -> X  -> Nx1, Hx12(1)
!     This is what the README calls "INPUT_FORMAT_YZX" -- it isn't a
!     separate call to make, it's baked into which argument slot gets which
!     value, so there is no missing set_global_input_format after all.
! (2) dim*SegSpacing is eigen::LaplOperatorT (wrapper/LaplOperatorType.h),
!     cast directly from the integer -- not an array length as I guessed
!     last time. EVDLapP's boundary rows (a single P1 or P2 term, not the
!     sum) match "UniformDeltaNodeCenteredLapl = 0" ("unknowns at nodes,
!     walls ON the first/last node's neighbouring position"), not
!     UniformDeltaStaggeredLapl -- so I've corrected this to 0 for all three
!     axes. There's no Fortran-side named constant for it in the wrapper you
!     gave me, hence the bare literal below.
! (3) Flat-index convention: EigenDecompForFortran.h states the internal
!     storage order is "dim1 (fastest varying), dim2, dim3 (slowest)" for
!     the GridDim(rows, cols, layers) constructor -- i.e. Y fastest, X
!     middle, Z slowest. That is NOT the same order OrdVarPres uses for
!     NumGlP (X fastest, Y middle, Z slowest). B's column indices and R's
!     row indices are grid-space indices that came from NumGlP (via
!     Div_F_X_ROW / R_Ftag_Matrix_Fx_Row), so they have to be converted --
!     see LibGridIdxFromNumGlP below. p0/f0 are all-zero so no conversion
!     is needed there. uStar and resultPPrime are handled directly in
!     time_step_Lid3D_z.f90 using the library's index order from the start
!     (see the comment there for the velocity component sizes, which are
!     NOT all Nx1*Ny1*Nz1 -- confirmed against bounds_Lid3D_z.f90's ghost
!     assignments, e.g. VMxNew(0,:,:) and VMxNew(Nx1,:,:)).
!
! Still not verified: whether the library's divergence stencil in
! ImerssedEquation.cu's setRHSPPrime matches Yuri's FdDiv/Ckor exactly (both
! look like standard 2nd-order MAC divergence scaled by 3/(2*dt), but I
! haven't proven the constants and boundary treatment agree). Compare
! against orig/Source's CPU path for the same case before trusting a full
! run.
SUBROUTINE Init_GPU_IBM_Solver
    IMPLICIT NONE
    INTEGER :: nnzB, nnzR, i
    REAL(C_DOUBLE), ALLOCATABLE :: p0(:), f0(:)
    REAL(C_DOUBLE) :: dim1Delta_(1), dim2Delta_(1), dim3Delta_(1)

    nnzB = SIZE(B_CSR_Prs)
    nnzR = SIZE(R_CSC_Val)

    IF (ALLOCATED(B_RowOffsets0)) DEALLOCATE(B_RowOffsets0)
    IF (ALLOCATED(B_ColInds0))    DEALLOCATE(B_ColInds0)
    ALLOCATE(B_RowOffsets0(3*TotalUnknownsP+1), B_ColInds0(nnzB))
    B_RowOffsets0 = B_Row_CSR_Prs(1:3*TotalUnknownsP+1) - 1
    DO i = 1, nnzB
        B_ColInds0(i) = LibGridIdxFromNumGlP(B_Col_CSR_Prs(i)) - 1
    END DO

    IF (ALLOCATED(R_ColOffsets0)) DEALLOCATE(R_ColOffsets0)
    IF (ALLOCATED(R_RowInds0))    DEALLOCATE(R_RowInds0)
    ALLOCATE(R_ColOffsets0(3*TotalUnknownsP+1), R_RowInds0(nnzR))
    R_ColOffsets0 = R_ColOffsets_CSC(1:3*TotalUnknownsP+1) - 1
    DO i = 1, nnzR
        R_RowInds0(i) = LibGridIdxFromNumGlP(R_RowInds_CSC(i)) - 1
    END DO

    ALLOCATE(p0(Nx1*Ny1*Nz1), f0(3*TotalUnknownsP))
    p0 = 0.D0
    f0 = 0.D0

    ImEqSolverForceSize = INT(3*TotalUnknownsP, C_SIZE_T)

    dim1Delta_(1) = Hy12(1)
    dim2Delta_(1) = Hz12(1)
    dim3Delta_(1) = Hx12(1)

    CALL init_immersed_eq_d_i32( &
        INT(Ny1, C_SIZE_T), INT(Nz1, C_SIZE_T), INT(Nx1, C_SIZE_T), &
        .TRUE., .TRUE., &   ! dim1 (Y) Neumann both ends
        .TRUE., .TRUE., &   ! dim2 (Z) Neumann both ends
        .TRUE., .TRUE., &   ! dim3 (X) Neumann both ends
        0.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0, &   ! homogeneous Neumann values
        INT(0, C_SIZE_T), INT(0, C_SIZE_T), INT(0, C_SIZE_T), &  ! UniformDeltaNodeCenteredLapl, all 3 axes
        ImEqSolverForceSize, INT(MAX(nnzB, nnzR), C_SIZE_T), &
        p0, f0, dim1Delta_, dim2Delta_, dim3Delta_, &
        Htime, &
        .TRUE., .TRUE., .TRUE., &
        Eps, INT(ItMax, C_SIZE_T))

    ImEqSolverInitialized = .TRUE.
    DEALLOCATE(p0, f0)
END SUBROUTINE Init_GPU_IBM_Solver


! Converts a 1-based NumGlP-style flat grid index (X fastest, then Y, then Z
! -- OrdVarPres's convention) to the 1-based flat index CudaBandedLib uses
! internally for the SAME (X,Y,Z) grid point (Y fastest, then X, then Z --
! see the note above Init_GPU_IBM_Solver). Both index the same Nx1*Ny1*Nz1
! grid; only the flattening order differs.
INTEGER FUNCTION LibGridIdxFromNumGlP(numGlPIdx) RESULT(libIdx)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: numGlPIdx
    INTEGER :: rem, ii, jj, kk
    rem = numGlPIdx - 1
    ii  = MOD(rem, Nx1) + 1
    rem = rem / Nx1
    jj  = MOD(rem, Ny1) + 1
    kk  = rem / Ny1 + 1
    libIdx = LibGridIdx(ii, jj, kk)
END FUNCTION LibGridIdxFromNumGlP


! The library's own 1-based flat index (Y fastest, X middle, Z slowest) for
! pressure/scalar grid point (i,j,k), i,j,k each in 1..Nx1/Ny1/Nz1. Used
! directly (not via LibGridIdxFromNumGlP) wherever the (i,j,k) triple is
! already in hand, e.g. mapping resultPPrime back into Dprs in
! time_step_Lid3D_z.f90.
INTEGER FUNCTION LibGridIdx(i, j, k) RESULT(libIdx)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: i, j, k
    libIdx = j + (i-1)*Ny1 + (k-1)*Ny1*Nx1
END FUNCTION LibGridIdx


SUBROUTINE Finalize_GPU_IBM_Solver
    IMPLICIT NONE
    IF (.NOT. ImEqSolverInitialized) RETURN
    CALL finalize_immersed_eq_d_i32()
    ImEqSolverInitialized = .FALSE.
END SUBROUTINE Finalize_GPU_IBM_Solver

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
! New cache files for the raw (pre-fusion) R triplets -- needed for the GPU
! solver's R argument, not read/written by the original (orig/) code at all.
Open  (12346, file='R_Reg_Fx_Tag.ddd',   form='unformatted',access='stream',status='unknown')
Open  (12347, file='R_Reg_Fy_Tag.ddd',   form='unformatted',access='stream',status='unknown')
Open  (12348, file='R_Reg_Fz_Tag.ddd',   form='unformatted',access='stream',status='unknown')


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
        
        ! NOTE: R_Ftag_Matrix_F{x,y,z} used to be deallocated right here, since
        ! only the fused Div_F_*_Tag output (B/BT) was ever needed downstream.
        ! We now also need the raw (pre-fusion) triplets to build R for the GPU
        ! solver, so they're written to the .ddd cache below (new files) instead
        ! of being thrown away, and are deallocated only after Assemble_R_Matrix_CSC
        ! consumes them, further down.
        
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

        Write (12346)   bdy(i)%Number_Of_Matrix_B_Entries_X
        Write (12346)   bdy(i)%R_Ftag_Matrix_Fx
        Write (12346)   bdy(i)%R_Ftag_Matrix_Fx_Row
        Write (12346)   bdy(i)%R_Ftag_Matrix_Fx_Col

        Write (12347)   bdy(i)%Number_Of_Matrix_B_Entries_Y
        Write (12347)   bdy(i)%R_Ftag_Matrix_Fy
        Write (12347)   bdy(i)%R_Ftag_Matrix_Fy_Row
        Write (12347)   bdy(i)%R_Ftag_Matrix_Fy_Col

        Write (12348)   bdy(i)%Number_Of_Matrix_B_Entries_Z
        Write (12348)   bdy(i)%R_Ftag_Matrix_Fz
        Write (12348)   bdy(i)%R_Ftag_Matrix_Fz_Row
        Write (12348)   bdy(i)%R_Ftag_Matrix_Fz_Col
        
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

        Read (12346)  bdy(i)%Number_Of_Matrix_B_Entries_X
        ALLOCATE( bdy(i)%R_Ftag_Matrix_Fx(bdy(i)%Number_Of_Matrix_B_Entries_X), &
                  bdy(i)%R_Ftag_Matrix_Fx_Row(bdy(i)%Number_Of_Matrix_B_Entries_X), &
                  bdy(i)%R_Ftag_Matrix_Fx_Col(bdy(i)%Number_Of_Matrix_B_Entries_X))
        Read (12346)  bdy(i)%R_Ftag_Matrix_Fx
        Read (12346)  bdy(i)%R_Ftag_Matrix_Fx_Row
        Read (12346)  bdy(i)%R_Ftag_Matrix_Fx_Col

        Read (12347)  bdy(i)%Number_Of_Matrix_B_Entries_Y
        ALLOCATE( bdy(i)%R_Ftag_Matrix_Fy(bdy(i)%Number_Of_Matrix_B_Entries_Y), &
                  bdy(i)%R_Ftag_Matrix_Fy_Row(bdy(i)%Number_Of_Matrix_B_Entries_Y), &
                  bdy(i)%R_Ftag_Matrix_Fy_Col(bdy(i)%Number_Of_Matrix_B_Entries_Y))
        Read (12347)  bdy(i)%R_Ftag_Matrix_Fy
        Read (12347)  bdy(i)%R_Ftag_Matrix_Fy_Row
        Read (12347)  bdy(i)%R_Ftag_Matrix_Fy_Col

        Read (12348)  bdy(i)%Number_Of_Matrix_B_Entries_Z
        ALLOCATE( bdy(i)%R_Ftag_Matrix_Fz(bdy(i)%Number_Of_Matrix_B_Entries_Z), &
                  bdy(i)%R_Ftag_Matrix_Fz_Row(bdy(i)%Number_Of_Matrix_B_Entries_Z), &
                  bdy(i)%R_Ftag_Matrix_Fz_Col(bdy(i)%Number_Of_Matrix_B_Entries_Z))
        Read (12348)  bdy(i)%R_Ftag_Matrix_Fz
        Read (12348)  bdy(i)%R_Ftag_Matrix_Fz_Row
        Read (12348)  bdy(i)%R_Ftag_Matrix_Fz_Col
    END IF
END DO 

close  (123)
close  (1234)
close  (12345)
close  (12346)
close  (12347)
close  (12348)

  
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
Call Assemble_R_Matrix_CSC
Call Init_GPU_IBM_Solver
DEALLOCATE( BT_expanded,lambdaTemp, B, BT,B_R_C, BT_R_C)

END  SUBROUTINE Build_B_And_BTranspose


SUBROUTINE Assemble_R_Matrix_CSC
! Builds R in CSC form (Nx1*Ny1*Nz1 grid rows x 3*TotalUnknownsP force columns)
! from the raw bdy(n)%R_Ftag_Matrix_F{x,y,z} triplets -- the same triplets
! Div_Reg_F{x,y,z}_Tag fuses with the gradient to build B, but here used
! unfused. Column (force-index) numbering mirrors exactly the running_index
! bookkeeping Build_B_And_BTranspose uses for B's rows, so R and B agree on
! what force index k means. Must be called after that bookkeeping is
! consistent, i.e. after bdy(n)%R_Ftag_Matrix_F* are populated (fresh build or
! cache read) for every body.
INTEGER :: n, i, running_index, runningIndexR
INTEGER*8 :: totalR
REAL(kind=8),    ALLOCATABLE :: R_All_Val(:)
INTEGER,         ALLOCATABLE :: R_All_ForceIdx(:), R_All_GridIdx(:)

totalR = 0
DO n=1, n_body
    totalR = totalR + bdy(n)%Number_Of_Matrix_B_Entries_X &
                     + bdy(n)%Number_Of_Matrix_B_Entries_Y &
                     + bdy(n)%Number_Of_Matrix_B_Entries_Z
END DO

ALLOCATE( R_All_Val(totalR), R_All_ForceIdx(totalR), R_All_GridIdx(totalR) )

runningIndexR = 1
running_index = 0

DO n=1, n_body
        DO i=1, SIZE(bdy(n)%R_Ftag_Matrix_Fx)
            R_All_Val(runningIndexR)      = bdy(n)%R_Ftag_Matrix_Fx(i)
            R_All_GridIdx(runningIndexR)  = bdy(n)%R_Ftag_Matrix_Fx_Row(i)
            R_All_ForceIdx(runningIndexR) = bdy(n)%R_Ftag_Matrix_Fx_Col(i) + running_index
            runningIndexR = runningIndexR + 1
        END DO
        running_index = running_index + bdy(n)%npts

        DO i=1, SIZE(bdy(n)%R_Ftag_Matrix_Fy)
            R_All_Val(runningIndexR)      = bdy(n)%R_Ftag_Matrix_Fy(i)
            R_All_GridIdx(runningIndexR)  = bdy(n)%R_Ftag_Matrix_Fy_Row(i)
            R_All_ForceIdx(runningIndexR) = bdy(n)%R_Ftag_Matrix_Fy_Col(i) + running_index
            runningIndexR = runningIndexR + 1
        END DO
        running_index = running_index + bdy(n)%npts

        DO i=1, SIZE(bdy(n)%R_Ftag_Matrix_Fz)
            R_All_Val(runningIndexR)      = bdy(n)%R_Ftag_Matrix_Fz(i)
            R_All_GridIdx(runningIndexR)  = bdy(n)%R_Ftag_Matrix_Fz_Row(i)
            R_All_ForceIdx(runningIndexR) = bdy(n)%R_Ftag_Matrix_Fz_Col(i) + running_index
            runningIndexR = runningIndexR + 1
        END DO
        running_index = running_index + bdy(n)%npts

        DEALLOCATE( bdy(n)%R_Ftag_Matrix_Fx, bdy(n)%R_Ftag_Matrix_Fx_Row, bdy(n)%R_Ftag_Matrix_Fx_Col, &
                    bdy(n)%R_Ftag_Matrix_Fy, bdy(n)%R_Ftag_Matrix_Fy_Row, bdy(n)%R_Ftag_Matrix_Fy_Col, &
                    bdy(n)%R_Ftag_Matrix_Fz, bdy(n)%R_Ftag_Matrix_Fz_Row, bdy(n)%R_Ftag_Matrix_Fz_Col )
END DO

! Feeding (ForceIdx, GridIdx) instead of (GridIdx, ForceIdx) to
! Sparse_To_CSR_Format -- i.e. compressing by the FORCE index -- produces CSR
! of R^T, which is exactly CSC of R: R_ColOffsets_CSC(k) is where force-column
! k's entries start, R_RowInds_CSC holds grid-row indices. Same row/col-swap
! trick already used above to derive BT's arrays from B's triplets.
Call Sparse_To_CSR_Format( R_All_Val, R_All_ForceIdx, R_All_GridIdx, totalR, &
                            3*TotalUnknownsP, R_CSC_Val, R_ColOffsets_CSC, R_RowInds_CSC )

DEALLOCATE( R_All_Val, R_All_ForceIdx, R_All_GridIdx )

END SUBROUTINE Assemble_R_Matrix_CSC



! Precond_Matrix_Vector_Product_For_Krylov_Space and Precond_RHS_P (the CPU
! matvec/preconditioning routines BICG5D used) and BICG5D itself have been
! removed from this (modified/) tree -- solve_immersed_eq_primes_d_i32 now
! does this work on the GPU (see Init_GPU_IBM_Solver above and the call site
! in time_step_Lid3D_z.f90). They both called MKL_B_MatVec/MKL_BT_MatVec,
! which are also gone, so they could not be left in place unmodified as dead
! code without breaking the build. Since orig/ already holds the CPU path,
! there was no separate reference copy worth keeping here; diff against
! orig/Source/MatrixFormAndOperate.f90 if you need to see them again.


END MODULE MatrixFormAndOperate
