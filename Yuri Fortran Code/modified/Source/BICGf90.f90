! *********************************************************************
! *                                                                   *
! *   Bi-CGSTAB(2)  -- REMOVED from this (modified/) tree.            *
! *                                                                   *
! *   solve_immersed_eq_primes_d_i32 (CudaBandedLib) now runs the     *
! *   whole BiCGStab loop on the GPU -- see Init_GPU_IBM_Solver in    *
! *   MatrixFormAndOperate.f90 and the call site in                   *
! *   time_step_Lid3D_z.f90. This subroutine called                   *
! *   Precond_Matrix_Vector_Product_For_Krylov_Space, which is also   *
! *   gone from MatrixFormAndOperate.f90, so it could not be left in  *
! *   place unmodified without breaking the build. orig/ still has    *
! *   the original CPU implementation if you need to see it again.    *
! *                                                                   *
! *********************************************************************
