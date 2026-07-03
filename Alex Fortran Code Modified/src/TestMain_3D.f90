! ************************************************************
! *  Main program for unsteady convection                    *
! *            in rectangular cavities                       *
! *                                                          *
! *  Combined benchmark: runs Alex's original CPU solvers     *
! *  (EVDmethod/EVD_Thomas/EVDmethod1/EVD_Thomas1, via MKL)   *
! *  alongside CudaBandedLib's GPU solvers (dense and Thomas  *
! *  handles), and cross-checks all of them against each      *
! *  other so a single run validates both implementations.    *
! ************************************************************

Use Numbers
Use Grid
Use Variables
Use EVD_Operators
use Thomas_coefficients
use eigenbcgsolver_eigen_mod
use iso_c_binding, only : C_SIZE_T

Implicit Real(kind=8) (A-H,O-Z)

Integer ::  omp_get_max_threads

Real(kind=4), Dimension(2) :: Time
Real(kind=4)               :: t0, t1

integer(C_SIZE_T) :: denseSolverHandle
integer(C_SIZE_T) :: thomasSolverHandle

! ----------------------------------------------------------------------
! GPU buffers: flat index = Y + Z*Ny1 + X*Ny1*Nz1, i.e. Y varies
! fastest, then Z, then X - NOT the same index order as the CPU arrays
! TmpOld/TmpNew/FDRHP below, which are (x, y, z), x fastest.
! xDenseAsXYZ/xThomasAsXYZ (declared further down) hold the GPU results
! re-ordered into (x, y, z) so they can be compared element-by-element
! against the CPU arrays.
! ----------------------------------------------------------------------
real(kind=8), allocatable :: x1dDense(:)
real(kind=8), allocatable :: x1dThomas(:)
real(kind=8), allocatable :: rhs1d(:)

! CPU-vs-GPU cross-check arrays, in the same (x, y, z) layout as
! TmpOld/TmpNew, so maxval(abs(...)) below compares like-for-like.
! Declared allocatable (heap-backed) rather than fixed-size locals:
! at 258^3 doubles each (~137 MB), these would otherwise be automatic
! stack arrays in the main program, which is a reliable way to trigger
! a stack-overflow SIGSEGV regardless of the -heap-arrays compile flag
! (that flag targets subroutine locals; PROGRAM-unit locals aren't
! guaranteed to be covered the same way). ALLOCATE always uses the
! heap, so this sidesteps the issue entirely. Sized to 1:Nx1/1:Ny1/1:Nz1
! rather than the full 0:Nxx2 ghost-cell padding TmpOld/TmpNew carry,
! since that's the only range actually used below.
Real(kind=8), Allocatable :: xDenseAsXYZ(:,:,:), xThomasAsXYZ(:,:,:)

! Loop indices used only for the GPU (y,x,z) -> CPU (x,y,z) re-ordering.
Integer :: ix, iy, iz

! ======================================================================

Nn = 256
Nx = Nn;  Ny=Nn;  Nz=Nn
Nx1 = Nx + 1
Nx2 = Nx + 2
Nx3 = Nx + 3
Ny1 = Ny + 1
Ny2 = Ny + 2
Ny3 = Ny + 3
Nz1 = Nz + 1
Nz2 = Nz + 2
Nz3 = Nz + 3

! AspRa/WidRa are the (i+1/2) far-boundary values MeshStretch writes into
! X12(Nx2)/Y12(Ny2). Set here (before Call Mesh) since, unlike the Z
! boundary, they aren't hardcoded inside MeshStretch itself.
AspRa = 1.d0
WidRa = 1.d0

! ............ Make grid ...........................

Call   Mesh

! .................. Make eigenvalue decompositions (CPU) ................

Call    EVDLapTmpr

! .................. Make eigenvalue decompositions (GPU) .................
! denseSolverHandle  <-> EVDmethod / EVDmethod1   (TPF / TPF1)
! thomasSolverHandle <-> EVD_Thomas / EVD_Thomas1 (TPT / TPT1)

denseSolverHandle = init_eigen_decomp_d( &
        &           int(Ny1, C_SIZE_T), int(Nx1, C_SIZE_T), int(Nz1, C_SIZE_T), &
        &           HPy(0:Ny1), HPx(0:Nx1), HPz(0:Nz1), &
        &           .false., .false., .false., &
        &           .false., .false., .false., .false., .false., .false., &
        &           0.d0, 0.d0, 0.d0, 0.d0, 0.d0, 0.d0, &
        &           .false., .false.)

thomasSolverHandle = init_eigen_decomp_d( &
        &           int(Ny1, C_SIZE_T), int(Nx1, C_SIZE_T), int(Nz1, C_SIZE_T), &
        &           HPy(0:Ny1), HPx(0:Nx1), HPz(0:Nz1), &
        &           .false., .false., .false., &
        &           .false., .false., .false., .false., .false., .false., &
        &           0.d0, 0.d0, 0.d0, 0.d0, 0.d0, 0.d0, &
        &           .false., .true.)

! ..... Make r.h.s. .....

FDRHP = 1.d0

allocate(x1dDense(Nx1*Ny1*Nz1))
allocate(x1dThomas(Nx1*Ny1*Nz1))
allocate(rhs1d(Nx1*Ny1*Nz1))

rhs1d = 1.0d0

! .......  Define number of CPUs ..........

Write (*,*) ' How many CPUs?'
Read (*,*) Ncpus

Call omp_set_num_threads(Ncpus)
Write (*,*) omp_get_max_threads(), '  CPUs will be used'

! ...........  Run TPF (CPU dense) .........................................

t0 = etime(Time);  t0 = Time(1)

Call  EVDmethod (TmpOld(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1), &
        &                ExTemp,Ex_invTemp,  EyTemp,Ey_invTemp,  EzTemp,Ez_invTemp, LambxTemp, LambyTemp, LambzTemp, Nx1, Ny1, Nz1, &
        &                1.d0, 1.d0, 1.d0, 0.d0)
t1 = etime(Time); t1 = Time(1)
Write (*,*) '  TPF worked ',t1-t0, ' secs', '   Tmp=',Sum(abs(TmpOld(1:Nx1,1:Ny1,1:Nz1)))

! ...........  Run TPT (CPU Thomas) .........................................

t0 = etime(Time);  t0 = Time(1)

Call  EVD_Thomas (TmpNew(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1), &
        &                 EyTemp,Ey_invTemp,  EzTemp,Ez_invTemp, &
        &                 LambyTemp, LambzTemp, T_left, T_center, T_right, Nx1, Ny1, Nz1, 1.d0)

t1 = etime(Time); t1 = Time(1)
Write (*,*) '  TPT worked ',t1-t0, ' secs', '   Tmp=',Sum(abs(TmpNew(1:Nx1,1:Ny1,1:Nz1)))
Write (*,*) ' Diff (CPU TPT vs CPU TPF)=', maxval(abs( TmpNew(1:Nx1,1:Ny1,1:Nz1) - TmpOld(1:Nx1,1:Ny1,1:Nz1) ))

! ...........  Run modified TPF (CPU) .......................................

t0 = etime(Time);  t0 = Time(1)

Call  EVDmethod1 (TmpNew(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1), &
        &                ExTemp,Ex_invTemp,  EyTemp,Ey_invTemp,  EzTemp,Ez_invTemp, LambxTemp, LambyTemp, LambzTemp, Nx1, Ny1, Nz1, &
        &                1.d0, 1.d0, 1.d0, 0.d0)

t1 = etime(Time); t1 = Time(1)
Write (*,*) '  TPF1 worked ',t1-t0, ' secs', '   Tmp=',Sum(abs(TmpNew(1:Nx1,1:Ny1,1:Nz1)))
Write (*,*) ' Diff (CPU TPF1 vs CPU TPF)=', maxval(abs( TmpNew(1:Nx1,1:Ny1,1:Nz1) - TmpOld(1:Nx1,1:Ny1,1:Nz1) ))

! ...........  Run modified TPT (CPU) .......................................

t0 = etime(Time);  t0 = Time(1)

Call  EVD_Thomas1 (TmpNew(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1), &
        &                 EyTemp,Ey_invTemp,  EzTemp,Ez_invTemp, &
        &                 LambyTemp, LambzTemp, T_left, T_center, T_right, Nx1, Ny1, Nz1, 1.d0)

t1 = etime(Time); t1 = Time(1)
Write (*,*) '  TPT1 worked ',t1-t0, ' secs', '   Tmp=',Sum(abs(TmpNew(1:Nx1,1:Ny1,1:Nz1)))
Write (*,*) ' Diff (CPU TPT1 vs CPU TPF)=', maxval(abs( TmpNew(1:Nx1,1:Ny1,1:Nz1) - TmpOld(1:Nx1,1:Ny1,1:Nz1) ))

! ...........  Run TPF (GPU dense) .........................................

t0 = etime(Time);  t0 = Time(1)

x1dDense = -9999.0d0
call solve_eigen_decomp_d( &
        denseSolverHandle, &
        x1dDense, &
        rhs1d)

t1 = etime(Time); t1 = Time(1)
Write (*,*) '  GPU TPF worked ',t1-t0, ' secs', '   Tmp=',Sum(abs(x1dDense))

! ...........  Run TPT (GPU Thomas) .........................................

t0 = etime(Time);  t0 = Time(1)

x1dThomas = -9999.0d0
call solve_eigen_decomp_d( &
        thomasSolverHandle, &
        x1dThomas, &
        rhs1d)

t1 = etime(Time); t1 = Time(1)
Write (*,*) '  GPU TPT worked ',t1-t0, ' secs', '   Tmp=',Sum(abs(x1dThomas))
Write (*,*) ' Diff (GPU TPT vs GPU TPF)=', maxval(abs( x1dThomas - x1dDense ))

! ---------------------------------------------------------------------------
! Re-order the flat GPU buffers, which are laid out (y, z, x) with y
! fastest, then z, then x, into xDenseAsXYZ/xThomasAsXYZ using the same
! (x, y, z) layout as the CPU arrays TmpOld/TmpNew, so they can be
! compared element-by-element below. Done as an explicit loop rather
! than RESHAPE(...,ORDER=...) so the index mapping is unambiguous to
! read.
! ---------------------------------------------------------------------------

allocate(xDenseAsXYZ(1:Nx1,1:Ny1,1:Nz1))
allocate(xThomasAsXYZ(1:Nx1,1:Ny1,1:Nz1))

! Flat GPU index (0-based) = Y + Z*Ny1 + X*Ny1*Nz1 - i.e. Y fastest,
! then Z, then X. Converted to 1-based Fortran indexing below.
Do iz = 1, Nz1
  Do ix = 1, Nx1
    Do iy = 1, Ny1
      xDenseAsXYZ(ix,iy,iz)  = x1dDense( iy + (iz-1)*Ny1 + (ix-1)*Ny1*Nz1 )
      xThomasAsXYZ(ix,iy,iz) = x1dThomas( iy + (iz-1)*Ny1 + (ix-1)*Ny1*Nz1 )
    End Do
  End Do
End Do

! ...........  Cross-validate CPU solvers against GPU solvers ..............
! TmpOld holds the CPU TPF (dense) result; TmpNew was last overwritten by
! CPU TPT1, so it's re-run here isn't needed - TmpOld is the stable
! reference for the dense comparison, and a fresh CPU EVD_Thomas call
! gives a stable reference for the Thomas comparison.

Write (*,*) ' Diff (GPU dense vs CPU TPF)  =', &
        &    maxval(abs( xDenseAsXYZ(1:Nx1,1:Ny1,1:Nz1) - TmpOld(1:Nx1,1:Ny1,1:Nz1) ))

Call  EVD_Thomas (TmpNew(1:Nx1,1:Ny1,1:Nz1), FDRHP(1:Nx1,1:Ny1,1:Nz1), &
        &                 EyTemp,Ey_invTemp,  EzTemp,Ez_invTemp, &
        &                 LambyTemp, LambzTemp, T_left, T_center, T_right, Nx1, Ny1, Nz1, 1.d0)

Write (*,*) ' Diff (GPU Thomas vs CPU TPT) =', &
        &    maxval(abs( xThomasAsXYZ(1:Nx1,1:Ny1,1:Nz1) - TmpNew(1:Nx1,1:Ny1,1:Nz1) ))

! ...........  Cleanup .....................................................

deallocate(xDenseAsXYZ)
deallocate(xThomasAsXYZ)

Call finalize_eigen_decomp_d()

Stop
End