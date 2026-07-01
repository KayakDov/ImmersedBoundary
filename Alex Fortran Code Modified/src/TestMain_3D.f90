! ************************************************************
! *  Main program for unsteady convection                    *
! *            in rectangular cavities                       *
! *                                                          *
! *  GPU version: EVDmethod/EVD_Thomas replaced by            *
! *  CudaBandedLib's init/solve/finalize_eigen_decomp_d,      *
! *  using a dense solver handle (TPF-equivalent) and a       *
! *  Thomas/tridiagonal solver handle (TPT-equivalent).       *
! ************************************************************

Use Numbers
Use Grid
Use Variables
Use EVD_Operators
use Thomas_coefficients
use eigenbcgsolver_eigen_mod
use iso_c_binding, only : C_SIZE_T

Implicit Real(kind=8) (A-H,O-Z)

Real(kind=4), Dimension(2) :: Time
Real(kind=4)               :: t0, t1

integer(C_SIZE_T) :: denseSolverHandle
integer(C_SIZE_T) :: thomasSolverHandle

! ----------------------------------------------------------------------
! Buffers in (y, x, z) order, matching init_eigen_decomp_d's argument order
! ----------------------------------------------------------------------
real(kind=8), allocatable :: x1dDense(:)
real(kind=8), allocatable :: x1dThomas(:)
real(kind=8), allocatable :: rhs1d(:)

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

! .................. Make eigenvalue decompositions (GPU) .................
! denseSolverHandle  <-> old EVDLapTmpr + EVDmethod  (TPF)
! thomasSolverHandle <-> old EVDLapTmpr + EVD_Thomas (TPT)

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

! ...........  Run TPF (dense solve) ......................................

t0 = etime(Time);  t0 = Time(1)

x1dDense = -9999.0d0
call solve_eigen_decomp_d( &
        denseSolverHandle, &
        x1dDense, &
        rhs1d)

t1 = etime(Time); t1 = Time(1)
Write (*,*) '  TPF worked ',t1-t0, ' secs', '   Tmp=',Sum(abs(x1dDense))

! ...........  Run TPT (Thomas solve) .....................................

t0 = etime(Time);  t0 = Time(1)

x1dThomas = -9999.0d0
call solve_eigen_decomp_d( &
        thomasSolverHandle, &
        x1dThomas, &
        rhs1d)

t1 = etime(Time); t1 = Time(1)
Write (*,*) '  TPT worked ',t1-t0, ' secs', '   Tmp=',Sum(abs(x1dThomas))
Write (*,*) ' Diff=', maxval(abs( x1dThomas - x1dDense ))

! ...........  Cleanup .....................................................

Call finalize_eigen_decomp_d()

Stop
End