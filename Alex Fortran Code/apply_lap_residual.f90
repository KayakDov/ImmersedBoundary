! ====================================================================
! apply_lap_residual.f90
!
! Provides: ComputeResidual(u, b, res_norm, Nx1, Ny1, Nz1)
!
! Computes res_norm = ||L*u - b|| where L is the 3D Laplacian:
!
!   L = Lx ⊗ Iy ⊗ Iz + Ix ⊗ Ly ⊗ Iz + Ix ⊗ Iy ⊗ Lz
!
! Each 1D operator is applied as a tridiagonal matvec using the
! coefficients T_left, T_center, T_right from the Thomas_coefficients
! module (populated by EVDLapTmpr).
!
! ASSUMPTION: uniform isotropic grid (Nx=Ny=Nz, same spacing in all
! directions), so Lx = Ly = Lz and a single set of tridiagonal
! coefficients covers all three directions.  For non-uniform or
! anisotropic grids, separate coefficient arrays would be needed.
! ====================================================================

      Subroutine ComputeResidual(u, b, res_norm, Nx1, Ny1, Nz1)

        Use Thomas_coefficients

        Implicit Real(kind=8) (A-H, O-Z)

        Integer,       Intent(in)  :: Nx1, Ny1, Nz1
        Real(kind=8),  Intent(in)  :: u(Nx1, Ny1, Nz1)
        Real(kind=8),  Intent(in)  :: b(Nx1, Ny1, Nz1)
        Real(kind=8),  Intent(out) :: res_norm

        Real(kind=8) :: Lu_val, acc

        acc = 0.d0

        Do k = 1, Nz1
          Do j = 1, Ny1
            Do i = 1, Nx1

              Lu_val = 0.d0

              ! --- Lx contribution: tridiagonal in i ---
              If (i > 1)   Lu_val = Lu_val + T_left(i)   * u(i-1, j, k)
                           Lu_val = Lu_val + T_center(i) * u(i,   j, k)
              If (i < Nx1) Lu_val = Lu_val + T_right(i)  * u(i+1, j, k)

              ! --- Ly contribution: tridiagonal in j ---
              ! (same coefficients as Lx for uniform isotropic grid)
              If (j > 1)   Lu_val = Lu_val + T_left(j)   * u(i, j-1, k)
                           Lu_val = Lu_val + T_center(j) * u(i, j,   k)
              If (j < Ny1) Lu_val = Lu_val + T_right(j)  * u(i, j+1, k)

              ! --- Lz contribution: tridiagonal in k ---
              If (k > 1)   Lu_val = Lu_val + T_left(k)   * u(i, j, k-1)
                           Lu_val = Lu_val + T_center(k) * u(i, j, k  )
              If (k < Nz1) Lu_val = Lu_val + T_right(k)  * u(i, j, k+1)

              acc = acc + (Lu_val - b(i, j, k))**2

            End Do
          End Do
        End Do

        res_norm = Sqrt(acc)

      End Subroutine ComputeResidual
