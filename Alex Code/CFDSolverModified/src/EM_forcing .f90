! .......... Calculate electic potential .......................
!
!   This potential is defined in the points X, Yp, Z

Subroutine Get_Potential_Launch
    Use Grid
    Use Numbers
    Use Numerica
    Use Operators
    Use Variables
    Use AlexCudaCompatibility, only : PotentialHandle, Potential_pinned
    Use eigenbcgsolver_eigen_mod, only : solve_eigen_decomp_d

    Implicit Real(kind=8) (A-H,O-Z)
! ___________________________________________________

!$OMP Parallel Do Private(i,j,k,DVx_dz,Dvz_dx)
   Do i=1,Nx
     Do j=1,Ny1
      Do k=1,Nz
          DVx_dz = ( VMx(i,j,k+1) - VMx(i,j,k) ) / HPz(k)
          DVz_dx = ( VMz(i+1,j,k) - VMz(i,j,k) ) / HPx(i)

          FDRHP(i,j,k) = DVz_dx - DVx_dz
      End Do
     End Do
   End Do

        ! GPU: pure Poisson (shift = 0, alpha = 1 on all axes -- no RHS
        ! scaling needed, no reindexing needed), was
        ! Call EVDmethod(..., 1,1,1, beta=0).
        ! Launch only -- does NOT synch. Potential_pinned is not valid until
        ! Get_Potential_Finish (called right before EM_force, which is
        ! the first place Potential is actually read) calls synch_d.
        !
        ! No GPU_RHS_Pot/GPU_SOL_Pot scratch arrays anymore (nor the module
        ! that held them, Use'd above -- see AlexCudaCompatibility.f90):
        ! Potential_pinned is the solver's own pinned host buffer, aliased
        ! once at init time, used directly for both this RHS and the
        ! solution Get_Potential_Finish reads below. FDRHP's true shape is
        ! still ghost-padded (0:Nx2,0:Ny2,0:Nz2), so the slice
        ! FDRHP(1:Nx,1:Ny1,1:Nz) is still not contiguous -- writing it
        ! directly into Potential_pinned (an exact-shape buffer) keeps the
        ! call safe for the C interop layer, exactly as GPU_RHS_Pot did,
        ! just without the extra copy that used to sit between this array
        ! and the library's own pinned buffer.
        Potential_pinned = FDRHP(1:Nx,1:Ny1,1:Nz)
        Call solve_eigen_decomp_d(PotentialHandle)

  Return
End Subroutine Get_Potential_Launch


Subroutine Get_Potential_Finish
    Use Grid
    Use Numbers
    Use Numerica
    Use Operators
    Use Variables
    Use AlexCudaCompatibility, only : PotentialHandle, Potential_pinned
    Use eigenbcgsolver_eigen_mod, only : synch_d

    Implicit Real(kind=8) (A-H,O-Z)

    ! Loop counters and the full declared bounds of Potential (0:Nxx1,
    ! 0:Nyy2,0:Nzz1 -- see the Allocate in ConvMain_3D_Q2D.f90). Pulled via
    ! LBound/UBound rather than hardcoded so every statement below touches
    ! exactly the same elements the original bare-colon ("Potential(:,:,:)")
    ! syntax did -- this is a parallelization/temp-array fix only, not a
    ! change to which cells get written.
    Integer :: i, j, k, iLo, iHi, jLo, jHi, kLo, kHi
    Real(kind=8) :: Pot000
! ___________________________________________________

        ! synch_d no longer takes an output argument: it just waits for the
        ! GPU work to finish. The solution is then read directly from
        ! Potential_pinned below -- same reasoning as Get_Potential_Launch,
        ! no separate GPU_SOL_Pot needed anymore.
        Call synch_d(PotentialHandle)

        ! Padded<->unpadded remap -- still needed, same as before (Potential's
        ! true declared shape (0:Nxx1,0:Nyy2,0:Nzz1) doesn't match this
        ! solve's region (1:Nx,1:Ny1,1:Nz), so this can't be a bare
        ! whole-array assignment without either being wrong or forcing a
        ! hidden compiler temporary). Just reads from Potential_pinned now
        ! instead of GPU_SOL_Pot.
!$OMP Parallel Do Private(i,j,k)
    Do i = 1, Nx
      Do j = 1, Ny1
        Do k = 1, Nz
          Potential(i,j,k) = Potential_pinned(i,j,k)
        End Do
      End Do
    End Do

        ! Was: Potential = Potential - Potential(1,1,1)
        ! Bare colons here default to Potential's FULL declared bounds
        ! (0:Nxx1,0:Nyy2,0:Nzz1), not just the active (Nx,Ny1,Nz) region --
        ! so this one statement was walking the entire preallocated buffer.
        ! Fortran's array-assignment semantics guarantee the whole RHS
        ! (including Potential(1,1,1)) is evaluated using the pre-assignment
        ! values before anything is written, so the scalar is captured once,
        ! up front, to reproduce that exactly under explicit looping/OpenMP.
        Pot000 = Potential(1,1,1)
        iLo = LBound(Potential,1);  iHi = UBound(Potential,1)
        jLo = LBound(Potential,2);  jHi = UBound(Potential,2)
        kLo = LBound(Potential,3);  kHi = UBound(Potential,3)

!$OMP Parallel Do Private(i,j,k)
    Do i = iLo, iHi
      Do j = jLo, jHi
        Do k = kLo, kHi
          Potential(i,j,k) = Potential(i,j,k) - Pot000
        End Do
      End Do
    End Do

    If(EVD_Pot_X == 1 ) then
        ! Was: Potential(0,:,:) = Potential(1,:,:);  Potential(Nx1,:,:) = Potential(Nx,:,:)
!$OMP Parallel Do Private(j,k)
        Do j = jLo, jHi
          Do k = kLo, kHi
             Potential(0,j,k)   = Potential(1,j,k)
             Potential(Nx1,j,k) = Potential(Nx,j,k)
          End Do
        End Do
     else
        ! Was: Potential(0,:,:) = 0.d0;  Potential(Nx1,:,:) = 0.d0
!$OMP Parallel Do Private(j,k)
        Do j = jLo, jHi
          Do k = kLo, kHi
             Potential(0,j,k)   = 0.d0
             Potential(Nx1,j,k) = 0.d0
          End Do
        End Do
    End If

     If(EVD_Pot_Y == 1 ) then
        ! Was: Potential(:,0,:) = Potential(:,1,:);  Potential(:,Ny2,:) = Potential(:,Ny1,:)
!$OMP Parallel Do Private(i,k)
       Do i = iLo, iHi
         Do k = kLo, kHi
            Potential(i,0,k)   = Potential(i,1,k)
            Potential(i,Ny2,k) = Potential(i,Ny1,k)
         End Do
       End Do
     else
        ! Was: Potential(:,0,:) = 0.d0;  Potential(:,Ny2,:) = 0.d0
!$OMP Parallel Do Private(i,k)
        Do i = iLo, iHi
          Do k = kLo, kHi
             Potential(i,0,k)   = 0.d0
             Potential(i,Ny2,k) = 0.d0
          End Do
        End Do
    End If

     If(EVD_Pot_Z == 1 ) then
        ! Was: Potential(:,:,0) = Potential(:,:,1);  Potential(:,:,Nz1) = Potential(:,:,Nz)
!$OMP Parallel Do Private(i,j)
        Do i = iLo, iHi
          Do j = jLo, jHi
             Potential(i,j,0)   = Potential(i,j,1)
             Potential(i,j,Nz1) = Potential(i,j,Nz)
          End Do
        End Do
     else
        ! Was: Potential(:,:,0) = 0.d0;  Potential(:,:,Nz1) = 0.d0
!$OMP Parallel Do Private(i,j)
        Do i = iLo, iHi
          Do j = jLo, jHi
             Potential(i,j,0)   = 0.d0
             Potential(i,j,Nz1) = 0.d0
          End Do
        End Do
    End If

 !   Write (*,*) ' EM: Potential=', Sum(Potential)
  Return
End Subroutine Get_Potential_Finish

! .......... Calculate electromagnetic force .......................

Subroutine EM_force
    Use Numbers
    Use Parameters
    Use Grid
    Use Operators
    Use Variables

    Implicit Real(kind=8) (A-H,O-Z)
! ___________________________________________________

         Coef = DGr * (Hartmann * WidRa)**2

!$OMP Parallel Do Private(i,j,k,DFi_dz)
   Do i=1,Nx
     Do j=1,Ny1
      Do k=1,Nz1
          DFi_dz = ( Potential(i,j,k) - Potential(i,j,k-1) ) / Hz12(k-1)

          RHSx(i,j,k) =  RHSx(i,j,k) + Coef * ( DFi_dz + VMx(i,j,k) )
      End Do
     End Do
   End Do
 !  Write (*,*) ' EM: RHSx=', Sum(RHSx)

!$OMP Parallel Do Private(i,j,k,DFi_dx)
   Do i=1,Nx1
     Do j=1,Ny1
      Do k=1,Nz
          DFi_dx = ( Potential(i,j,k) - Potential(i-1,j,k) ) / Hx12(i-1)

         RHSz(i,j,k) = RHSz(i,j,k) - Coef * ( DFi_dx -VMz(i,j,k) )
      End Do
     End Do
    End Do
!   Write (*,*) ' EM: RHSz=', Sum(RHSz)

! ........... Make div-free force .........................

!        Call EVDbounds_V(Work_flow)
!        Call Make_divfree(Stream, 0)

!        Work_flow%P = 0.d0

End Subroutine EM_force
