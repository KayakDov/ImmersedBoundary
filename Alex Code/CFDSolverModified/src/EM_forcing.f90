! .......... Calculate electic potential .......................
!
!   This potential is defined in the points X, Yp, Z

! Shared between Get_Potential_Launch and Get_Potential_Finish: a plain local
! Save variable in each subroutine would NOT be the same storage (Fortran
! locals are private per-subroutine even with Save), so these have to live
! in a module both subroutines Use.
Module GPU_Potential_Scratch
    Real(kind=8), Allocatable, Save :: GPU_RHS_Pot(:,:,:), GPU_SOL_Pot(:,:,:)
    Logical, Save :: GPU_Pot_Allocated = .false.
End Module GPU_Potential_Scratch

Subroutine Get_Potential_Launch
    Use Grid
    Use Numbers
    Use Numerica
    Use Operators
    Use Variables
    Use AlexCudaCompatibility, only : PotentialHandle
    Use eigenbcgsolver_eigen_mod, only : solve_eigen_decomp_d
    Use GPU_Potential_Scratch

    Implicit Real(kind=8) (A-H,O-Z)
! ___________________________________________________

    If (.not. GPU_Pot_Allocated) Then
        Allocate( GPU_RHS_Pot(1:Nx,1:Ny1,1:Nz) )
        Allocate( GPU_SOL_Pot(1:Nx,1:Ny1,1:Nz) )
        GPU_Pot_Allocated = .true.
    End If

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
        ! Launch only -- does NOT synch. GPU_SOL_Pot is not valid until
        ! Get_Potential_Finish (called right before EM_force, which is
        ! the first place Potential is actually read) calls synch_d.
        ! solve_eigen_decomp_d no longer takes x -- the result is retrieved by
        ! synch_d instead (see Get_Potential_Finish). GPU_RHS_Pot is still
        ! required here even though this is a pure copy (no scaling): FDRHP's
        ! true shape is ghost-padded (0:Nx2,0:Ny2,0:Nz2), so the slice
        ! FDRHP(1:Nx,1:Ny1,1:Nz) is not contiguous, and this exact-shape
        ! buffer keeps the call safe for the C interop layer.
        GPU_RHS_Pot = FDRHP(1:Nx,1:Ny1,1:Nz)
        Call solve_eigen_decomp_d(PotentialHandle, GPU_RHS_Pot)

  Return
End Subroutine Get_Potential_Launch


Subroutine Get_Potential_Finish
    Use Grid
    Use Numbers
    Use Numerica
    Use Operators
    Use Variables
    Use AlexCudaCompatibility, only : PotentialHandle
    Use eigenbcgsolver_eigen_mod, only : synch_d
    Use GPU_Potential_Scratch

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

        ! synch_d now retrieves the result directly (was: bare wait, then a
        ! separate Fortran array copy from GPU_SOL_Pot). GPU_SOL_Pot is kept
        ! here (not eliminated) because Potential's true declared shape
        ! (0:Nxx1,0:Nyy2,0:Nzz1) does not match this solve's region
        ! (1:Nx,1:Ny1,1:Nz) -- passing Potential's slice directly would be
        ! non-contiguous and force a hidden, non-pinned compiler temporary.
        Call synch_d(PotentialHandle, GPU_SOL_Pot)

        ! Was: Potential(1:Nx,1:Ny1,1:Nz) = GPU_SOL_Pot
        ! An explicit-shape whole-array assignment like that is exactly the
        ! kind of statement ifx tends to route through a compiler-generated
        ! temporary when the LHS is a module variable (it can't fully rule
        ! out aliasing across the procedure boundary), which shows up at
        ! runtime as a fresh Allocate/Deallocate (and the page faults that
        ! come with touching brand-new pages) on every single timestep. An
        ! explicit Do loop assigns element-by-element with no array
        ! temporary possible, and doubles as free OpenMP parallelism.
!$OMP Parallel Do Private(i,j,k)
    Do i = 1, Nx
      Do j = 1, Ny1
        Do k = 1, Nz
          Potential(i,j,k) = GPU_SOL_Pot(i,j,k)
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
