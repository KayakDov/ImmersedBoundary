! ..... Distance between 2D matrices upper left squares ...

        Real(kind=8) Function Dist2D (A, B, N, M, K, N1, M1, K1)
        Implicit Real(kind=8) (A-H, O-Z)

        Real(kind=8), Dimension(0:N,0:M,0:K) ::   A, B

        Parameter ( Epd = 0.1 )

        ! Was: a local automatic array TST(0:N,0:M,0:K) -- since N,M,K are
        ! dummy arguments (not compile-time constants), TST's shape is only
        ! known at runtime, so the compiler must allocate it fresh (heap,
        ! for anything beyond a small stack threshold) on every single call.
        ! This function is called 4x per timestep (RTmpr/RNSx/RNSy/RNSz in
        ! time_step_Q2D.f90), so that was 4 fresh allocate/deallocate pairs
        ! -- and the page faults that come with touching brand-new pages --
        ! per timestep, on top of walking the data twice (once to fill TST,
        ! once more inside MAXVAL). Below, both passes are fused into one
        ! loop that tracks a running max directly, so TST is never needed at
        ! all, and the loop picks up OMP parallelism (dist2d_ had none
        ! before) for free. MAXVAL is order-independent -- unlike a sum,
        ! there is no floating-point rounding difference from computing the
        ! max in a different order, so this is bit-for-bit the same result.
        Integer :: i, j, L
        Real(kind=8) :: Diff, DMax

        DMax = 0.d0
!$OMP Parallel Do Private(i,j,L,Diff) Reduction(max:DMax)
        Do i=1,N1
          Do j=1,M1
           Do L=1,K1
              Diff = Abs( A(i,j,L) - B(i,j,L) )
              If ( Abs(B(i,j,L)) > Epd) Diff = Diff / Abs( B(i,j,L) )
              DMax = Max( DMax, Diff )
           End Do
          End Do
         End Do
           Dist2D = DMax

        Return
        End Function Dist2D

