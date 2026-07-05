! ..... Distance between 2D matrices upper left squares ...

        Real(kind=8) Function Dist2D (A, B, N, M, K, N1, M1, K1)
        Implicit Real(kind=8) (A-H, O-Z)

        Real(kind=8), Dimension(0:N,0:M,0:K) ::   A, B, TST

        Parameter ( Epd = 0.1 )

! =======================================================

         TST(1:N1,1:M1,1:K1) = Abs(A(1:N1,1:M1,1:K1)-B(1:N1,1:M1,1:K1))

         Do i=1,N1
          Do j=1,M1
           Do L=1,K1
              If ( Abs(B(i,j,L)) > Epd) TST(i,j,L) = TST(i,j,L) / Abs( B(i,j,L) )
           End Do
          End Do
         End Do
           Dist2D =  MAXVAL( TST(1:N1,1:M1,1:K1) ) 

        Return
        End Function Dist2D


