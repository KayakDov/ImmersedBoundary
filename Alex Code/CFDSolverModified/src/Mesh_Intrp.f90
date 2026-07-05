! ===================================================
!        InteXpolation from grid to grid 
! ===================================================

      Parameter( Nxx=300, Nyy=1000, Nzz=200)
      Parameter( Nxx1=Nxx+1, Nxx2=Nxx+2, Nzz1=Nzz+1, Nzz2=Nzz+2)
      Parameter( Nyy1=Nyy+1, Nyy2=Nyy+2)

      Implicit Real(kind=8) (a-h,o-z)

      Real(kind=8), Dimension(0:Nxx1) :: X,  Xold
      Real(kind=8), Dimension(0:Nxx2) :: Xp, Xpold
      Real(kind=8), Dimension(0:Nyy1) :: Y,  Yold
      Real(kind=8), Dimension(0:Nyy2) :: Yp, Ypold
      Real(kind=8), Dimension(0:Nzz1) :: Z,  Zold
      Real(kind=8), Dimension(0:Nzz2) :: Zp, Zpold

      Real(kind=8), Dimension(0:Nxx1,1:Nyy1,1:Nzz1) :: P, Pold
      Real(kind=8), Dimension(0:Nxx1,0:Nyy2,0:Nzz2) :: U, Uold
      Real(kind=8), Dimension(0:Nxx2,0:Nyy1,0:Nzz2) :: V, Vold
      Real(kind=8), Dimension(0:Nxx2,0:Nyy2,0:Nzz1) :: W, Wold
      Real(kind=8), Dimension(0:Nxx2,0:Nyy2,0:Nzz2) :: T, Told

! ===============================================================

       Open(1,  file='fromto.dat', form='formatted', status='old')
       Open(2,  file='Conv.dat', form='formatted', status='old')
	Open(22, file='BasOld.ddd', form='unformatted',  status='old')
	Open(33, file='BasNew.ddd', form='unformatted',  status='unknown')

        P = 0.d0;  Pold = 0.D0
        U = 0.d0;  Uold = 0.D0
        V = 0.d0;  Vold = 0.D0
        W = 0.d0;  Wold = 0.D0
        T = 0.d0;  Told = 0.D0

! ********* Input ************************

        Read (1,100) NxOld
        Read (1,100) NyOld
        Read (1,100) NzOld
        Read (1,100) NxNew
        Read (1,100) NyNew
        Read (1,100) NzNew

	  Write (*,*) ' NxOld=', NxOld, ' NyOld=', NyOld, '  NzOld=', NzOld
	  Write (*,*) ' NxNew=', NxNew, ' NyNew=', NyNew, '  NzNew=', NzNew

        Read (2,101) AspRa, WidRa

        Write (*,*) 'AspRa =', AspRa, 'WidRa =', AspRa, Nz1, Nz2

! ********* Old flow ***********************************

        Nx = NxOld;   Ny = NyOld;  Nz = NzOld

        Nx1 = Nx + 1;  Nx2 = Nx + 2
        Ny1 = Ny + 1;  Ny2 = Ny + 2
        Nz1 = Nz + 1;  Nz2 = Nz + 2
        
        Write (*,*) ' Nz=', Nz, Nz1, Nz2

        Call Mesh
        Write (*,*) '1 Mesh exited'

	   XOld(0:Nx1) =  X(0:Nx1);   XpOld(0:Nx2) = Xp(0:Nx2)
	   YOld(0:Ny1) =  Y(0:Ny1);   YpOld(0:Ny2) = Yp(0:Ny2)
	   ZOld(0:Nz1) =  Z(0:Nz1);   ZpOld(0:Nz2) = Zp(0:Nz2)

          Read (22) (((  Uold(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
          Read (22) (((  Vold(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
          Read (22) (((  Wold(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
          Read (22) (((  Told(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)
          Read (22) (((  Pold(i,j,k), i=1,Nx1), j=1,Ny1), k=1,Nz1)

! ********* New grid ***************************

        Nx = NxNew;  Ny = NyNew;   Nz = NzNew

        Nx1 = Nx + 1;    Nx2 = Nx + 2
        Ny1 = Ny + 1;    Ny2 = Ny + 2
        Nz1 = Nz + 1;    Nz2 = Nz + 2
        
        Call Mesh

        Write (*,*) '2 Mesh exited'
! ******** Form new fields **********************

       Do i=0,Nx2

        If(i <= Nx1) then
           ii = 0
           Do While(( Xp(i)-XpOld(ii) ) * ( Xp(i)-XpOld(ii+1) ) > 0.D0 )
	       ii = ii+1
           End Do
           
         Else
            ii=NxOld+1
        End If
	 
             xx  = Xp(i)
             xx0 = XpOld( ii )
	      xx1 = XpOld(ii+1)
 
	  Do j=0,Ny2

        If(j <= Ny1) then
           jj = 0
           Do While(( Yp(j)-YpOld(jj) ) * ( Yp(j)-YpOld(jj+1) ) > 0.D0 )
	       jj = jj+1
           End Do
           
         Else
            jj=NyOld+1
        End If
	
	       yy = yp(j)
	      
	       yy0 = ypOld(jj)
	       yy1 = ypOld(jj+1)

	    Do k=0,Nz2

        If(k <= Nz1) then
              kk = 0
              Do While(( Zp(k)-ZpOld(kk) ) * ( Zp(k)-ZpOld(kk+1) ) > 0.D0 )
	          kk = kk+1
              End Do
            
         Else
            kk=NzOld+1
        End If
             
	       zz  = Zp(k)	      
	       zz0 = ZpOld(kk)
	       zz1 = ZpOld(kk+1)

	       f000 = TOld(ii  ,jj  ,kk  )
	       f010 = TOld(ii  ,jj+1,kk  )
	       f100 = TOld(ii+1,jj  ,kk  ) 
	       f110 = TOld(ii+1,jj+1,kk  )

	       f001 = TOld(ii  ,jj  ,kk+1)
	       f011 = TOld(ii  ,jj+1,kk+1)
	       f101 = TOld(ii+1,jj  ,kk+1) 
	       f111 = TOld(ii+1,jj+1,kk+1)

	       T(i,j,k) = Fint4()
	   End Do
	  End Do
	 End Do
       Write (*,*) ' T=', MaxVal(T), MinVal(T)

       Do i=1,Nx1

	   If( i < Nx1 ) then
           ii = 1
           Do While(( Xp(i)-XpOld(ii) ) * ( Xp(i)-XpOld(ii+1) ) > 0.D0 )
	       ii = ii+1
           End Do

	    Else
                ii = NxOld
	   End If
	 
             xx  = Xp(i)
             xx0 = XpOld( ii )
             xx1 = XpOld(ii+1)
 
	  Do j=1,Ny1

	   If( j < Ny1 ) then
             jj = 1
             Do While(( Yp(j)-YpOld(jj) ) * ( Yp(j)-YpOld(jj+1) ) > 0.D0 )
	       jj = jj+1
             End Do

	    Else
                jj = NyOld
	   End If

	       yy = Yp(j)
	       yy0 = YpOld(jj)
	       yy1 = YpOld(jj+1)

	   Do k=1,Nz1
	         If( k < Nz1 ) then
                   kk = 1
                   Do While(( Zp(k)-ZpOld(kk) ) * ( Zp(k)-ZpOld(kk+1) ) > 0.D0 )
	           kk = kk+1
                  End Do

	          Else
                  kk = NzOld
              End If
              
	       zz  = Zp(k)
	       zz0 = ZpOld(kk)
	       zz1 = ZpOld(kk+1)

	       f000 = POld(ii  ,jj  ,kk  )
	       f010 = POld(ii  ,jj+1,kk  )
	       f100 = POld(ii+1,jj  ,kk  ) 
	       f110 = POld(ii+1,jj+1,kk  )
	       f001 = POld(ii  ,jj  ,kk+1)
	       f011 = POld(ii  ,jj+1,kk+1)
	       f101 = POld(ii+1,jj  ,kk+1) 
	       f111 = POld(ii+1,jj+1,kk+1)

	       P(i,j,k) = Fint4()

	        If( Abs( P(i,j,k) ) .LT. 1.D-10) P(i,j,k) = 0.D0
	   End Do
  	  End Do
	 End Do
       Write (*,*) ' Pmin=', Minval(Pold), MinVal(P), Sum(P)
       Write (*,*) ' Pmax=', Maxval(Pold), MaxVal(P)

       Do i=0,Nx1
 
         If( i < Nx1 ) then
           ii = 0
           Do While(( X(i)-XOld(ii) ) * ( X(i)-XOld(ii+1) ) > 0.D0 )
	       ii = ii+1
           End Do
	   Else
	      ii = NxOld
	   End If

            xx = X(i)
            xx0 = XOld( ii )
            xx1 = XOld(ii+1)

	  Do j=0,Ny2

          If( j < Ny2 ) then
          jj = 0
           Do While(( Yp(j)-YpOld(jj) ) * ( Yp(j)-YpOld(jj+1) ) > 0.D0 )
	       jj = jj+1
           End Do
	   Else
	      jj = NyOld+1
	   End If
     
	       yy  = Yp(j)
	       yy0 = YpOld(jj)
	       yy1 = YpOld(jj+1)

	   Do k=0,Nz2

          If( k < Nz2 ) then
          kk = 0
           Do While(( Zp(k)-ZpOld(kk) ) * ( Zp(k)-ZpOld(kk+1) ) > 0.D0 )
	       kk = kk+1
           End Do
	   Else
	      kk = NzOld+1
	   End If
	
	       zz  = Zp(k)
	       zz0 = ZpOld(kk)
	       zz1 = ZpOld(kk+1)

	       f000 = UOld(ii  ,jj  ,kk  )
	       f010 = UOld(ii  ,jj+1,kk  )
	       f100 = UOld(ii+1,jj  ,kk  ) 
	       f110 = UOld(ii+1,jj+1,kk  )
	       f001 = UOld(ii  ,jj  ,kk+1)
	       f011 = UOld(ii  ,jj+1,kk+1)
	       f101 = UOld(ii+1,jj  ,kk+1) 
	       f111 = UOld(ii+1,jj+1,kk+1)

	       U(i,j,k) = Fint4()

          End Do
         End Do
        End Do
       Write (*,*) ' Umin=', Minval(Uold), MinVal(U)
       Write (*,*) ' Umax=', Maxval(Uold), MaxVal(U)

       Do i=0,Nx2

         If( i < Nx2 ) then
           ii = 0
           Do While(( Xp(i)-XpOld(ii) ) * ( Xp(i)-XpOld(ii+1) ) > 0.D0 )
	       ii = ii+1
           End Do
	     Else
	        ii = NxOld+1
	    End If
	 
	     xx  = Xp(i)
            xx0 = XpOld( ii )
	     xx1 = XpOld(ii+1)

	 Do j=0,Ny1

            If( j < Ny1 ) then
                 jj = 0
              Do While(( Y(j)-YOld(jj) ) * ( Y(j)-YOld(jj+1) ) > 0.D0 )
	         jj = jj+1
              End Do
	     Else
	        jj = NyOld
	    End If

	       yy  = Y(j)
	       yy0 = YOld(jj)
	       yy1 = YOld(jj+1)

  	    Do k=0,Nz2

         If( k < Nz2) then
             kk = 0
             Do While(( Zp(k)-ZpOld(kk) ) * ( Zp(k)-ZpOld(kk+1) ) > 0.D0 )
	       kk = kk+1
             End Do
	     Else
	        kk = NzOld+1
	    End If
	
	       zz  = Zp(k)
	       zz0 = ZpOld(kk)
	       zz1 = ZpOld(kk+1)

	       f000 = VOld(ii  ,jj  ,kk  )
	       f010 = VOld(ii  ,jj+1,kk  )
	       f100 = VOld(ii+1,jj  ,kk  ) 
	       f110 = VOld(ii+1,jj+1,kk  )
	       f001 = VOld(ii  ,jj  ,kk+1)
	       f011 = VOld(ii  ,jj+1,kk+1)
	       f101 = VOld(ii+1,jj  ,kk+1) 
	       f111 = VOld(ii+1,jj+1,kk+1)

	       V(i,j,k) = Fint4()
          End Do
         End Do
        End Do
       Write (*,*) ' Vmin=', Minval(Vold), MinVal(V)
       Write (*,*) ' Vmax=', Maxval(Vold), MaxVal(V)

       Do i=0,Nx2

        If( i < Nx2 ) then
          ii = 0
           Do While(( Xp(i)-XpOld(ii) ) * ( Xp(i)-XpOld(ii+1) ) > 0.D0 )
	       ii = ii+1
           End Do
        Else
         ii = NxOld+1
        End If

	 
	     xx  = Xp(i)
            xx0 = XpOld( ii )
	     xx1 = XpOld(ii+1)

	  Do j=0,Ny2

        If(j < Ny2 ) then
           jj = 0
           Do While(( Yp(j)-YpOld(jj) ) * ( Yp(j)-YpOld(jj+1) ) > 0.D0 )
	       jj = jj+1
           End Do
        Else
	         jj = NyOld+1
        End If

	 
	     yy  = Yp(j)
	     yy0 = YpOld( jj )
	     yy1 = YpOld(jj+1)

	   Do k=0,Nz1

               If( k < Nz1 ) then
                    kk = 0
                 Do While(( Z(k)-ZOld(kk) ) * ( Z(k)-ZOld(kk+1) ) > 0.D0 )
	            kk = kk+1
                 End Do
	        Else
	         kk = NzOld
	       End If

	       zz  = Z(k)
	       zz0 = ZOld(kk)
	       zz1 = ZOld(kk+1)

	       f000 = WOld(ii  ,jj  ,kk  )
	       f010 = WOld(ii  ,jj+1,kk  )
	       f100 = WOld(ii+1,jj  ,kk  ) 
	       f110 = WOld(ii+1,jj+1,kk  )
	       f001 = WOld(ii  ,jj  ,kk+1)
	       f011 = WOld(ii  ,jj+1,kk+1)
	       f101 = WOld(ii+1,jj  ,kk+1) 
	       f111 = WOld(ii+1,jj+1,kk+1)

	       W(i,j,k) = Fint4()

          End Do
         End Do
        End Do
       Write (*,*) ' Wmin=', Minval(Wold), MinVal(W)
       Write (*,*) ' Wmax=', Maxval(Wold), MaxVal(W)

! ........... Write new fields .........................................
            Tstart = 0.d0

          Write (33) (((  U(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
          Write (33) (((  V(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
          Write (33) (((  W(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
          Write (33) (((  T(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)
          Write (33) (((  P(i,j,k), i=1,Nx1), j=1,Ny1), k=1,Nz1)

          Write (33) (((  U(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
          Write (33) (((  V(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
          Write (33) (((  W(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
          Write (33) (((  T(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)
          Write (33) Tstart
      Stop
100    Format(I4)
101    Format(G15.8)

      Contains
! .............. InteXpolation between 8 points .....................

      Double Precision Function Fint4()
         
       Implicit Real(kind=8)(a-h,o-z)
      
! =============================================         

         Fint4 = ( f000 * (xx1 - xx ) * (yy1 - yy ) * (zz1 - zz )  +  &
     &             f010 * (xx1 - xx ) * (yy  - yy0) * (zz1 - zz )  +  &
     &             f100 * (xx  - xx0) * (yy1 - yy ) * (zz1 - zz )  +  &
     &             f110 * (xx  - xx0) * (yy  - yy0) * (zz1 - zz )  +  & 
     &             f001 * (xx1 - xx ) * (yy1 - yy ) * (zz  - zz0)  +  &
     &             f011 * (xx1 - xx ) * (yy  - yy0) * (zz  - zz0)  +  &
     &             f101 * (xx  - xx0) * (yy1 - yy ) * (zz  - zz0)  +  &
     &             f111 * (xx  - xx0) * (yy  - yy0) * (zz  - zz0)     & 
     &                                                             ) /&
     &               ( (xx1-xx0) * (yy1-yy0) * (zz1-zz0) )
         
      Return
      End Function Fint4

! .............. Stretched grid .....................

      Subroutine    Mesh

        Implicit Real(kind=8) (A-H,O-Z), Integer (I-N)

! ==========================================================

        Pi = 4.D0 * Atan(1.D0)
!         a = 0.12D0
!         b = 0.12D0
!         c = 0.12D0

          s = 13.d0

         delta_x = s
         delta_y = s
         delta_z = s
  
! ######## Forming X mesh ########################

        h    = 1.D0 / Nx1

        X(0) = 0.D0

        Do i=1,Nx1
            X(i) = X(i-1) + h
        End Do

     !       X = X - a * Sin( 2.D0 * Pi * X )
           X = 0.5d0 + 0.5d0 * tanh( delta_x * ( X-0.5d0) ) / tanh( 0.5d0 * delta_x )
           X = X * AspRa

!   ........... Forming i+1/2 points ...........

            Xp(0)   = 0.D0
            Xp(Nx2) = AspRa 

        Do i=1,Nx1
            Xp(i) = ( X(i) + X(i-1) ) / 2.D0
        End Do

! ######### Forming Y mesh #################

        H    = 1.D0 / Ny1
        Y(0) = 0.D0

        Do i=1,Ny1
            Y(i) = Y(i-1) + H
        End Do

 !           Y = Y - b * Sin( 2.D0 * Pi * Y )
          Y = 0.5d0 + 0.5d0 * tanh( delta_y * ( Y-0.5d0) ) / tanh( 0.5d0 * delta_y )
             Y = Y * WidRa

!   ........... Forming i+1/2 points ...........

        Yp(0)   = 0.D0
        Yp(Ny2) = WidRa 

        Do i=1,Ny1
            Yp(i)  = 0.5D0 * ( Y(i) + Y(i-1) )
        End Do

! ######### Forming Z mesh #################

        h    = 1.D0 / Nz1
        Z(0) = 0.D0

        Do i=1,Nz1
            Z(i) = Z(i-1) + h
        End Do

 !           Z = Z - c * Sin( 2.D0 * Pi * Z )
            Z = 0.5d0 + 0.5d0 * tanh( delta_z * ( Z-0.5d0) ) / tanh( 0.5d0 * delta_z )

!   ........... Forming i+1/2 points ...........

        Zp(0)   = 0.D0
        Zp(Nz2) = 1.D0 

        Do i=1,Nz1
            Zp(i)  = 0.5D0 * ( Z(i) + Z(i-1) )
        End Do

        Return
        End Subroutine    Mesh

        End
