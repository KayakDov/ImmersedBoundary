 
   Subroutine Outp ( lpr )
         
         Use Numbers
         Use Grid
         Use variables
         Use Numerica
         Use Ibmethod
         Use MatrixFormAndOperate

         Implicit none

	     Integer lpr,nbd
	     Real*8 Vtx, Vty, Vtz

         Character*50  Head


       
! ##########  Saving the current result ###################

        Rewind 3

        Write (3) ((( VMx(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
        Write (3) ((( VMy(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
        Write (3) ((( VMz(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
        Write (3) ((( Prs(i,j,k), i=1,Nx1), j=1,Ny1), k=1,Nz1)

        Write (3) ((( VMxOld(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
        Write (3) ((( VMyOld(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
        Write (3) ((( VMzOld(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
       
      
      DO nbd=1,  n_body
             Write (3) ( bdy(nbd)%fb_x(i), i=1,bdy(nbd)%npts)
             Write (3) ( bdy(nbd)%fb_y(i), i=1,bdy(nbd)%npts)
             Write (3) ( bdy(nbd)%fb_z(i), i=1,bdy(nbd)%npts)
      End Do
      
    
      
       If (lpr==4)then 
           
        rewind(777)
        Write (777,  *) 'VARIABLES = "X","Y","Z","Vx","Vy","Vz","P"'
        Write (777,301) Nx1,Ny1,Nz1
       
        Do m=1, Nz1
          Do j=1,Ny1
            Do i=1,Nx1
                Vtx=(VmxNew(i,j,m)  *(X12(i)-X(i-1))+&
                    VmxNew(i-1,j,m) *(X(i)  -X12(i)))/Hx12(i-1)
  
                Vty=(VmyNew(i,j,m)  *(Y12(j)-Y(j-1))+&
                     VmyNew(i,j-1,m)*(Y(j)-Y12(j)))/Hy12(j-1)
            
                Vtz=(VmzNew(i,j,m)  *(Z12(m)-Z(m-1))+&
                     VmzNew(i,j,m-1)*(Z(m)-Z12(m)))/Hz12(m-1)
               
                Write (777,310) X12(i),Y12(j),Z12(m),Vtx,Vty,Vtz,Prs(i,j,m) 
              End Do
            End Do
         End Do   
        
       End If  

        Write (*,200)

        Return
200        Format (' Outp: current results are written')
301     Format(' ZONE I=',I4, ', J=',I4,', K=',I4,'  DATAPACKING=POINT')
310     Format ((7E15.5,2x) )   
        End

