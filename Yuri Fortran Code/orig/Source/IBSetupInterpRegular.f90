MODULE IBsetupInetrpRegular
 USE Grid
 USE Variables
 Use Numbers
 Use Numerica
 Use Operators
 Use Ibmethod 
 Use Parameters
 Use EVD_Operators
 Use Thomas_coefficients
 Use EvdProcedures
 Use Matrices
 !Use ISO_C_BINDING 
 IMPLICIT NONE
 INTEGER :: ijsmear, proba


 INTEGER :: i,j,k,m,n,o, counter, tmp
 
 
 REAL(KIND(0.D0)), DIMENSION(1:1000):: RegValues
 INTEGER, DIMENSION(1:1000):: I_Index,J_Index,K_Index
 
 CONTAINS
  
    
    
SUBROUTINE setup_geometry
 
    LOGICAL :: readinput
    INTEGER :: i,i_bdy,i_act, next, stat, N_interp_points, i_interp_point
    INTEGER*8 :: n_entries_x, n_entries_y, n_entries_z
    CHARACTER(3) :: file_num

    
    d2X=0.5d0*Hpx(2)
    d2Y=0.5d0*Hpy(2)
    d2Z=0.5d0*Hpz(2)
    supx= support*d2X
    supy= support*d2Y 
    supz= support*d2Z
    
       
    ! look for bodies in input directory
    readinput = .TRUE.
    n_body = 0
    
     Open (2022, file='interppoint.inp', form='formatted', iostat=stat, status='old')
    
     IF (stat .NE. 0) THEN
        WRITE(*,*) ' Setup:  Attempt to read set of points for interpolation was not successfull, no interpolation will be performed' 
        N_interp_points=0
     ELSE 
        READ(2022,*)   N_interp_points
        DO i_interp_point=1, N_interp_points
            READ(2022,*)   interp_points(i_interp_point)%X,interp_points(i_interp_point)%Y,interp_points(i_interp_point)%Z
        END DO
        CLOSE (2022)
     END IF
          
!$OMP PARALLEL DO DEFAULT(Shared) Private(m,i,j,k,counter,I_Index,J_Index,K_Index,RegValues)  
            DO m=1,N_interp_points  
              counter=0
              RegValues=0.d0
!Precomputing for all Velocity components      
               DO i=1,Nx
                DO j=1,Ny1
                  DO k=1,Nz1
                      IF ((ABS(X  (i)-interp_points(m)%X)< supx).AND.&
                          (ABS(Y12(j)-interp_points(m)%Y)< supy).AND.&
                          (ABS(Z12(k)-interp_points(m)%Z)< supz)) THEN
                
                             
                            counter=counter+1
                            I_Index(counter)=i
                            J_Index(counter)=j
                            K_Index(counter)=k
                            RegValues(counter)=  deltafnc(X  (i),  interp_points(m)%X,  Hpx (i))& 
                                                 *deltafnc(Y12(j), interp_points(m)%Y,  Hy12(j))&
                                                 *deltafnc(Z12(k), interp_points(m)%Z,  Hz12(k))  

                     END IF
                  END DO
                END DO
              END DO   
        
              interp_points(m)%INTERP_POINT_INFO%RegNumberVx=counter
              ALLOCATE( interp_points(m)%INTERP_POINT_INFO%Vx_X_IND(counter), interp_points(m)%INTERP_POINT_INFO%Vx_Y_IND(counter), &
                        interp_points(m)%INTERP_POINT_INFO%Vx_Z_IND(counter), interp_points(m)%INTERP_POINT_INFO%Vx_WEIGHT(counter))
      
                        interp_points(m)%INTERP_POINT_INFO%Vx_X_IND (1:counter) = I_Index(1:counter)
                        interp_points(m)%INTERP_POINT_INFO%Vx_Y_IND (1:counter) = J_Index(1:counter)
                        interp_points(m)%INTERP_POINT_INFO%Vx_Z_IND (1:counter) = K_Index(1:counter)
                        interp_points(m)%INTERP_POINT_INFO%Vx_Weight(1:counter)=RegValues(1:counter)
    
    
              counter=0
              RegValues=0.d0
              
              DO i=1,Nx1
                  DO j=1,Ny
                    DO k=1,Nz1
                      IF ((ABS(X12(i)-interp_points(m)%X)< supx).AND.&
                          (ABS(Y(j)  -interp_points(m)%Y)< supy).AND.&
                          (ABS(Z12(k)-interp_points(m)%Z)< supz)) THEN
                
                             
                            counter=counter+1
                            I_Index(counter)=i
                            J_Index(counter)=j
                            K_Index(counter)=k
                            RegValues(counter)=  deltafnc(X12 (i),interp_points(m)%X, Hx12 (i))& 
                                                *deltafnc(Y  (j), interp_points(m)%Y, Hpy(j))&
                                                *deltafnc(Z12(k), interp_points(m)%Z, Hz12(k))  

                     END IF
                  END DO
                END DO
              END DO   
        
              interp_points(m)%INTERP_POINT_INFO%RegNumberVy=counter
              ALLOCATE( interp_points(m)%INTERP_POINT_INFO%Vy_X_IND(counter),interp_points(m)%INTERP_POINT_INFO%Vy_Y_IND(counter), &
                        interp_points(m)%INTERP_POINT_INFO%Vy_Z_IND(counter),interp_points(m)%INTERP_POINT_INFO%Vy_WEIGHT(counter))
              
                        interp_points(m)%INTERP_POINT_INFO%Vy_X_IND (1:counter) = I_Index(1:counter)
                        interp_points(m)%INTERP_POINT_INFO%Vy_Y_IND (1:counter) = J_Index(1:counter)
                        interp_points(m)%INTERP_POINT_INFO%Vy_Z_IND (1:counter) = K_Index(1:counter)
                        interp_points(m)%INTERP_POINT_INFO%Vy_Weight(1:counter)=RegValues(1:counter)
                
                
       
              counter=0
              RegValues=0.d0
            DO i=1,Nx1 
                DO j=1,Ny1 
                    DO k=1,Nz 
                          IF ((ABS(X12(i)-interp_points(m)%X)< supx).AND.&
                              (ABS(Y12(j)-interp_points(m)%Y)< supy).AND.&
                              (ABS(Z  (k)-interp_points(m)%Z)< supz)) THEN
                
                             
                            counter=counter+1
                            I_Index(counter)=i
                            J_Index(counter)=j
                            K_Index(counter)=k
                            RegValues(counter)=  deltafnc(X12 (i),interp_points(m)%X, Hx12(i))& 
                                                *deltafnc(Y12(j), interp_points(m)%Y, Hy12(j))&
                                                *deltafnc(Z  (k), interp_points(m)%Z, Hpz(k))  

                     END IF
                  END DO
                END DO
              END DO   
        
              interp_points(m)%INTERP_POINT_INFO%RegNumberVz=counter
              ALLOCATE( interp_points(m)%INTERP_POINT_INFO%Vz_X_IND(counter),interp_points(m)%INTERP_POINT_INFO%Vz_Y_IND(counter), &
                        interp_points(m)%INTERP_POINT_INFO%Vz_Z_IND(counter),interp_points(m)%INTERP_POINT_INFO%Vz_WEIGHT(counter))
              
                        interp_points(m)%INTERP_POINT_INFO%Vz_X_IND (1:counter) = I_Index(1:counter)
                        interp_points(m)%INTERP_POINT_INFO%Vz_Y_IND (1:counter) = J_Index(1:counter)
                        interp_points(m)%INTERP_POINT_INFO%Vz_Z_IND (1:counter) = K_Index(1:counter)
                        interp_points(m)%INTERP_POINT_INFO%Vz_Weight(1:counter)=RegValues(1:counter)
      
           
    END DO
    
    
    DO WHILE (readinput)
       WRITE(file_num,"(I3.3)") n_body+1
       INQUIRE(file="body."//file_num//".inp",exist=readinput)
       IF (readinput) THEN
          n_body= n_body+1
          OPEN(unit=8,file="body."//file_num//".inp",form='formatted',status='old')
          READ(8,*)   bdy(n_body)%npts
          READ(8,*)   bdy(n_body)%moving
          READ(8,101) bdy(n_body)%X_C, bdy(n_body)%Y_C, bdy(n_body)%Z_C
          
          
          ALLOCATE( bdy(n_body)%x(bdy(n_body)%npts),         	 bdy(n_body)%y(bdy(n_body)%npts),         	    bdy(n_body)%z(bdy(n_body)%npts),&
                    bdy(n_body)%Vx_interp_New(bdy(n_body)%npts), bdy(n_body)%Vy_interp_New(bdy(n_body)%npts), 	bdy(n_body)%Vz_interp_New(bdy(n_body)%npts),&
                    bdy(n_body)%Vx_b(bdy(n_body)%npts),      	 bdy(n_body)%Vy_b(bdy(n_body)%npts),      	    bdy(n_body)%Vz_b(bdy(n_body)%npts),         &
                    bdy(n_body)%fb_x(bdy(n_body)%npts),      	 bdy(n_body)%fb_y(bdy(n_body)%npts),         	bdy(n_body)%fb_z(bdy(n_body)%npts),         &
                    bdy(n_body)%fb_x_tag(bdy(n_body)%npts),      bdy(n_body)%fb_y_tag(bdy(n_body)%npts),        bdy(n_body)%fb_z_tag(bdy(n_body)%npts),&
                    bdy(n_body)%point_info(bdy(n_body)%npts))
         
          
          DO i=1,bdy(n_body)%npts
             READ(8,*) bdy(n_body)%x(i), bdy(n_body)%y(i),bdy(n_body)%z(i)
          END DO
          bdy(n_body)%x=bdy(n_body)%x+bdy(n_body)%X_C
          bdy(n_body)%y=bdy(n_body)%y+bdy(n_body)%Y_C
          bdy(n_body)%z=bdy(n_body)%z+bdy(n_body)%Z_C
          CLOSE(8)
         
       
        bdy(n_body)%Vx_interp_New=0.d0
        bdy(n_body)%Vy_interp_New=0.d0
        bdy(n_body)%Vz_interp_New=0.d0
        bdy(n_body)%Vx_b=0.D0      
        bdy(n_body)%Vy_b=0.D0      
        bdy(n_body)%Vz_b=0.D0  
        
        IF (IprintIBM < 0) THEN
             write(*,*) "Starting to read IB forces. If error is generated assign"
             write(*,*) "positive value to IprintForces flag in Lid.dat file."
             Read (3) ( bdy(n_body)%fb_x(i), i=1,bdy(n_body)%npts)
             Read (3) ( bdy(n_body)%fb_y(i), i=1,bdy(n_body)%npts)
             Read (3) ( bdy(n_body)%fb_z(i), i=1,bdy(n_body)%npts)
        ELSE
            bdy(n_body)%fb_x=0.D0 
            bdy(n_body)%fb_y=0.D0 
            bdy(n_body)%fb_z=0.D0 
            bdy(n_body)%Number_Of_Matrix_B_Entries_X=0
            bdy(n_body)%Number_Of_Matrix_B_Entries_Y=0
            bdy(n_body)%Number_Of_Matrix_B_Entries_Z=0
        
        END IF
        
        bdy(n_body)%Dv=(AspRa/Ny1)**3
         
       ! If body is stationary then all regularization and interpoaltion wheights are precomputed
       
       IF (bdy(n_body)%moving==.FALSE.) THEN 
                    
            n_entries_x = bdy(n_body)%Number_Of_Matrix_B_Entries_X
            n_entries_y = bdy(n_body)%Number_Of_Matrix_B_Entries_Y
            n_entries_z = bdy(n_body)%Number_Of_Matrix_B_Entries_Z

!$OMP PARALLEL DO DEFAULT(Shared) Private(m,i,j,k,counter,I_Index,J_Index,K_Index,RegValues) &
!$OMP& reduction(+:n_entries_x,n_entries_y,n_entries_z)
            DO m=1,bdy(n_body)%npts  
       
  
              counter=0
              RegValues=0.d0
       
    
          !Precomputing for all Velocity components      
               
            DO k=1,Nz1
                DO j=1,Ny1
                  DO i=1,Nx
                      
                    IF ((ABS(X  (i)-bdy(n_body)%X(m))< supx).AND.&
                        (ABS(Y12(j)-bdy(n_body)%Y(m))< supy).AND.&
                        (ABS(Z12(k)-bdy(n_body)%Z(m))< supz)) THEN
                
                             
                            counter=counter+1
                            I_Index(counter)=i
                            J_Index(counter)=j
                            K_Index(counter)=k
                            RegValues(counter)=  deltafnc(X  (i),  bdy(n_body)%X(m),  Hpx (i))& 
                                                 *deltafnc(Y12(j), bdy(n_body)%Y(m), Hy12(j))&
                                                 *deltafnc(Z12(k), bdy(n_body)%Z(m), Hz12(k))  

                     END IF
                  END DO
                END DO
              END DO   
        
              bdy(n_body)%POINT_INFO(m)%RegNumberVx=counter
              ALLOCATE( bdy(n_body)%POINT_INFO(m)%Vx_X_IND(counter), bdy(n_body)%POINT_INFO(m)%Vx_Y_IND(counter), &
                        bdy(n_body)%POINT_INFO(m)%Vx_Z_IND(counter), bdy(n_body)%POINT_INFO(m)%Vx_WEIGHT(counter))
                        bdy(n_body)%POINT_INFO(m)%Vx_X_IND (1:counter) = I_Index(1:counter)
                        bdy(n_body)%POINT_INFO(m)%Vx_Y_IND (1:counter) = J_Index(1:counter)
                        bdy(n_body)%POINT_INFO(m)%Vx_Z_IND (1:counter) = K_Index(1:counter)
                        bdy(n_body)%POINT_INFO(m)%Vx_Weight(1:counter)=RegValues(1:counter)
                        n_entries_x = n_entries_x + bdy(n_body)%POINT_INFO(m)%RegNumberVx
    
              counter=0
              RegValues=0.d0
         
            DO k=1,Nz1
                 DO i=1,Nx1
                      DO j=1,Ny
                    
                      IF ((ABS(X12(i)-bdy(n_body)%X(m))< supx).AND.&
                         (ABS(Y(j)  -bdy(n_body)%Y(m))< supy).AND.&
                         (ABS(Z12(k)-bdy(n_body)%Z(m))< supz)) THEN
                
                             
                            counter=counter+1
                            I_Index(counter)=i
                            J_Index(counter)=j
                            K_Index(counter)=k
                            RegValues(counter)=  deltafnc(X12 (i), bdy(n_body)%X(m), Hx12 (i))& 
                                                 *deltafnc(Y  (j), bdy(n_body)%Y(m), Hpy(j))&
                                                 *deltafnc(Z12(k), bdy(n_body)%Z(m), Hz12(k))  

                     END IF
                  END DO
                END DO
              END DO   
        
              bdy(n_body)%POINT_INFO(m)%RegNumberVy=counter
              ALLOCATE( bdy(n_body)%POINT_INFO(m)%Vy_X_IND(counter), bdy(n_body)%POINT_INFO(m)%Vy_Y_IND(counter), &
                        bdy(n_body)%POINT_INFO(m)%Vy_Z_IND(counter), bdy(n_body)%POINT_INFO(m)%Vy_WEIGHT(counter))
                        bdy(n_body)%POINT_INFO(m)%Vy_X_IND (1:counter) = I_Index(1:counter)
                        bdy(n_body)%POINT_INFO(m)%Vy_Y_IND (1:counter) = J_Index(1:counter)
                        bdy(n_body)%POINT_INFO(m)%Vy_Z_IND (1:counter) = K_Index(1:counter)
                        bdy(n_body)%POINT_INFO(m)%Vy_Weight(1:counter)=RegValues(1:counter)
                        n_entries_y = n_entries_y + bdy(n_body)%POINT_INFO(m)%RegNumberVy
                
       
              counter=0
              RegValues=0.d0
                     
            DO i=1,Nx1
               DO j=1,Ny1
                 DO k=1,Nz
                    IF ((ABS(X12(i)-bdy(n_body)%X(m))< supx).AND.&
                        (ABS(Y12(j)-bdy(n_body)%Y(m))< supy).AND.&
                        (ABS(Z  (k)-bdy(n_body)%Z(m))< supz)) THEN
                
                             
                            counter=counter+1
                            I_Index(counter)=i
                            J_Index(counter)=j
                            K_Index(counter)=k
                            RegValues(counter)=  deltafnc(X12 (i),bdy(n_body)%X(m), Hx12(i))& 
                                                *deltafnc(Y12(j), bdy(n_body)%Y(m), Hy12(j))&
                                                *deltafnc(Z  (k), bdy(n_body)%Z(m), Hpz(k))  

                     END IF
                  END DO
                END DO
              END DO   
        
              bdy(n_body)%POINT_INFO(m)%RegNumberVz=counter
              ALLOCATE( bdy(n_body)%POINT_INFO(m)%Vz_X_IND(counter), bdy(n_body)%POINT_INFO(m)%Vz_Y_IND(counter), &
                        bdy(n_body)%POINT_INFO(m)%Vz_Z_IND(counter), bdy(n_body)%POINT_INFO(m)%Vz_WEIGHT(counter))
                        bdy(n_body)%POINT_INFO(m)%Vz_X_IND (1:counter) = I_Index(1:counter)
                        bdy(n_body)%POINT_INFO(m)%Vz_Y_IND (1:counter) = J_Index(1:counter)
                        bdy(n_body)%POINT_INFO(m)%Vz_Z_IND (1:counter) = K_Index(1:counter)
                        bdy(n_body)%POINT_INFO(m)%Vz_Weight(1:counter)=RegValues(1:counter)
                        n_entries_z = n_entries_z + bdy(n_body)%POINT_INFO(m)%RegNumberVz
            END DO 

            bdy(n_body)%Number_Of_Matrix_B_Entries_X = n_entries_x
            bdy(n_body)%Number_Of_Matrix_B_Entries_Y = n_entries_y
            bdy(n_body)%Number_Of_Matrix_B_Entries_Z = n_entries_z
     
       END IF
       END IF        
    END DO

101 Format (G15.8)

END SUBROUTINE setup_geometry 

!---------Numbers for v_x------------------

    

SUBROUTINE Build_R_Ftag_Matrix_P_Attached(n)

Integer n,l
INTEGER*8 runningIndexBX,runningIndexBY,runningIndexBZ  

ALLOCATE( bdy(n)%R_Ftag_Matrix_Fx(bdy(n)%Number_Of_Matrix_B_Entries_X),bdy(n)%R_Ftag_Matrix_Fx_Row(bdy(n)%Number_Of_Matrix_B_Entries_X),bdy(n)%R_Ftag_Matrix_Fx_Col(bdy(n)%Number_Of_Matrix_B_Entries_X))
ALLOCATE( bdy(n)%R_Ftag_Matrix_Fy(bdy(n)%Number_Of_Matrix_B_Entries_Y),bdy(n)%R_Ftag_Matrix_Fy_Row(bdy(n)%Number_Of_Matrix_B_Entries_Y),bdy(n)%R_Ftag_Matrix_Fy_Col(bdy(n)%Number_Of_Matrix_B_Entries_Y))
ALLOCATE( bdy(n)%R_Ftag_Matrix_Fz(bdy(n)%Number_Of_Matrix_B_Entries_Z),bdy(n)%R_Ftag_Matrix_Fz_Row(bdy(n)%Number_Of_Matrix_B_Entries_Z),bdy(n)%R_Ftag_Matrix_Fz_Col(bdy(n)%Number_Of_Matrix_B_Entries_Z))

    runningIndexBX=0
    runningIndexBY=0
    runningIndexBZ=0

    DO m=1,bdy(n)%npts
!!$OMP PARALLEL SECTIONS SHARED(bdy, runningIndexBX, runningIndexBY, runningIndexBZ,m,n) PRIVATE(l)
!!$OMP SECTION
        DO l=1, bdy(n)%POINT_INFO(m)%RegNumberVx
               bdy(n)%R_Ftag_Matrix_Fx(runningIndexBX+1)= bdy(n)%POINT_INFO(m)%Vx_Weight(l)*bdy(n)%Dv
               bdy(n)%R_Ftag_Matrix_Fx_Row(runningIndexBX+1)=NumGlp(bdy(n)%POINT_INFO(m)%Vx_X_IND(l),bdy(n)%POINT_INFO(m)%Vx_Y_IND(l),bdy(n)%POINT_INFO(m)%Vx_Z_IND(l))
               bdy(n)%R_Ftag_Matrix_Fx_Col(runningIndexBX+1)=m
               runningIndexBX=runningIndexBX+1
        END DO
        
!!$OMP SECTION     
        DO l=1, bdy(n)%POINT_INFO(m)%RegNumberVy
            bdy(n)%R_Ftag_Matrix_Fy(runningIndexBY+1)= bdy(n)%POINT_INFO(m)%Vy_Weight(l)*bdy(n)%Dv
            bdy(n)%R_Ftag_Matrix_Fy_Row(runningIndexBY+1)=NumGlp(bdy(n)%POINT_INFO(m)%Vy_X_IND(l),bdy(n)%POINT_INFO(m)%Vy_Y_IND(l),bdy(n)%POINT_INFO(m)%Vy_Z_IND(l))
            bdy(n)%R_Ftag_Matrix_Fy_Col(runningIndexBY+1)=m
            runningIndexBY=runningIndexBY+1
        END DO
        
! !$OMP SECTION
        DO l=1, bdy(n)%POINT_INFO(m)%RegNumberVz
            bdy(n)%R_Ftag_Matrix_Fz(runningIndexBZ+1)= bdy(n)%POINT_INFO(m)%Vz_Weight(l)*bdy(n)%Dv
            bdy(n)%R_Ftag_Matrix_Fz_Row(runningIndexBZ+1)=NumGlp(bdy(n)%POINT_INFO(m)%Vz_X_IND(l),bdy(n)%POINT_INFO(m)%Vz_Y_IND(l),bdy(n)%POINT_INFO(m)%Vz_Z_IND(l))
            bdy(n)%R_Ftag_Matrix_Fz_Col(runningIndexBZ+1)=m
            runningIndexBZ=runningIndexBZ+1
        END DO 
!!$OMP END PARALLEL SECTIONS        
    END DO

END SUBROUTINE  Build_R_Ftag_Matrix_P_Attached
       
SUBROUTINE interpolation (x_Velocity, y_Velocity,z_Velocity, object_x,object_y,object_z, n)
Real(kind=8), POINTER::x_Velocity(:,:,:),y_Velocity(:,:,:),z_Velocity(:,:,:),object_x(:),object_y(:),object_z(:)
Integer n
       
    object_x=0.D0      
    object_y=0.D0      
    object_z=0.d0
    
!$OMP PARALLEL DO DEFAULT(Shared) Private(m,i,j,k,o)       
    DO m=1,bdy(n)%npts
        
    IF (bdy(n)%moving==.TRUE.) THEN     
    ! x-directional vector    
      DO i=0,Nx-1
        DO j=0,Ny
          DO k=0,Nz
            IF ((ABS(X(i)  -bdy(n)%X(m))< supx).AND.&
               (ABS(Y12(j)-bdy(n)%Y(m))< supy).AND.&
                (ABS(Z12(k)-bdy(n)%Z(m))< supz)) THEN           
                    object_x(m)=object_x(m)+x_Velocity(i,j,k)* Hpx(i)* Hy12(j)*Hz12(k)&
                                                             *deltafnc(X(i),   bdy(n)%X(m), Hpx(i))& 
                                                             *deltafnc(Y12(j), bdy(n)%Y(m), Hy12(j))&
                                                             *deltafnc(Z12(k), bdy(n)%Z(m), Hz12(k)) 
             END IF
          END DO
        END DO
      END DO 
      
   ! y-directional vector
    DO i=0,Nx
      DO j=0,Ny-1
       DO k=0,Nz
            IF ((ABS(X12(i)-bdy(n)%X(m)) < supx).AND.&
                (ABS(Y(j)  -bdy(n)%Y(m)) < supy).AND.&
                (ABS(Z12(k)-bdy(n)%Z(m)) < supx)) THEN
                    object_y(m)=object_y(m)+y_Velocity(i,j,k)*Hx12(i)* Hpy(j)*Hz12(k)&
                                                             *deltafnc(X12(i),bdy(n)%X(m), Hx12(i))& 
                                                             *deltafnc(Y(j),  bdy(n)%Y(m), Hpy(j)) &
                                                             *deltafnc(Z12(k),bdy(n)%Z(m), Hz12(k))  
             END IF
          END DO
       END DO
    END DO

! z-dirrectional vectorbject_x(m)=object_x(m)
    DO i=0,Nx
       DO j=0,Ny
         DO k=0,Nz-1
       
            IF ((ABS(X12(i)-bdy(n)%X(m))< supx).AND.&
                (ABS(Y12(j)-bdy(n)%Y(m))< supy).AND.&
                (ABS(Z(k)  -bdy(n)%Z(m))< supz)) THEN    
                    object_z(m)=object_z(m)+z_Velocity(i,j,k)*Hx12(i)* Hy12(j)*Hpz(k)&
                                                             *deltafnc(X12(i), bdy(n)%X(m), Hx12(i))& 
                                                             *deltafnc(Y12(j), bdy(n)%Y(m), Hy12(j))&
                                                             *deltafnc(Z  (k), bdy(n)%Z(m), Hpz(k))              
             END IF
          END DO
       END DO
    END DO
    
    ELSE

     DO o=1,bdy(n)%POINT_INFO(m)%RegNumberVx
!$OMP atomic      
          object_x(m)=object_x(m)+&
          x_Velocity(bdy(n)%POINT_INFO(m)%Vx_X_IND(o),bdy(n)%POINT_INFO(m)%Vx_Y_IND(o),bdy(n)%POINT_INFO(m)%Vx_Z_IND(o))&
         *Hpx(bdy(n)%POINT_INFO(m)%Vx_X_IND(o))* Hy12(bdy(n)%POINT_INFO(m)%Vx_Y_IND(o))*Hz12(bdy(n)%POINT_INFO(m)%Vx_Z_IND(o))&
         *bdy(n)%POINT_INFO(m)%Vx_Weight(o)                                                           
     END DO
 
      DO o=1,bdy(n)%POINT_INFO(m)%RegNumberVy
!$OMP atomic      
          object_y(m)=object_y(m)+&
          y_Velocity(bdy(n)%POINT_INFO(m)%Vy_X_IND(o),bdy(n)%POINT_INFO(m)%Vy_Y_IND(o),bdy(n)%POINT_INFO(m)%Vy_Z_IND(o))&
         *Hx12(bdy(n)%POINT_INFO(m)%Vy_X_IND(o))* Hpy(bdy(n)%POINT_INFO(m)%Vy_Y_IND(o))*Hz12(bdy(n)%POINT_INFO(m)%Vy_Z_IND(o))&
         *bdy(n)%POINT_INFO(m)%Vy_Weight(o)                                                           
      END DO
      
     DO o=1,bdy(n)%POINT_INFO(m)%RegNumberVz
!$OMP atomic      
          object_z(m)=object_z(m)+&
          z_Velocity(bdy(n)%POINT_INFO(m)%Vz_X_IND(o),bdy(n)%POINT_INFO(m)%Vz_Y_IND(o),bdy(n)%POINT_INFO(m)%Vz_Z_IND(o))&
         *Hx12(bdy(n)%POINT_INFO(m)%Vz_X_IND(o))* Hy12(bdy(n)%POINT_INFO(m)%Vz_Y_IND(o))*Hpz(bdy(n)%POINT_INFO(m)%Vz_Z_IND(o))&
         *bdy(n)%POINT_INFO(m)%Vz_Weight(o)                                                           
      END DO
      
  END IF 
END DO


END SUBROUTINE interpolation



SUBROUTINE interpolation_Velocity_postproc (x_Velocity, y_Velocity,z_Velocity, object_x,object_y,object_z,n)
Real(kind=8), POINTER::x_Velocity(:,:,:),y_Velocity(:,:,:),z_Velocity(:,:,:)
Real(kind=8)::object_x,object_y,object_z
Integer n,o
       
    object_x=0.D0      
    object_y=0.D0      
    object_z=0.d0
    
     
    DO o=1,interp_points(n)%INTERP_POINT_INFO%RegNumberVx

          object_x=object_x+&
          x_Velocity(interp_points(n)%INTERP_POINT_INFO%Vx_X_IND(o),&
                     interp_points(n)%INTERP_POINT_INFO%Vx_Y_IND(o),&
                     interp_points(n)%INTERP_POINT_INFO%Vx_Z_IND(o))&
         
          *Hpx(interp_points(n)%INTERP_POINT_INFO%Vx_X_IND(o))*&
          Hy12(interp_points(n)%INTERP_POINT_INFO%Vx_Y_IND(o))*&
          Hz12(interp_points(n)%INTERP_POINT_INFO%Vx_Z_IND(o))&
         
        *interp_points(n)%INTERP_POINT_INFO%Vx_Weight(o)                                                            
    END DO
    
     
    DO o=1,interp_points(n)%INTERP_POINT_INFO%RegNumberVy

          object_y=object_y+&
          y_Velocity(interp_points(n)%INTERP_POINT_INFO%Vy_X_IND(o),&
                     interp_points(n)%INTERP_POINT_INFO%Vy_Y_IND(o),&
                     interp_points(n)%INTERP_POINT_INFO%Vy_Z_IND(o))&
         
         *Hx12(interp_points(n)%INTERP_POINT_INFO%Vy_X_IND(o))*&
           Hpy(interp_points(n)%INTERP_POINT_INFO%Vy_Y_IND(o))*&
          Hz12(interp_points(n)%INTERP_POINT_INFO%Vy_Z_IND(o))&
         
        *interp_points(n)%INTERP_POINT_INFO%Vy_Weight(o)                                                            
    END DO
    
    DO o=1,interp_points(n)%INTERP_POINT_INFO%RegNumberVz

          object_z=object_z+&
          z_Velocity(interp_points(n)%INTERP_POINT_INFO%Vz_X_IND(o),&
                     interp_points(n)%INTERP_POINT_INFO%Vz_Y_IND(o),&
                     interp_points(n)%INTERP_POINT_INFO%Vz_Z_IND(o))&
         
         *Hx12(interp_points(n)%INTERP_POINT_INFO%Vz_X_IND(o))*&
          Hy12(interp_points(n)%INTERP_POINT_INFO%Vz_Y_IND(o))*&
           Hpz(interp_points(n)%INTERP_POINT_INFO%Vz_Z_IND(o))&    
        *interp_points(n)%INTERP_POINT_INFO%Vz_Weight(o)                                                            
    END DO
 
    

END SUBROUTINE interpolation_Velocity_postproc



SUBROUTINE BodyForce(n)
Integer n 

    bdy(n)%fb_x= (bdy(n)%Vx_b-bdy(n)%Vx_interp_New)/Htime
    bdy(n)%fb_y= (bdy(n)%Vy_b-bdy(n)%Vy_interp_New)/Htime
    bdy(n)%fb_z= (bdy(n)%Vz_b-bdy(n)%Vz_interp_New)/Htime
    
!    bdy(n)%fb_x= (3.d0*bdy(n)%Vx_b-4.0*bdy(n)%Vx_interp_new+bdy(n)%Vx_interp)/(2.d0*Htime)
!    bdy(n)%fb_y= (3.d0*bdy(n)%Vy_b-4.0*bdy(n)%Vy_interp_new+bdy(n)%Vy_interp)/(2.d0*Htime)
!    bdy(n)%fb_z= (3.d0*bdy(n)%Vz_b-4.0*bdy(n)%Vz_interp_new+bdy(n)%Vz_interp)/(2.d0*Htime)
    
END SUBROUTINE BodyForce




SUBROUTINE regularization(n)
Integer n 
Real(kind=8) proba
!$OMP PARALLEL DO DEFAULT(Shared) Private(m,i,j,k,o,proba) 
    DO m=1,bdy(n)%npts
    IF (bdy(n)%moving==.TRUE.) THEN
    ! x-directional vector     
      DO i=0,Nx-1
        DO j=0,Ny
          DO k=0,Nz
            IF ((ABS(X(i)  -bdy(n)%X(m))< supx).AND.&
                (ABS(Y12(j)-bdy(n)%Y(m))< supy).AND.&
                (ABS(Z12(k)-bdy(n)%Z(m))< supz)) THEN        
                   
                RHSx(i,j,k)=RHSx(i,j,k)-bdy(n)%fb_x(m)*bdy(n)%dV&   
                                                *deltafnc(X(i),   bdy(n)%X(m), Hpx(i))& 
                                                *deltafnc(Y12(j), bdy(n)%Y(m), Hy12(j))&
                                                *deltafnc(Z12(k), bdy(n)%Z(m), Hz12(k))  
             END IF
          END DO
        END DO
      END DO 
      
    ! y-directional vector
    DO i=0,Nx
      DO j=0,Ny-1
        DO k=0,Nz
            IF ((ABS(X12(i)-bdy(n)%X(m)) < supx).AND.&
                (ABS(Y(j)  -bdy(n)%Y(m)) < supy).AND.&
                (ABS(Z12(k)-bdy(n)%Z(m)) < supx)) THEN
                    
                    RHSy(i,j,k)=RHSy(i,j,k)-bdy(n)%fb_y(m)*bdy(n)%dV&
                                                *deltafnc(X12(i),bdy(n)%X(m), Hx12(i))& 
                                                *deltafnc(Y(j),  bdy(n)%Y(m), Hpy(j))&
                                                *deltafnc(Z12(k),bdy(n)%Z(m), Hz12(k))    
             END IF
          END DO
      END DO
    END DO

! z-dirrectional vector
    DO i=0,Nx
       DO j=0,Ny
         DO k=0,Nz-1
            IF ((ABS(X12(i)-bdy(n)%X(m))< supx).AND.&
                (ABS(Y12(j)-bdy(n)%Y(m))< supy).AND.&
                (ABS(Z(k)  -bdy(n)%Z(m))< supz)) THEN  
                    
                    RHSz(i,j,k)=RHSz(i,j,k)-bdy(n)%fb_z(m)*bdy(n)%dV& 
                                                *deltafnc(X12(i),bdy(n)%X(m), Hx12(i))& 
                                                *deltafnc(Y12(j),bdy(n)%Y(m), Hy12(j))&
                                                *deltafnc(Z(k),  bdy(n)%Z(m), Hpz(k)) 
             END IF
          END DO
       END DO
    END DO
    
    ELSE 
        
     DO o=1,bdy(n)%POINT_INFO(m)%RegNumberVx
           proba=bdy(n)%fb_x(m)*bdy(n)%dV*bdy(n)%POINT_INFO(m)%Vx_Weight(o)
!$OMP atomic 
           RHSx(bdy(n)%POINT_INFO(m)%Vx_X_IND(o),bdy(n)%POINT_INFO(m)%Vx_Y_IND(o),bdy(n)%POINT_INFO(m)%Vx_Z_IND(o))=&
           RHSx(bdy(n)%POINT_INFO(m)%Vx_X_IND(o),bdy(n)%POINT_INFO(m)%Vx_Y_IND(o),bdy(n)%POINT_INFO(m)%Vx_Z_IND(o))-& !- apperas as it goes to the left
           proba
     END DO
     
     DO o=1,bdy(n)%POINT_INFO(m)%RegNumberVy
           proba= bdy(n)%fb_y(m)*bdy(n)%dV*bdy(n)%POINT_INFO(m)%Vy_Weight(o)
!$OMP atomic 
           RHSy(bdy(n)%POINT_INFO(m)%Vy_X_IND(o),bdy(n)%POINT_INFO(m)%Vy_Y_IND(o),bdy(n)%POINT_INFO(m)%Vy_Z_IND(o))=&
           RHSy(bdy(n)%POINT_INFO(m)%Vy_X_IND(o),bdy(n)%POINT_INFO(m)%Vy_Y_IND(o),bdy(n)%POINT_INFO(m)%Vy_Z_IND(o))-& !- apperas as it goes to the left
           proba
     END DO
     
    DO o=1,bdy(n)%POINT_INFO(m)%RegNumberVz
	   proba=bdy(n)%fb_z(m)*bdy(n)%dV*bdy(n)%POINT_INFO(m)%Vz_Weight(o)
!$OMP atomic 
           RHSz(bdy(n)%POINT_INFO(m)%Vz_X_IND(o),bdy(n)%POINT_INFO(m)%Vz_Y_IND(o),bdy(n)%POINT_INFO(m)%Vz_Z_IND(o))=&
           RHSz(bdy(n)%POINT_INFO(m)%Vz_X_IND(o),bdy(n)%POINT_INFO(m)%Vz_Y_IND(o),bdy(n)%POINT_INFO(m)%Vz_Z_IND(o))-& !- apperas as it goes to the left
           proba
     END DO
     
    
    END IF
    END DO
END SUBROUTINE regularization


SUBROUTINE regularization_tag(n)
Integer n 
Real(kind=8) proba
!$OMP PARALLEL DO DEFAULT(Shared) Private(m,i,j,k,o,proba) 
    DO m=1,bdy(n)%npts
    IF (bdy(n)%moving==.TRUE.) THEN
    ! x-directional vector     
      DO i=0,Nx-1
        DO j=0,Ny
          DO k=0,Nz
            IF ((ABS(X(i)  -bdy(n)%X(m))< supx).AND.&
                (ABS(Y12(j)-bdy(n)%Y(m))< supy).AND.&
                (ABS(Z12(k)-bdy(n)%Z(m))< supz)) THEN        
                   
                RHSx(i,j,k)=RHSx(i,j,k)+bdy(n)%fb_x_tag(m)*bdy(n)%dV&   
                                                *deltafnc(X(i),   bdy(n)%X(m), Hpx(i))& 
                                                *deltafnc(Y12(j), bdy(n)%Y(m), Hy12(j))&
                                                *deltafnc(Z12(k), bdy(n)%Z(m), Hz12(k))  
             END IF
          END DO
        END DO
      END DO 
      
    ! y-directional vector
    DO i=0,Nx
      DO j=0,Ny-1
        DO k=0,Nz
            IF ((ABS(X12(i)-bdy(n)%X(m)) < supx).AND.&
                (ABS(Y(j)  -bdy(n)%Y(m)) < supy).AND.&
                (ABS(Z12(k)-bdy(n)%Z(m)) < supx)) THEN
                    
                    RHSy(i,j,k)=RHSy(i,j,k)+bdy(n)%fb_y_tag(m)*bdy(n)%dV&
                                                *deltafnc(X12(i),bdy(n)%X(m), Hx12(i))& 
                                                *deltafnc(Y(j),  bdy(n)%Y(m), Hpy(j))&
                                                *deltafnc(Z12(k),bdy(n)%Z(m), Hz12(k))    
             END IF
          END DO
      END DO
    END DO

! z-dirrectional vector
    DO i=0,Nx
       DO j=0,Ny
         DO k=0,Nz-1
            IF ((ABS(X12(i)-bdy(n)%X(m))< supx).AND.&
                (ABS(Y12(j)-bdy(n)%Y(m))< supy).AND.&
                (ABS(Z(k)  -bdy(n)%Z(m))< supz)) THEN  
                    
                    RHSz(i,j,k)=RHSz(i,j,k)+bdy(n)%fb_z_tag(m)*bdy(n)%dV& 
                                                *deltafnc(X12(i),bdy(n)%X(m), Hx12(i))& 
                                                *deltafnc(Y12(j),bdy(n)%Y(m), Hy12(j))&
                                                *deltafnc(Z(k),  bdy(n)%Z(m), Hpz(k)) 
             END IF
          END DO
       END DO
    END DO
    
    ELSE 
        
     DO o=1,bdy(n)%POINT_INFO(m)%RegNumberVx
           proba=bdy(n)%fb_x_tag(m)*bdy(n)%dV*bdy(n)%POINT_INFO(m)%Vx_Weight(o)
!$OMP atomic 
           RHSx(bdy(n)%POINT_INFO(m)%Vx_X_IND(o),bdy(n)%POINT_INFO(m)%Vx_Y_IND(o),bdy(n)%POINT_INFO(m)%Vx_Z_IND(o))=&
           RHSx(bdy(n)%POINT_INFO(m)%Vx_X_IND(o),bdy(n)%POINT_INFO(m)%Vx_Y_IND(o),bdy(n)%POINT_INFO(m)%Vx_Z_IND(o))+&
           proba
     END DO
     
     DO o=1,bdy(n)%POINT_INFO(m)%RegNumberVy
           proba= bdy(n)%fb_y_tag(m)*bdy(n)%dV*bdy(n)%POINT_INFO(m)%Vy_Weight(o)
!$OMP atomic 
           RHSy(bdy(n)%POINT_INFO(m)%Vy_X_IND(o),bdy(n)%POINT_INFO(m)%Vy_Y_IND(o),bdy(n)%POINT_INFO(m)%Vy_Z_IND(o))=&
           RHSy(bdy(n)%POINT_INFO(m)%Vy_X_IND(o),bdy(n)%POINT_INFO(m)%Vy_Y_IND(o),bdy(n)%POINT_INFO(m)%Vy_Z_IND(o))+&
           proba
     END DO
     
    DO o=1,bdy(n)%POINT_INFO(m)%RegNumberVz
	   proba=bdy(n)%fb_z_tag(m)*bdy(n)%dV*bdy(n)%POINT_INFO(m)%Vz_Weight(o)
!$OMP atomic 
           RHSz(bdy(n)%POINT_INFO(m)%Vz_X_IND(o),bdy(n)%POINT_INFO(m)%Vz_Y_IND(o),bdy(n)%POINT_INFO(m)%Vz_Z_IND(o))=&
           RHSz(bdy(n)%POINT_INFO(m)%Vz_X_IND(o),bdy(n)%POINT_INFO(m)%Vz_Y_IND(o),bdy(n)%POINT_INFO(m)%Vz_Z_IND(o))+&
           proba
     END DO
     
    
    END IF
    END DO
END SUBROUTINE regularization_tag



FUNCTION deltafnc( r,r0, dr ) 

    REAL(KIND(0.D0)) :: r,r0,dr,deltafnc

    ! Roma, Peskin, & Berger (JCP 1999)
    IF (ABS(r-r0)<=(0.5D0*dr)) THEN
       deltafnc = (1.D0+SQRT(-3.D0*((r-r0)/dr)**2+1.D0))/(3.0D0*dr)
    ELSEIF (ABS(r-r0)<=(1.5D0*dr)) THEN
       deltafnc = (5.D0-3.D0*ABS((r-r0)/dr)-SQRT(-3.D0*(1.D0-ABS((r-r0)/dr))**2+1.D0))/(6.D0*dr)
    ELSE 
       deltafnc = 0.D0
    END IF
END FUNCTION deltafnc


!FUNCTION deltafnc( r,r0, dr ) 

!    REAL(KIND(0.D0)) :: r,r0,dr,deltafnc
!    REAL(KIND(0.D0)) :: arg,S1,S2
!    arg=ABS((r-r0)/dr)
    
!    IF (ABS(r-r0)<=(dr)) THEN 
!       deltafnc = 61.d0/112.d0-11.d0/42.d0*arg-11.d0/56.d0*arg**2.d0+1.d0/12.d0*arg**3.d0+&
!           3.d0**0.5d0/336.d0*(243.d0+1584.d0*arg-748.d0*arg**2.d0-1560.d0*arg**3.d0+&
!           500.d0*arg**4.d0+ 336.d0*arg**5.d0 - 112.d0*arg**6.d0)**0.5d0
!       deltafnc=deltafnc/dr
!    ELSEIF (ABS(r-r0)<=(2.D0*dr)) THEN
!        S1=21.d0/16.d0+7.d0/12.d0*arg-7.d0/8.d0*arg**2.d0+1.d0/6.d0*arg**3.d0
!        arg=arg-1.d0
!        S2= 61.d0/112.d0-11.d0/42.d0*arg-11.d0/56.d0*arg**2.d0+1.d0/12.d0*arg**3.d0+&
!            3.d0**0.5d0/336.d0*(243.d0+1584.d0*arg-748.d0*arg**2.d0-1560.d0*arg**3.d0+&
!            500.d0*arg**4.d0+ 336.d0*arg**5.d0 - 112.d0*arg**6.d0)**0.5d0
!        deltafnc=(S1-1.5d0*S2)
!        deltafnc=deltafnc/dr
!    ELSEIF (ABS(r-r0)<=(3.D0*dr)) THEN
!        S1=9.d0/8.d0-23.d0/12.d0*arg+0.75d0*arg**2.d0-1.d0/12.d0*arg**3.d0
!        arg=arg-2.d0
!        S2= 61.d0/112.d0-11.d0/42.d0*arg-11.d0/56.d0*arg**2.d0+1.d0/12.d0*arg**3.d0+&
!            3.d0**0.5d0/336.d0*(243.d0+1584.d0*arg-748.d0*arg**2.d0-1560.d0*arg**3.d0+&
!            500.d0*arg**4.d0+ 336.d0*arg**5.d0 - 112.d0*arg**6.d0)**0.5d0
!        deltafnc=(S1+0.5d0*S2)  
!        deltafnc=deltafnc/dr!
!    ELSE 
!       deltafnc = 0.D0
!    END IF
!END FUNCTION deltafnc





SUBROUTINE DestroyAll 
 
    INTEGER :: i,m
      
    DO i=1,n_body
       
        IF (bdy(i)%moving==.FALSE.) THEN
        DO m=1,bdy(i)%npts
           IF (bdy(i)%Tparticip==.TRUE.)&  
          
           DEALLOCATE( bdy(i)%POINT_INFO(m)%Vx_X_IND, bdy(i)%POINT_INFO(m)%Vx_Y_IND, &
                       bdy(i)%POINT_INFO(m)%Vx_Z_IND, bdy(i)%POINT_INFO(m)%Vx_WEIGHT)
           
           DEALLOCATE( bdy(i)%POINT_INFO(m)%Vy_X_IND, bdy(i)%POINT_INFO(m)%Vy_Y_IND, &
                       bdy(i)%POINT_INFO(m)%Vy_Z_IND, bdy(i)%POINT_INFO(m)%Vy_WEIGHT)
           
           DEALLOCATE( bdy(i)%POINT_INFO(m)%Vz_X_IND, bdy(i)%POINT_INFO(m)%Vz_Y_IND, &
                       bdy(i)%POINT_INFO(m)%Vz_Z_IND, bdy(i)%POINT_INFO(m)%Vz_WEIGHT)
           
         END DO 
         END IF
         DEALLOCATE ( &
                       bdy(i)%x, bdy(i)%y, bdy(i)%z,&
                       bdy(i)%Vx_interp_New, bdy(i)%Vy_interp_New,  bdy(i)%Vz_interp_New, &                     
                       bdy(i)%Vx_b,          bdy(i)%Vy_b,           bdy(i)%Vz_b,          &
                       bdy(i)%fb_x,          bdy(i)%fb_y,           bdy(i)%fb_z,          & 
                       bdy(i)%fb_x_tag,      bdy(i)%fb_y_tag,       bdy(i)%fb_z_tag,& 
                       bdy(i)%point_info, &
                       bdy(i)%Div_F_X_Val,          bdy(i)%Div_F_X_ROW,         bdy(i)%Div_F_X_COL,&
                       bdy(i)%Div_F_Y_Val,          bdy(i)%Div_F_Y_ROW,         bdy(i)%Div_F_Y_COL,&
                       bdy(i)%Div_F_Z_Val,          bdy(i)%Div_F_Z_ROW,         bdy(i)%Div_F_Z_COL &
                     )  
        
    END DO
    
    
    DEALLOCATE( B_CSR_Prs, BT_CSR_Prs,   B_Row_CSR_Prs,   B_Col_CSR_Prs,   BT_Row_CSR_Prs,   BT_Col_CSR_Prs, RHS_F_tag, RHS_Precond, F_tag,IGPrs)

   

       
END SUBROUTINE 

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    

 SUBROUTINE OrdVarPres
Integer kk, i, j,k
kk=1
!---------Numbers for q_k------------------
 Do k=1, Nz1
    Do j=1, Ny1
        Do i=1, Nx1
                    NumGlP(i,j,k)=kk
                    kk=kk+1
                End Do
            End Do
        End Do
 END SUBROUTINE OrdVarPres
 

SUBROUTINE Div_Reg_Fx_Tag(Div_F_X_Val, Div_F_X_ROW,Div_F_X_COL,nb, dx, Szx)  
  

  REAL(kind=8),  ALLOCATABLE :: Div_F_X_Val(:)
  INTEGER,       ALLOCATABLE :: Div_F_X_ROW(:),Div_F_X_COL(:), a(:), b(:), width(:)
  INTEGER*8 Sz1,Sz2,Szx 
  INTEGER  nb, i, counter_width,counter_b,rcv_counter
 Real(kind=8) dx
  Sz1=size(bdy(nb)%R_Ftag_Matrix_Fx)
  ALLOCATE(a (Sz1-1))
  a   = bdy(nb)%R_Ftag_Matrix_Fx_Row(2:Sz1)-bdy(nb)%R_Ftag_Matrix_Fx_Row(1:Sz1-1)
  b   = pack([(i, i=1,Sz1-1)], a/= 1)
  Sz2 = size(b) 
  ALLOCATE(width(Sz2+1))
  
  counter_width=1
  counter_b=1
  rcv_counter=1

  width(1)=b(1)
  width(2:Sz2)=b(2:Sz2)-b(1:Sz2-1)
  width(Sz2+1)=Sz1-sum(width(1:Sz2))
  
  Szx=size(width)+sum(width)
  ALLOCATE(Div_F_X_Val(Szx),Div_F_X_ROW(Szx),  Div_F_X_COL(Szx))
  
 DO WHILE (rcv_counter<=Sz1)
      
     Div_F_X_Val(counter_width)=bdy(nb)%R_Ftag_Matrix_Fx(rcv_counter)/dx
     Div_F_X_Val(counter_width+1:counter_width+(width(counter_b)-1)) = & 
    (bdy(nb)%R_Ftag_Matrix_Fx(rcv_counter+1:rcv_counter+(width(counter_b)-1))-bdy(nb)%R_Ftag_Matrix_Fx(rcv_counter:rcv_counter+(width(counter_b)-2)))/dx
     
     Div_F_X_Val(counter_width+width(counter_b))=-bdy(nb)%R_Ftag_Matrix_Fx(rcv_counter+(width(counter_b)-1))/dx;

     Div_F_X_ROW(counter_width)=bdy(nb)%R_Ftag_Matrix_Fx_Row(rcv_counter);
     Div_F_X_ROW(counter_width+1:counter_width+(width(counter_b)-1))=bdy(nb)%R_Ftag_Matrix_Fx_Row(rcv_counter+1:rcv_counter+(width(counter_b)-1));
     Div_F_X_ROW(counter_width+width(counter_b))=bdy(nb)%R_Ftag_Matrix_Fx_Row(rcv_counter+(width(counter_b)-1))+1;

     Div_F_X_COL(counter_width:counter_width+width(counter_b)-1)=bdy(nb)%R_Ftag_Matrix_Fx_Col(rcv_counter:rcv_counter+(width(counter_b)-1));
     Div_F_X_COL(counter_width:counter_width+width(counter_b))  =bdy(nb)%R_Ftag_Matrix_Fx_Col(rcv_counter+(width(counter_b)-1));
     
      counter_width=counter_width +width(counter_b)+1;
      rcv_counter=rcv_counter+width(counter_b);
      counter_b=counter_b+1;
       
 END DO 
DEALLOCATE( a,b,width)

310     Format ((  I5,2x, I8,2x, G15.8) ) 
 
END SUBROUTINE Div_Reg_Fx_Tag 

SUBROUTINE Div_Reg_Fy_Tag(Div_F_Y_Val, Div_F_Y_ROW,Div_F_Y_COL,nb, dy, Szy)  
  

  REAL(kind=8),  ALLOCATABLE :: Div_F_Y_Val(:)
  INTEGER,       ALLOCATABLE :: Div_F_Y_ROW(:),Div_F_Y_COL(:), a(:), b(:), width(:)
  INTEGER*8 Sz1,Sz2,Szy 
  INTEGER  nb, i, counter_width,counter_b,rcv_counter
  Real(kind=8) dy
  Sz1=size(bdy(nb)%R_Ftag_Matrix_Fy)
  ALLOCATE(a (Sz1-1))
  a   = bdy(nb)%R_Ftag_Matrix_Fy_Row(2:Sz1)-bdy(nb)%R_Ftag_Matrix_Fy_Row(1:Sz1-1)
  b   = pack([(i, i=1,Sz1-1)], a/= Nx1)
  Sz2 = size(b) 
  ALLOCATE(width(Sz2+1))
  
  counter_width=1
  counter_b=1
  rcv_counter=1

  width(1)=b(1)
  width(2:Sz2)=b(2:Sz2)-b(1:Sz2-1)
  width(Sz2+1)=Sz1-sum(width(1:Sz2))
  
  Szy=size(width)+sum(width)
  ALLOCATE(Div_F_Y_Val(Szy),Div_F_Y_ROW(Szy),  Div_F_Y_COL(Szy))
  
 DO WHILE (rcv_counter<=Sz1)
      
     Div_F_Y_Val(counter_width)=bdy(nb)%R_Ftag_Matrix_Fy(rcv_counter)/dy
     Div_F_Y_Val(counter_width+1:counter_width+(width(counter_b)-1)) = & 
    (bdy(nb)%R_Ftag_Matrix_Fy(rcv_counter+1:rcv_counter+(width(counter_b)-1))-bdy(nb)%R_Ftag_Matrix_Fy(rcv_counter:rcv_counter+(width(counter_b)-2)))/dy
     
     Div_F_Y_Val(counter_width+width(counter_b))=-bdy(nb)%R_Ftag_Matrix_Fy(rcv_counter+(width(counter_b)-1))/dy

     Div_F_Y_ROW(counter_width)=bdy(nb)%R_Ftag_Matrix_Fy_Row(rcv_counter);
     Div_F_Y_ROW(counter_width+1:counter_width+(width(counter_b)-1))=bdy(nb)%R_Ftag_Matrix_Fy_Row(rcv_counter+1:rcv_counter+(width(counter_b)-1));
     Div_F_Y_ROW(counter_width+width(counter_b))=bdy(nb)%R_Ftag_Matrix_Fy_Row(rcv_counter+(width(counter_b)-1))+Nx1;

     Div_F_Y_COL(counter_width:counter_width+width(counter_b)-1)=bdy(nb)%R_Ftag_Matrix_Fy_Col(rcv_counter:rcv_counter+(width(counter_b)-1));
     Div_F_Y_COL(counter_width:counter_width+width(counter_b))  =bdy(nb)%R_Ftag_Matrix_Fy_Col(rcv_counter+(width(counter_b)-1));
     
      counter_width=counter_width +width(counter_b)+1;
      rcv_counter=rcv_counter+width(counter_b);
      counter_b=counter_b+1;
       
 END DO 
  DEALLOCATE( a,b,width)
   
END SUBROUTINE Div_Reg_Fy_Tag 


SUBROUTINE Div_Reg_Fz_Tag(Div_F_Z_Val, Div_F_Z_ROW,Div_F_Z_COL,nb, dz, Szz)  
  

  REAL(kind=8),  ALLOCATABLE :: Div_F_Z_Val(:)
  INTEGER,       ALLOCATABLE :: Div_F_Z_ROW(:),Div_F_Z_COL(:), a(:), b(:), width(:)
  INTEGER*8 Sz1,Sz2,Szz 
  INTEGER  nb, i, counter_width,counter_b,rcv_counter
  Real(kind=8) dz
  Sz1=size(bdy(nb)%R_Ftag_Matrix_Fz)
  ALLOCATE(a (Sz1-1))
  a   = bdy(nb)%R_Ftag_Matrix_Fz_Row(2:Sz1)-bdy(nb)%R_Ftag_Matrix_Fz_Row(1:Sz1-1)
  b   = pack([(i, i=1,Sz1-1)], a/= Nx1*Ny1)
  Sz2 = size(b) 
  ALLOCATE(width(Sz2+1))
  
  counter_width=1
  counter_b=1
  rcv_counter=1

  width(1)=b(1)
  width(2:Sz2)=b(2:Sz2)-b(1:Sz2-1)
  width(Sz2+1)=Sz1-sum(width(1:Sz2))
  
  Szz=size(width)+sum(width)
  ALLOCATE(Div_F_Z_Val(Szz),Div_F_Z_ROW(Szz),  Div_F_Z_COL(Szz))
  
 DO WHILE (rcv_counter<=Sz1)
      
     Div_F_Z_Val(counter_width)=bdy(nb)%R_Ftag_Matrix_Fz(rcv_counter)/dz
     Div_F_Z_Val(counter_width+1:counter_width+(width(counter_b)-1)) = & 
    (bdy(nb)%R_Ftag_Matrix_Fz(rcv_counter+1:rcv_counter+(width(counter_b)-1))-bdy(nb)%R_Ftag_Matrix_Fz(rcv_counter:rcv_counter+(width(counter_b)-2)))/dz
     
     Div_F_Z_Val(counter_width+width(counter_b))=-bdy(nb)%R_Ftag_Matrix_Fz(rcv_counter+(width(counter_b)-1))/dz

     Div_F_Z_ROW(counter_width)=bdy(nb)%R_Ftag_Matrix_Fz_Row(rcv_counter);
     Div_F_Z_ROW(counter_width+1:counter_width+(width(counter_b)-1))=bdy(nb)%R_Ftag_Matrix_Fz_Row(rcv_counter+1:rcv_counter+(width(counter_b)-1));
     Div_F_Z_ROW(counter_width+width(counter_b))=bdy(nb)%R_Ftag_Matrix_Fz_Row(rcv_counter+(width(counter_b)-1))+Nx1*Ny1;

     Div_F_Z_COL(counter_width:counter_width+width(counter_b)-1)=bdy(nb)%R_Ftag_Matrix_Fz_Col(rcv_counter:rcv_counter+(width(counter_b)-1));
     Div_F_Z_COL(counter_width:counter_width+width(counter_b))  =bdy(nb)%R_Ftag_Matrix_Fz_Col(rcv_counter+(width(counter_b)-1));
     
      counter_width=counter_width +width(counter_b)+1;
      rcv_counter=rcv_counter+width(counter_b);
      counter_b=counter_b+1;
       
 END DO 
  DEALLOCATE( a,b,width)
  
END SUBROUTINE Div_Reg_Fz_Tag 

END MODULE IBsetupInetrpRegular
