Subroutine EVDbounds 

         Use Numbers
         Use Numerica
         Use Parameters
         Use Variables
         Use EVD_Operators

       Implicit Real*8 (a-h,o-z)
       
! =====================================================

                    VMxNew(0  ,:,:) = 0.D0
                    VMxNew(Nx1,:,:) = 0.D0
                    
                    VMxNew(:,  0,:) = 0.D0
                    VMxNew(:,Ny2,:) = 0.D0

                    VMxNew(:,:,  0) = 0.D0
                    VMxNew(:,:,Nz2) = Cos(angle)
                    
                    VMyNew(:,  0,:) = 0.D0
                    VMyNew(:,Ny1,:) = 0.D0
                    
                    VMyNew(:,:,  0) = 0.D0
                    VMyNew(:,:,Nz2) = Sin(angle)

                    VMyNew(  0,:,:) = 0.D0
                    VMyNew(Nx2,:,:) = 0.d0
                    
                    VMzNew(:,  0,:) = 0.D0
                    VMzNew(:,Ny2,:) = 0.D0
                    
                    VMzNew(:,:,  0) = 0.D0
                    VMzNew(:,:,Nz1) = 0.D0
                    
                    VMzNew(  0,:,:) = 0.D0
                    VMzNew(Nx2,:,:) = 0.d0
End Subroutine EVDbounds 
