Subroutine EVDbounds
    Use Numbers
    Use Numerica
    Use Variables

    Implicit Real*8 (a-h,o-z)

    ! =====================================================

    ! --- Temperature Boundary Conditions ---
    if (EVD_BCx == 0) then
        TmpNew(0,:,:)   = 0.D0
        TmpNew(Nx2,:,:) = 0.D0
    Else
        TmpNew(0,:,:)   = TmpNew(1,:,:)
        TmpNew(Nx2,:,:) = TmpNew(Nx1,:,:)
    end if

    if (EVD_BCy == 0) then
        TmpNew(:,0,:)   = 0.D0
        TmpNew(:,Ny2,:) = 0.D0
    Else
        TmpNew(:,0,:)   = TmpNew(:,1,:)
        TmpNew(:,Ny2,:) = TmpNew(:,Ny1,:)
    end if

    if (EVD_BCz == 0) then
        TmpNew(:,:,0)   = 0.D0
        TmpNew(:,:,Nz2) = 0.D0
    Else
        TmpNew(:,:,0)   = TmpNew(:,:,1)
        TmpNew(:,:,Nz2) = TmpNew(:,:,Nz1)
    end if

    ! --- Velocity Boundary Conditions (VMxNew) ---
    VMxNew(0,:,:)   = 0.D0
    VMxNew(Nx1,:,:) = 0.D0
    VMxNew(:,0,:)   = 0.D0
    VMxNew(:,Ny2,:) = 0.D0
    VMxNew(:,:,0)   = 0.D0
    VMxNew(:,:,Nz2) = 0.D0

    ! --- Velocity Boundary Conditions (VMyNew) ---
    VMyNew(0,:,:)   = 0.D0
    VMyNew(Nx2,:,:) = 0.D0
    VMyNew(:,0,:)   = 0.D0
    VMyNew(:,Ny1,:) = 0.D0
    VMyNew(:,:,0)   = 0.D0
    VMyNew(:,:,Nz2) = 0.D0

    ! --- Velocity Boundary Conditions (VMzNew) ---
    VMzNew(0,:,:)   = 0.D0
    VMzNew(Nx2,:,:) = 0.D0
    VMzNew(:,0,:)   = 0.D0
    VMzNew(:,Ny2,:) = 0.D0
    VMzNew(:,:,0)   = 0.D0
    VMzNew(:,:,Nz1) = 0.D0

End Subroutine EVDbounds