! ************************************************************
! *     POSING INITIAL VALUES FOR UNKNOWN FUNCTIONS          *
! *     (Updated for (Y, Z, X) Memory Layout)                *
! ************************************************************

Subroutine Init

    Use Numbers
    Use Parameters
    Use Numerica
    Use Grid
    Use Variables

    Implicit Real(kind=8) (A-H,O-Z)

    ! Temporary buffers to hold legacy (X, Y, Z) data during read
    Real(kind=8), Allocatable, Dimension(:,:,:) :: bufX, bufY, bufZ, bufT, bufP

    ! ======================================================================

    If (Iprint < 0) then
        Rewind 3

        ! 1. Allocate buffers using the legacy (X, Y, Z) shapes
        Allocate(bufX(0:Nx1, 0:Ny2, 0:Nz2))
        Allocate(bufY(0:Nx2, 0:Ny1, 0:Nz2))
        Allocate(bufZ(0:Nx2, 0:Ny2, 0:Nz1))
        Allocate(bufT(0:Nx2, 0:Ny2, 0:Nz2))
        Allocate(bufP(1:Nx1, 1:Ny1, 1:Nz1))

        ! 2. Read from file using ORIGINAL loop order (from legacy source)
        Read (3) ((( bufX(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
        Read (3) ((( bufY(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
        Read (3) ((( bufZ(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
        ! [Your existing code]
        Read (3) ((( bufT(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)
        Read (3) ((( bufP(i,j,k), i=1,Nx1), j=1,Ny1), k=1,Nz1)

        ! ======================================================================
        ! AUDIT MILESTONE A: Raw Buffer Data Ingestion (Legacy X, Y, Z order)
        ! ======================================================================
        print *, "--- [AUDIT A] Raw bufT Content from File (X, Y, Z space) ---"
        do k = 0, Nz2
            do j = 0, Ny2
                do i = 0, Nx2
                    if (bufT(i,j,k) /= 0.d0) then
                        print *, "bufT(X=", i, ", Y=", j, ", Z=", k, ") = ", bufT(i,j,k)
                    end if
                end do
            end do
        end do

        ! [Your existing mapping code]
        Do k=0,Nz2; Do j=0,Ny2; Do i=0,Nx2; Tmpr(j,k,i) = bufT(i,j,k); EndDo; EndDo; EndDo
        Do k=1,Nz1; Do j=1,Ny1; Do i=1,Nx1; Prs(j,k,i)  = bufP(i,j,k); EndDo; EndDo; EndDo

        ! ======================================================================
        ! AUDIT MILESTONE B: Transposed Global Array Structure (Y, Z, X order)
        ! ======================================================================
        print *, ""
        print *, "--- [AUDIT B] Global Transposed Tmpr Array (Y, Z, X space) ---"
        do i = 0, Nx2
            do k = 0, Nz2
                do j = 0, Ny2
                    if (Tmpr(j,k,i) /= 0.d0) then
                        print *, "Tmpr(Y=", j, ", Z=", k, ", X=", i, ") = ", Tmpr(j,k,i)
                    end if
                end do
            end do
        end do

        ! 3. Map (X,Y,Z) buffer to (Y,Z,X) global arrays
        Do k=0,Nz2; Do j=0,Ny2; Do i=0,Nx1; VMx(j,k,i) = bufX(i,j,k); EndDo; EndDo; EndDo
        Do k=0,Nz2; Do j=0,Ny1; Do i=0,Nx2; VMy(j,k,i) = bufY(i,j,k); EndDo; EndDo; EndDo
        Do k=0,Nz1; Do j=0,Ny2; Do i=0,Nx2; VMz(j,k,i) = bufZ(i,j,k); EndDo; EndDo; EndDo
        Do k=0,Nz2; Do j=0,Ny2; Do i=0,Nx2; Tmpr(j,k,i) = bufT(i,j,k); EndDo; EndDo; EndDo
        Do k=1,Nz1; Do j=1,Ny1; Do i=1,Nx1; Prs(j,k,i) = bufP(i,j,k); EndDo; EndDo; EndDo

        ! Read OLD fields (repeat buffer logic for VMxOld, VMyOld, etc.)
        Read (3, Err=55) ((( bufX(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
        Do k=0,Nz2; Do j=0,Ny2; Do i=0,Nx1; VMxOld(j,k,i) = bufX(i,j,k); EndDo; EndDo; EndDo

        Read (3, Err=55) ((( bufY(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
        Do k=0,Nz2; Do j=0,Ny1; Do i=0,Nx2; VMyOld(j,k,i) = bufY(i,j,k); EndDo; EndDo; EndDo

        Read (3, Err=55) ((( bufZ(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
        Do k=0,Nz1; Do j=0,Ny2; Do i=0,Nx2; VMzOld(j,k,i) = bufZ(i,j,k); EndDo; EndDo; EndDo

        Read (3, Err=55) ((( bufT(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)
        Do k=0,Nz2; Do j=0,Ny2; Do i=0,Nx2; TmpOld(j,k,i) = bufT(i,j,k); EndDo; EndDo; EndDo

        ! Cleanup buffers
        Deallocate(bufX, bufY, bufZ, bufT, bufP)

        Read(3) Tstart
        Iprint = - Iprint
        Ckor = 1.5D0
        Return

    Else
        ! ........ Meridional flow (Initialization) ..........................
        VMx  = 10.D-06
        VMy  = 1.D-06
        VMz  = 1.D-06
        Prs  = 1.D-06
        Tmpr = 1.D-06
    End If

    55  Continue

    VMxOld = VMx
    VMyOld = VMy
    VMzOld = VMz
    TmpOld = Tmpr
    Ckor   = 1.5D0
    Tstart = 0.d0

    Return
End Subroutine Init