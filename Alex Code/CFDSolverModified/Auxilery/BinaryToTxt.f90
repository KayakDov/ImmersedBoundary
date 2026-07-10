Program GenerateMock3D
    Use Numbers
    Use Parameters
    Use Numerica
    Use Grid
    Use Variables

    Implicit None

    Real(kind=8), Allocatable, Dimension(:,:,:) :: bufX, bufY, bufZ, bufT, bufP
    Integer :: i, j, k, val_counter

    ! 1. Allocate using the legacy (X, Y, Z) shapes exactly as in your Init subroutine
    Allocate(bufX(0:Nx1, 0:Ny2, 0:Nz2))
    Allocate(bufY(0:Nx2, 0:Ny1, 0:Nz2))
    Allocate(bufZ(0:Nx2, 0:Ny2, 0:Nz1))
    Allocate(bufT(0:Nx2, 0:Ny2, 0:Nz2))
    Allocate(bufP(1:Nx1, 1:Ny1, 1:Nz1))

    ! Initialize everything to zero (ghost cells)
    bufX = 0.d0; bufY = 0.d0; bufZ = 0.d0; bufT = 0.d0; bufP = 0.d0

    ! 2. Populate bufT with a simple sequential matrix for a 2x3x4 inner grid
    ! We will loop through Z (layers), then Y (rows), then X (columns)
    val_counter = 1
    Do k = 1, 4      ! 4 layers deep (Z)
        Do j = 1, 2  ! 2 rows high (Y)
            Do i = 1, 3 ! 3 columns wide (X)
                bufT(i,j,k) = real(val_counter, kind=8)
                val_counter = val_counter + 1
            End Do
        End Do
    End Do

    ! Duplicate the trace sequence pattern to other variables
    bufX = bufT; bufY = bufT; bufZ = bufT

    ! Map to pressure buffer cleanly (starts indexing at 1)
    Do k=1,Nz1
        Do j=1,Ny1
            Do i=1,Nx1
                bufP(i,j,k) = bufT(i,j,k)
            EndDo
        EndDo
    EndDo

    ! 3. Write exactly to the unformatted binary structure
    Open(unit=33, file='Conv.ddd', form='unformatted', status='replace')

    Write (33) ((( bufX(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
    Write (33) ((( bufY(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
    Write (33) ((( bufZ(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
    Write (33) ((( bufT(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)
    Write (33) ((( bufP(i,j,k), i=1,Nx1), j=1,Ny1), k=1,Nz1)

    ! Write the OLD fields
    Write (33) ((( bufX(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
    Write (33) ((( bufY(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
    Write (33) ((( bufZ(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
    Write (33) ((( bufT(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)

    Write (33) 0.d0 ! Tstart

    Close(33)
    Print *, "New 2x3x4 mock binary file 'Conv.ddd' generated successfully."
End Program GenerateMock3D


!Program ConvertBinaryToText
!    Use Numbers
!    Use Parameters
!    Use Numerica
!    Use Grid
!    Use Variables
!
!    Implicit None
!
!    Real(kind=8), Allocatable, Dimension(:,:,:) :: bufX, bufY, bufZ, bufT, bufP
!    Integer :: i, j, k
!    Real(kind=8) :: Tstart_val
!
!    ! Allocate with staggered grid boundaries
!    Allocate(bufX(0:Nx1, 0:Ny2, 0:Nz2))
!    Allocate(bufY(0:Nx2, 0:Ny1, 0:Nz2))
!    Allocate(bufZ(0:Nx2, 0:Ny2, 0:Nz1))
!    Allocate(bufT(0:Nx2, 0:Ny2, 0:Nz2))
!    Allocate(bufP(1:Nx1, 1:Ny1, 1:Nz1))
!
!    Open(unit=3, file='Conv.ddd', form='unformatted', status='old')
!
!    ! Read primary fields
!    Read (3) ((( bufX(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
!    Read (3) ((( bufY(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
!    Read (3) ((( bufZ(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
!    Read (3) ((( bufT(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)
!    Read (3) ((( bufP(i,j,k), i=1,Nx1), j=1,Ny1), k=1,Nz1)
!
!    ! =======================================================
!    ! Write bufX (Vx)
!    ! =======================================================
!    Open(unit=10, file='Vx_converted.txt', status='replace')
!    Write(10, '(A)') 'X   Y   Z   Value'
!    Do k=0,Nz2
!        Do j=0,Ny2
!            Do i=0,Nx1
!                Write(10, '(I4, I4, I4, F16.6)') i, j, k, bufX(i,j,k)
!            EndDo
!        EndDo
!    EndDo
!    Close(10)
!
!    ! =======================================================
!    ! Write bufY (Vy)
!    ! =======================================================
!    Open(unit=11, file='Vy_converted.txt', status='replace')
!    Write(11, '(A)') 'X   Y   Z   Value'
!    Do k=0,Nz2
!        Do j=0,Ny1
!            Do i=0,Nx2
!                Write(11, '(I4, I4, I4, F16.6)') i, j, k, bufY(i,j,k)
!            EndDo
!        EndDo
!    EndDo
!    Close(11)
!
!    ! =======================================================
!    ! Write bufZ (Vz)
!    ! =======================================================
!    Open(unit=12, file='Vz_converted.txt', status='replace')
!    Write(12, '(A)') 'X   Y   Z   Value'
!    Do k=0,Nz1
!        Do j=0,Ny2
!            Do i=0,Nx2
!                Write(12, '(I4, I4, I4, F16.6)') i, j, k, bufZ(i,j,k)
!            EndDo
!        EndDo
!    EndDo
!    Close(12)
!
!    ! =======================================================
!    ! Write bufT (Temperature)
!    ! =======================================================
!    Open(unit=13, file='Tmpr_converted.txt', status='replace')
!    Write(13, '(A)') 'X   Y   Z   Value'
!    Do k=0,Nz2
!        Do j=0,Ny2
!            Do i=0,Nx2
!                Write(13, '(I4, I4, I4, F16.6)') i, j, k, bufT(i,j,k)
!            EndDo
!        EndDo
!    EndDo
!    Close(13)
!
!    ! =======================================================
!    ! Write bufP (Pressure)
!    ! =======================================================
!    Open(unit=14, file='Pres_converted.txt', status='replace')
!    Write(14, '(A)') 'X   Y   Z   Value'
!    Do k=1,Nz1
!        Do j=1,Ny1
!            Do i=1,Nx1
!                Write(14, '(I4, I4, I4, F16.6)') i, j, k, bufP(i,j,k)
!            EndDo
!        EndDo
!    EndDo
!    Close(14)
!
!    ! =======================================================
!    ! Finish reading the rest of the binary file to ensure
!    ! EOF isn't prematurely triggered if you need to test it
!    ! =======================================================
!    Read (3) ((( bufX(i,j,k), i=0,Nx1), j=0,Ny2), k=0,Nz2)
!    Read (3) ((( bufY(i,j,k), i=0,Nx2), j=0,Ny1), k=0,Nz2)
!    Read (3) ((( bufZ(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz1)
!    Read (3) ((( bufT(i,j,k), i=0,Nx2), j=0,Ny2), k=0,Nz2)
!    Read (3) Tstart_val
!
!    Close(3)
!
!    Print *, "Binary conversion complete."
!    Print *, "Files written: Vx, Vy, Vz, Tmpr, and Pres (_converted.txt)"
!
!End Program ConvertBinaryToText