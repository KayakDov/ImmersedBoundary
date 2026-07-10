module AlexCudaCompatibility
    use iso_c_binding, only : C_SIZE_T
    implicit none

    public :: TemperatureHandle, VxHandle, VyHandle, VzHandle, PressureHandle, PotentialHandle
    public :: Initialize_GPU_Solvers

    integer(C_SIZE_T) :: TemperatureHandle = 0_C_SIZE_T
    integer(C_SIZE_T) :: VxHandle          = 0_C_SIZE_T
    integer(C_SIZE_T) :: VyHandle          = 0_C_SIZE_T
    integer(C_SIZE_T) :: VzHandle          = 0_C_SIZE_T
    integer(C_SIZE_T) :: PressureHandle    = 0_C_SIZE_T
    integer(C_SIZE_T) :: PotentialHandle   = 0_C_SIZE_T
    integer :: r, c, l, idx


contains

    Subroutine Initialize_GPU_Solvers()
        Use eigenbcgsolver_eigen_mod, only : init_eigen_decomp_d
        Use iso_c_binding, only : C_SIZE_T
        Use Numbers
        Use Parameters
        Use Grid
        Use Numerica
        Use Variables
        Implicit None

        logical :: bc_left_is_neumann, bc_right_is_neumann
        logical :: bc_top_is_neumann, bc_bottom_is_neumann
        logical :: bc_front_is_neumann, bc_back_is_neumann
        real(kind=8) :: bc_left_value, bc_right_value
        real(kind=8) :: bc_top_value, bc_bottom_value
        real(kind=8) :: bc_front_value, bc_back_value

        bc_left_is_neumann=.false.;  bc_right_is_neumann=.false.
        bc_top_is_neumann=.false.;   bc_bottom_is_neumann=.false.
        bc_front_is_neumann=.false.; bc_back_is_neumann=.false.
        bc_left_value=0.d0; bc_right_value=0.d0
        bc_top_value=0.d0;  bc_bottom_value=0.d0
        bc_front_value=0.d0; bc_back_value=0.d0

        ! ======================================================================
        ! AUDIT MILESTONE C: Array state immediately before passing to CUDA
        ! ======================================================================
        print *, ""
        print *, "--- [AUDIT C] Final Pre-CUDA Flattening Audit ---"
        print *, "Target layout target dimensions: Rows(Y)=", Ny1, " Cols(X)=", Nx1, " Layers(Z)=", Nz1

        block
            idx = 1
            ! Walking the structure to see how standard column-major reference treats our data
            do l = 0, int(Nz1)
                do c = 0, int(Nx1)
                    do r = 0, int(Ny1)
                        if (Tmpr(r, l, c) /= 0.d0) then
                            print *, "PreCUDA[", idx, "] (Y=", r, ", Z=", l, ", X=", c, ") = ", Tmpr(r, l, c)
                        end if
                        idx = idx + 1
                    end do
                end do
            end do
        end block
        print *, "--------------------------------------------------------"

        call assert_spacing_sizes("Temperature", HPx(0:Nx1), HPy(0:Ny1), HPz(0:Nz1), Nx1, Ny1, Nz1)
        TemperatureHandle = init_eigen_decomp_d( &
                rows=int(Ny1,C_SIZE_T), cols=int(Nx1,C_SIZE_T), layers=int(Nz1,C_SIZE_T), &
                dx=HPx(0:Nx1), dy=HPy(0:Ny1), dz=HPz(0:Nz1), &
                uniformDeltaX=.false., uniformDeltaY=.false., uniformDeltaZ=.false., &
                leftIsNeumann=bc_left_is_neumann, rightIsNeumann=bc_right_is_neumann, &
                topIsNeumann=bc_top_is_neumann, bottomIsNeumann=bc_bottom_is_neumann, &
                frontIsNeumann=bc_front_is_neumann, backIsNeumann=bc_back_is_neumann, &
                leftVal=bc_left_value, rightVal=bc_right_value, topVal=bc_top_value, bottomVal=bc_bottom_value, &
                frontVal=bc_front_value, backVal=bc_back_value, isStaggered=.false., thomas=.true. )

        call assert_spacing_sizes("Vx", HPx(0:Nx), HPy(0:Ny1), HPz(0:Nz1), Nx, Ny1, Nz1)
        VxHandle = init_eigen_decomp_d( &
                rows=int(Ny1,C_SIZE_T), cols=int(Nx,C_SIZE_T), layers=int(Nz1,C_SIZE_T), &
                dx=HPx(0:Nx), dy=HPy(0:Ny1), dz=HPz(0:Nz1), &
                uniformDeltaX=.false., uniformDeltaY=.false., uniformDeltaZ=.false., &
                leftIsNeumann=bc_left_is_neumann, rightIsNeumann=bc_right_is_neumann, &
                topIsNeumann=bc_top_is_neumann, bottomIsNeumann=bc_bottom_is_neumann, &
                frontIsNeumann=bc_front_is_neumann, backIsNeumann=bc_back_is_neumann, &
                leftVal=bc_left_value, rightVal=bc_right_value, topVal=bc_top_value, bottomVal=bc_bottom_value, &
                frontVal=bc_front_value, backVal=bc_back_value, isStaggered=.false., thomas=.true. )

        call assert_spacing_sizes("Vy", HPx(0:Nx1), HPy(0:Ny), HPz(0:Nz1), Nx1, Ny, Nz1)
        VyHandle = init_eigen_decomp_d( &
                rows=int(Ny,C_SIZE_T), cols=int(Nx1,C_SIZE_T), layers=int(Nz1,C_SIZE_T), &
                dx=HPx(0:Nx1), dy=HPy(0:Ny), dz=HPz(0:Nz1), &
                uniformDeltaX=.false., uniformDeltaY=.false., uniformDeltaZ=.false., &
                leftIsNeumann=bc_left_is_neumann, rightIsNeumann=bc_right_is_neumann, &
                topIsNeumann=bc_top_is_neumann, bottomIsNeumann=bc_bottom_is_neumann, &
                frontIsNeumann=bc_front_is_neumann, backIsNeumann=bc_back_is_neumann, &
                leftVal=bc_left_value, rightVal=bc_right_value, topVal=bc_top_value, bottomVal=bc_bottom_value, &
                frontVal=bc_front_value, backVal=bc_back_value, isStaggered=.false., thomas=.true. )

        call assert_spacing_sizes("Vz", HPx(0:Nx1), HPy(0:Ny1), HPz(0:Nz), Nx1, Ny1, Nz)
        VzHandle = init_eigen_decomp_d( &
                rows=int(Ny1,C_SIZE_T), cols=int(Nx1,C_SIZE_T), layers=int(Nz,C_SIZE_T), &
                dx=HPx(0:Nx1), dy=HPy(0:Ny1), dz=HPz(0:Nz), &
                uniformDeltaX=.false., uniformDeltaY=.false., uniformDeltaZ=.false., &
                leftIsNeumann=bc_left_is_neumann, rightIsNeumann=bc_right_is_neumann, &
                topIsNeumann=bc_top_is_neumann, bottomIsNeumann=bc_bottom_is_neumann, &
                frontIsNeumann=bc_front_is_neumann, backIsNeumann=bc_back_is_neumann, &
                leftVal=bc_left_value, rightVal=bc_right_value, topVal=bc_top_value, bottomVal=bc_bottom_value, &
                frontVal=bc_front_value, backVal=bc_back_value, isStaggered=.false., thomas=.true. )

        call assert_spacing_sizes("Pressure", HPx(0:Nx1), HPy(0:Ny1), HPz(0:Nz1), Nx1, Ny1, Nz1)
        PressureHandle = init_eigen_decomp_d( &
                rows=int(Ny1,C_SIZE_T), cols=int(Nx1,C_SIZE_T), layers=int(Nz1,C_SIZE_T), &
                dx=HPx(0:Nx1), dy=HPy(0:Ny1), dz=HPz(0:Nz1), &
                uniformDeltaX=.false., uniformDeltaY=.false., uniformDeltaZ=.false., &
                leftIsNeumann=bc_left_is_neumann, rightIsNeumann=bc_right_is_neumann, &
                topIsNeumann=bc_top_is_neumann, bottomIsNeumann=bc_bottom_is_neumann, &
                frontIsNeumann=bc_front_is_neumann, backIsNeumann=bc_back_is_neumann, &
                leftVal=bc_left_value, rightVal=bc_right_value, topVal=bc_top_value, bottomVal=bc_bottom_value, &
                frontVal=bc_front_value, backVal=bc_back_value, isStaggered=.false., thomas=.false. )

        call assert_spacing_sizes("Potential", HPx(0:Nx), HPy(0:Ny1), HPz(0:Nz1), Nx, Ny1, Nz1)
        PotentialHandle = init_eigen_decomp_d( &
                rows=int(Ny1,C_SIZE_T), cols=int(Nx,C_SIZE_T), layers=int(Nz,C_SIZE_T), &
                dx=HPx(0:Nx), dy=HPy(0:Ny1), dz=HPz(0:Nz), &
                uniformDeltaX=.false., uniformDeltaY=.false., uniformDeltaZ=.false., &
                leftIsNeumann=.true., rightIsNeumann=.true., &
                topIsNeumann=.true., bottomIsNeumann=.true., &
                frontIsNeumann=.true., backIsNeumann=.true., &
                leftVal=0.d0, rightVal=0.d0, topVal=0.d0, bottomVal=0.d0, frontVal=0.d0, backVal=0.d0, isStaggered=.false., thomas=.false. )
    End Subroutine Initialize_GPU_Solvers

    subroutine assert_spacing_sizes(name, dx, dy, dz, ncol, nrow, nlay)
        implicit none
        character(len=*), intent(in) :: name
        real(kind=8), intent(in) :: dx(:), dy(:), dz(:)
        integer, intent(in) :: ncol, nrow, nlay
        if (size(dx) /= ncol+1) error stop "dx size mismatch: "//trim(name)
        if (size(dy) /= nrow+1) error stop "dy size mismatch: "//trim(name)
        if (size(dz) /= nlay+1) error stop "dz size mismatch: "//trim(name)
    end subroutine assert_spacing_sizes

end module AlexCudaCompatibility