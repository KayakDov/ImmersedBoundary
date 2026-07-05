Subroutine Initialize_GPU_Solvers()
    Use AlexCudaCompatibility
    Use eigenbcgsolver_eigen_mod, only : init_eigen_decomp_d
    Use iso_c_binding, only : C_SIZE_T

    ! ACTIVATE THESE so the subroutine knows what Nx1, dx_array, etc., are:
    Use Parameters
    Use Grid
    ! Use BoundaryConditions (or whichever module holds BC_Left_Neumann_Temp, etc.)

    Implicit None

    !-------------------------------------------------------------------------
    ! 1. TEMPERATURE (Centered Grid)
    !-------------------------------------------------------------------------
    TemperatureHandle = init_eigen_decomp_d( &
            rows            = int(Ny1, C_SIZE_T), &
            cols            = int(Nx1, C_SIZE_T), &
            layers          = int(Nz1, C_SIZE_T), &
            dx              = HPy(0:Ny1), &
            dy              = HPx(0:Nx1), &
            dz              = HPz(0:Nz1), &
            uniformDeltaX   = .false., &
            uniformDeltaY   = .false., &
            uniformDeltaZ   = .false., &
            leftIsNeumann   = BC_Left_Neumann_Temp, &
            rightIsNeumann  = BC_Right_Neumann_Temp, &
            topIsNeumann    = BC_Top_Neumann_Temp, &
            bottomIsNeumann = BC_Bottom_Neumann_Temp, &
            frontIsNeumann  = BC_Front_Neumann_Temp, &
            backIsNeumann   = BC_Back_Neumann_Temp, &
            leftVal         = BC_Left_Val_Temp, &
            rightVal        = BC_Right_Val_Temp, &
            topVal          = BC_Top_Val_Temp, &
            bottomVal       = BC_Bottom_Val_Temp, &
            frontVal        = BC_Front_Val_Temp, &
            backVal         = BC_Back_Val_Temp, &
            isStaggered     = .false., &
            thomas          = .true. &
            )

    !-------------------------------------------------------------------------
    ! 2. X-VELOCITY (Boundary distances baked into dx_array_Vx)
    !-------------------------------------------------------------------------
    VxHandle = init_eigen_decomp_d( &
            rows            = int(Ny, C_SIZE_T), &
            cols            = int(Nx1, C_SIZE_T), &
            layers          = int(Nz1, C_SIZE_T), &
            dx              = HPy(0:Ny1), &
            dy              = HPx(0:Nx1), &
            dz              = HPz(0:Nz1), &
            uniformDeltaX   = .false., &
            uniformDeltaY   = .false., &
            uniformDeltaZ   = .false., &
            leftIsNeumann   = BC_Left_Neumann_Vx, &
            rightIsNeumann  = BC_Right_Neumann_Vx, &
            topIsNeumann    = BC_Top_Neumann_Vx, &
            bottomIsNeumann = BC_Bottom_Neumann_Vx, &
            frontIsNeumann  = BC_Front_Neumann_Vx, &
            backIsNeumann   = BC_Back_Neumann_Vx, &
            leftVal         = BC_Left_Val_Vx, &
            rightVal        = BC_Right_Val_Vx, &
            topVal          = BC_Top_Val_Vx, &
            bottomVal       = BC_Bottom_Val_Vx, &
            frontVal        = BC_Front_Val_Vx, &
            backVal         = BC_Back_Val_Vx, &
            isStaggered     = .false., &
            thomas          = .true. &
            )

    !-------------------------------------------------------------------------
    ! 3. Y-VELOCITY (Boundary distances baked into dy_array_Vy)
    !-------------------------------------------------------------------------
    VyHandle = init_eigen_decomp_d( &
            rows            = int(Ny1, C_SIZE_T), &
            cols            = int(Nx, C_SIZE_T), &
            layers          = int(Nz1, C_SIZE_T), &
            dx              = HPy(0:Ny1), &
            dy              = HPx(0:Nx1), &
            dz              = HPz(0:Nz1), &
            uniformDeltaX   = .false., &
            uniformDeltaY   = .false., &
            uniformDeltaZ   = .false., &
            leftIsNeumann   = BC_Left_Neumann_Vy, &
            rightIsNeumann  = BC_Right_Neumann_Vy, &
            topIsNeumann    = BC_Top_Neumann_Vy, &
            bottomIsNeumann = BC_Bottom_Neumann_Vy, &
            frontIsNeumann  = BC_Front_Neumann_Vy, &
            backIsNeumann   = BC_Back_Neumann_Vy, &
            leftVal         = BC_Left_Val_Vy, &
            rightVal        = BC_Right_Val_Vy, &
            topVal          = BC_Top_Val_Vy, &
            bottomVal       = BC_Bottom_Val_Vy, &
            frontVal        = BC_Front_Val_Vy, &
            backVal         = BC_Back_Val_Vy, &
            isStaggered     = .false., &
            thomas          = .true. &
            )

    !-------------------------------------------------------------------------
    ! 4. Z-VELOCITY (Boundary distances baked into dz_array_Vz)
    !-------------------------------------------------------------------------
    VzHandle = init_eigen_decomp_d( &
            rows            = int(Ny1, C_SIZE_T), &
            cols            = int(Nx1, C_SIZE_T), &
            layers          = int(Nz, C_SIZE_T), &
            dx              = HPy(0:Ny1), &
            dy              = HPx(0:Nx1), &
            dz              = HPz(0:Nz1), &
            uniformDeltaX   = .false., &
            uniformDeltaY   = .false., &
            uniformDeltaZ   = .false., &
            leftIsNeumann   = BC_Left_Neumann_Vz, &
            rightIsNeumann  = BC_Right_Neumann_Vz, &
            topIsNeumann    = BC_Top_Neumann_Vz, &
            bottomIsNeumann = BC_Bottom_Neumann_Vz, &
            frontIsNeumann  = BC_Front_Neumann_Vz, &
            backIsNeumann   = BC_Back_Neumann_Vz, &
            leftVal         = BC_Left_Val_Vz, &
            rightVal        = BC_Right_Val_Vz, &
            topVal          = BC_Top_Val_Vz, &
            bottomVal       = BC_Bottom_Val_Vz, &
            frontVal        = BC_Front_Val_Vz, &
            backVal         = BC_Back_Val_Vz, &
            isStaggered     = .false., &
            thomas          = .true. &
            )

    !-------------------------------------------------------------------------
    ! 5. PRESSURE (Centered Grid)
    !-------------------------------------------------------------------------
    PressureHandle = init_eigen_decomp_d( &
            rows            = int(Ny1, C_SIZE_T), &
            cols            = int(Nx1, C_SIZE_T), &
            layers          = int(Nz1, C_SIZE_T), &
            dx              = HPy(0:Ny1), &
            dy              = HPx(0:Nx1), &
            dz              = HPz(0:Nz1), &
            uniformDeltaX   = .false., &
            uniformDeltaY   = .false., &
            uniformDeltaZ   = .false., &
            leftIsNeumann   = BC_Left_Neumann_P, &
            rightIsNeumann  = BC_Right_Neumann_P, &
            topIsNeumann    = BC_Top_Neumann_P, &
            bottomIsNeumann = BC_Bottom_Neumann_P, &
            frontIsNeumann  = BC_Front_Neumann_P, &
            backIsNeumann   = BC_Back_Neumann_P, &
            leftVal         = BC_Left_Val_P, &
            rightVal        = BC_Right_Val_P, &
            topVal          = BC_Top_Val_P, &
            bottomVal       = BC_Bottom_Val_P, &
            frontVal        = BC_Front_Val_P, &
            backVal         = BC_Back_Val_P, &
            isStaggered     = .false., &
            thomas          = .false. &
            )

    PotentialHandle = init_eigen_decomp_d( &
            rows            = int(Ny1, C_SIZE_T), &
            cols            = int(Nx, C_SIZE_T), &
            layers          = int(Nz1, C_SIZE_T), &
            dx              = HPy(0:Ny1), &
            dy              = HPx(0:Nx1), &
            dz              = HPz(0:Nz1), &
            uniformDeltaX   = .false., &
            uniformDeltaY   = .false., &
            uniformDeltaZ   = .false., &
            leftIsNeumann   = ?, &
            rightIsNeumann  = ?, &
            topIsNeumann    = ?, &
            bottomIsNeumann = ?, &
            frontIsNeumann  = ?, &
            backIsNeumann   = ?, &
            leftVal         = 0.d0, &
            rightVal        = 0.d0, &
            topVal          = 0.d0, &
            bottomVal       = 0.d0, &
            frontVal        = 0.d0, &
            backVal         = 0.d0, &
            isStaggered     = .false., &
            thomas          = .false. &
    )

End Subroutine Initialize_GPU_Solvers
