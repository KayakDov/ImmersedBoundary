cmake_minimum_required(VERSION 3.18)
project(FortranCudaProject LANGUAGES Fortran CXX)

# -----------------------------
# 1. Find Dependencies
# -----------------------------
find_package(CUDAToolkit REQUIRED)

# -----------------------------
# 2. Paths & Library Discovery
# -----------------------------
# Default to the CLion/Standard build folder, but allow override
set(DEP_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cmake-build-debug" CACHE PATH "Path to CudaBandedLib")

find_library(CUDA_LIB
        NAMES CudaBandedLib
        PATHS ${DEP_LIB_DIR}
        NO_DEFAULT_PATH
)

# If the library isn't found, stop early with a helpful message
if(NOT CUDA_LIB)
    message(FATAL_ERROR "CudaBandedLib not found! Please set DEP_LIB_DIR to the folder containing the library.")
endif()

set(FORTRAN_MODULE_DIR "${DEP_LIB_DIR}/modules")

# -----------------------------
# 3. Sources and Executable
# -----------------------------
add_executable(fortran_app main.f90)

target_include_directories(fortran_app PRIVATE
        ${FORTRAN_MODULE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

# -----------------------------
# 4. Link libraries
# -----------------------------
target_link_libraries(fortran_app PRIVATE
        ${CUDA_LIB}
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
        CUDA::cudart
)

if(NOT MSVC)
    target_link_libraries(fortran_app PRIVATE stdc++)
endif()

# -----------------------------
# 5. Target Properties
# -----------------------------
set_target_properties(fortran_app PROPERTIES
        LINKER_LANGUAGE CXX
        # Required if your library has internal device-code dependencies
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)