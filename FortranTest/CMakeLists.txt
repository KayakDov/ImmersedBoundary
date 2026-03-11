cmake_minimum_required(VERSION 3.18)
project(FortranCudaProject LANGUAGES Fortran CXX)

# -----------------------------
# Paths
# -----------------------------
set(BASE_LIB_DIR "/home/usr/Documents/TelAvivU/cmake-build-debug")
set(CUDA_LIB "${BASE_LIB_DIR}/libCudaBandedLib.a")
set(FORTRAN_MODULE_DIR "${BASE_LIB_DIR}/modules")

# CUDA install
set(CUDA_ROOT "/usr/local/cuda")
set(CUDA_LIB_DIR "${CUDA_ROOT}/lib64")
set(CUDA_INCLUDE_DIR "${CUDA_ROOT}/include")

# -----------------------------
# Sources and modules
# -----------------------------
set(FORTRAN_SOURCES main.f90)
set(CMAKE_Fortran_MODULE_DIRECTORY ${FORTRAN_MODULE_DIR})

add_executable(fortran_app ${FORTRAN_SOURCES})

target_include_directories(fortran_app PRIVATE
        ${FORTRAN_MODULE_DIR}
        ${CUDA_INCLUDE_DIR}        # CUDA headers
)

# -----------------------------
# Link libraries
# -----------------------------
target_link_libraries(fortran_app PRIVATE
        ${CUDA_LIB}                # your static CUDA library
        stdc++                     # C++ standard library
        ${CUDA_LIB_DIR}/libcudart.so
        ${CUDA_LIB_DIR}/libcusolver.so
        ${CUDA_LIB_DIR}/libcublas.so
        ${CUDA_LIB_DIR}/libcusparse.so
)

# Ensure C++ linker is used for CUDA objects
set_target_properties(fortran_app PROPERTIES LINKER_LANGUAGE CXX)