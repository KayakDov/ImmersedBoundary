cmake_minimum_required(VERSION 3.18)

# Force CMake to use the correct CUDA compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# Force host compiler
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc)
set(CMAKE_CUDA_ARCHITECTURES 86)

project(CudaBanded LANGUAGES CXX CUDA Fortran)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_Fortran_STANDARD 2008)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)


# Include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/deviceArrays/headers
        src/solvers/BiCGSTAB
        ${CMAKE_CURRENT_SOURCE_DIR}/src/poisson
        ${CMAKE_CURRENT_SOURCE_DIR}/shroud_project
        /usr/local/cuda/include
)

# Collect all source files
file(GLOB_RECURSE ALL_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Remove all possible main files except testMethods.cu
list(FILTER ALL_SRC EXCLUDE REGEX ".*/main\\.cu$")

# Explicitly add the main file
set(MAIN_FILE "src/immersedBoundary/YuriWork.cu")

# Collect Shroud wrapper C++ files
set(SHROUD_CPPf
        ${CMAKE_CURRENT_SOURCE_DIR}/shroud_project/wrapFortranBindings.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shroud_project/utilFortranBindings.cpp
)

add_library(CudaBandedLib STATIC
        ${ALL_SRC}
        ${SHROUD_CPPf}
        ${FORTRAN_WRAPPER}
)

add_executable(testMethods
        ${ALL_SRC}
        shroud_project/wrapFortranBindings.cpp
)



# Enable separable compilation
set_target_properties(testMethods PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA libraries
target_link_libraries(testMethods PRIVATE
        cublas
        cusolver
        cuda
        cusparse
)

target_link_libraries(testMethods PRIVATE
        CudaBandedLib
)