cmake_minimum_required(VERSION 3.18)

# Force CMake to use the correct CUDA compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# Force host compiler
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc)
set(CMAKE_CUDA_ARCHITECTURES 86)

project(CudaBanded LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/deviceArrays/headers
        ${CMAKE_CURRENT_SOURCE_DIR}/src/BiCGSTAB
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Poisson
        /usr/local/cuda/include
)

# Collect all source files
file(GLOB_RECURSE ALL_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Remove all possible main files except testMethods.cu
list(FILTER ALL_SRC EXCLUDE REGEX ".*/main\\.cu$")
list(FILTER ALL_SRC EXCLUDE REGEX ".*/RunDirectSolver\\.cu$")

# Explicitly add the main file
set(MAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/testMethods.cu")


add_executable(testMethods
        ${ALL_SRC}
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
)

# Optional compiler flags
target_compile_options(testMethods PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
