cmake_minimum_required(VERSION 3.24)
project(BiCGSTAB LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect all .cu sources
file(GLOB_RECURSE SOURCES src/*.cu src/*.cpp src/*.c)

# Executable target
add_executable(${PROJECT_NAME} ${SOURCES})

# Include dirs
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)

# CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Libraries
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        CUDA::cudart
        Threads::Threads
)

