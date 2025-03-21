# CMakeLists.txt for C++ Extensions

cmake_minimum_required(VERSION 3.10)
project(cuda_extensions)

set(CMAKE_CXX_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/cuda_ops.cpp
)

# Create the library
cuda_add_library(cuda_ops ${SOURCES})

# Specify the output directory for the library
set_target_properties(cuda_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)