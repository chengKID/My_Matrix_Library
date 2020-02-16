cmake_minimum_required(VERSION 2.8)
project(Matrix_Operations)

set(CMAKE_BUILD_TYPE Debug)
#set(BUILD_TYPE Release)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "-std=c++11")

# required libraries
#find_package(Eigen3 REQUIRED NO_MODULE)

# configure a header file to pass some of the CMAKE settings
# to the source code
include_directories (
    ${PROJECT_SOURCE_DIR}/src
)

# sources files
set(matrix_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/matrix.cpp
    ${PROJECT_SOURCE_DIR}/src/svd.cpp
    ${PROJECT_SOURCE_DIR}/src/matrix_operations.cpp
)

# compile main library
#add_library(dso ${dso_SOURCE_FILES} ${dso_opencv_SOURCE_FILES} ${dso_pangolin_SOURCE_FILES})

# add the executable
add_executable(matrix_operations ${matrix_SOURCE_FILES})