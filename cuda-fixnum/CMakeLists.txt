# CMakeLists.txt for G4CU project

project(cuda-fixnum)

# required cmake version
cmake_minimum_required(VERSION 3.5)


# packages
find_package(CUDA)

include_directories ("${PROJECT_SOURCE_DIR}/src")

# nvcc flags
set(CUDA_NVCC_FLAGS
  -O3 -G -g -std=c++11
  -gencode arch=compute_60,code=sm_60
  )

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

cuda_add_library(${PROJECT_NAME} SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
#cuda_add_library(gpu STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
