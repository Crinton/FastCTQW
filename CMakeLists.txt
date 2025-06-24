cmake_minimum_required(VERSION 3.22)
project(fastexpm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17) # pybind11 通常推荐 C++17 或更高版本以获得更好的兼容性
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(Python REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(pybind11)

# file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(PYTHON_BINDING_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu)


pybind11_add_module(_fastexpm_core ${PYTHON_BINDING_SRC})



target_link_libraries(_fastexpm_core PRIVATE 
                        ${CMAKE_CURRENT_SOURCE_DIR}/lib/libexpmcore.a
                        CUDA::cublas CUDA::cusolver CUDA::cudart)
target_compile_options(_fastexpm_core PRIVATE -O3 -fPIC)
install(TARGETS _fastexpm_core DESTINATION .)