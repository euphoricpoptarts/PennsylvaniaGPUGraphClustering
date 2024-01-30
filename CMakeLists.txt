#IMPORTANT: Use the cmake flag -DCMAKE_CXX_COMPILER=/path/to/your/nvcc_wrapper
cmake_minimum_required(VERSION 3.18)
project(jetpartition CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#ALL DEVICE BUILDS
find_package(KokkosKernels REQUIRED)
add_compile_options(-O3 -Wall -Wextra -Wshadow)
#add_compile_options(-g -DDEBUG)

add_executable(lp partition.cpp)

target_compile_definitions(lp PUBLIC HASHMAP_P)
target_link_libraries(lp Kokkos::kokkos Kokkos::kokkoskernels)
