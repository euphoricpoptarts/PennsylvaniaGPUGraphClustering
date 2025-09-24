#IMPORTANT: Use the cmake flag -DCMAKE_CXX_COMPILER=/path/to/your/nvcc_wrapper
cmake_minimum_required(VERSION 3.18)
project(jetpartition CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#ALL DEVICE BUILDS
find_package(KokkosKernels REQUIRED)
add_compile_options(-O3 -Wall -Wextra -Wshadow)
#add_compile_options(-g -DDEBUG)

add_executable(louvain partition.cpp)
add_executable(leiden partition.cpp)
add_executable(calc_mod part_eval.cpp)
#add_executable(classify classifier_acc.cpp)

target_compile_definitions(leiden PUBLIC LEIDEN)
target_link_libraries(louvain Kokkos::kokkos Kokkos::kokkoskernels)
target_link_libraries(leiden Kokkos::kokkos Kokkos::kokkoskernels)
target_link_libraries(calc_mod Kokkos::kokkos Kokkos::kokkoskernels)
