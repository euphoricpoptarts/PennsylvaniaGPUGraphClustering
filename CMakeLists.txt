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
add_executable(leiden+ partition.cpp)
add_executable(calc_mod part_eval.cpp)
#add_executable(classify classifier_acc.cpp)

target_compile_definitions(leiden PUBLIC LEIDEN)
target_compile_definitions(leiden+ PUBLIC LEIDEN LEIDEN_PLUS)
foreach(prog louvain leiden leiden+ calc_mod)
    target_link_libraries(${prog} Kokkos::kokkos Kokkos::kokkoskernels)
endforeach()
