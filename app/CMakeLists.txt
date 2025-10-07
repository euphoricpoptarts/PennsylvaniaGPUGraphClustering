add_executable(plouvain partition.cpp)
add_executable(pleiden partition.cpp)
add_executable(pleiden+ partition.cpp)
add_executable(calc_mod part_eval.cpp)

foreach(prog plouvain pleiden pleiden+ calc_mod)
    target_include_directories(${prog} PRIVATE ${CMAKE_SOURCE_DIR}/header ${CMAKE_SOURCE_DIR}/src)
endforeach(prog)

target_compile_definitions(pleiden PUBLIC LEIDEN)
target_compile_definitions(pleiden+ PUBLIC LEIDEN LEIDEN_PLUS)
foreach(prog plouvain pleiden pleiden+ calc_mod)
    target_link_libraries(${prog} Kokkos::kokkos Kokkos::kokkoskernels)
endforeach()
