add_executable(plouvain cluster_main.cpp)
add_executable(pleiden cluster_main.cpp)
add_executable(pleiden+ cluster_main.cpp)
add_executable(meme memetic_clustering.cpp)
add_executable(verify_obj verify_objective.cpp)
add_library(cluster)

set(TARGETS plouvain pleiden pleiden+ verify_obj meme cluster)
set(TARGETS_FOR_LIB plouvain pleiden pleiden+ meme)
target_sources(cluster PRIVATE
    ${CMAKE_SOURCE_DIR}/src/leidenR.cpp
    ${CMAKE_SOURCE_DIR}/src/contract.cpp
    ${CMAKE_SOURCE_DIR}/src/local_mover.cpp
    ${CMAKE_SOURCE_DIR}/src/clustering_methods.cpp
    ${CMAKE_SOURCE_DIR}/src/ordering.cpp)

foreach(prog IN LISTS TARGETS)
    target_include_directories(${prog} PRIVATE ${CMAKE_SOURCE_DIR}/header ${CMAKE_SOURCE_DIR}/src)
endforeach(prog)

target_compile_definitions(pleiden PUBLIC LEIDEN)
target_compile_definitions(pleiden+ PUBLIC LEIDEN LEIDEN_PLUS)
target_compile_definitions(meme PUBLIC LEIDEN_PLUS)
foreach(prog IN LISTS TARGETS)
    target_link_libraries(${prog} Kokkos::kokkos Kokkos::kokkoskernels)
endforeach()
foreach(prog IN LISTS TARGETS_FOR_LIB)
    target_link_libraries(${prog} cluster)
endforeach(prog)