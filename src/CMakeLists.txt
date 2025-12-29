add_library(cluster STATIC)
target_sources(cluster PRIVATE
    leidenR.cpp
    contract.cpp
    local_mover.cpp
    clustering_methods.cpp
    ordering.cpp)
target_include_directories(cluster PRIVATE ${CMAKE_SOURCE_DIR}/interface)
target_link_libraries(cluster PRIVATE Kokkos::kokkos Kokkos::kokkoskernels)