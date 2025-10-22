#pragma once
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"

namespace jet_community {

// combines a crs matrix/graph with vertex weights
template <typename matrix_t>
struct weighted_graph {
    using scalar_t = typename matrix_t::value_type;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;

    matrix_t mtx;
    wgt_vt vtx_w;
    bool edge_uniform = true;
};

}