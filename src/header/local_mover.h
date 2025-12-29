#pragma once
#include <Kokkos_Core.hpp>
#include "memory_store.hpp"
#include "cluster_data.hpp"
#include "weighted_graph.h"

namespace jet_community {

namespace local_move_heuristic {

    // define internal types
    using exec_space = typename matrix_t::execution_space;
    using ordinal_t = typename matrix_t::ordinal_type;
    using vtx_vt = Kokkos::View<ordinal_t*, exec_space>;
    using refine_data = cluster_data;
    using mem_t = memory_store;
    using wg_t = weighted_graph;

    template <bool uniform, bool constrained>
    void local_move(const wg_t wg, vtx_vt best_part, refine_data& best_state, bool is_initial, mem_t& mem, vtx_vt constraint);

    template <bool uniform, bool constrained>
    void local_move_strict(const wg_t wg, vtx_vt best_part, refine_data& best_state, bool is_initial, mem_t& mem, vtx_vt constraint);

}

}