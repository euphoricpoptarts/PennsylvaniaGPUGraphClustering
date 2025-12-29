#pragma once
#include <Kokkos_Core.hpp>
#include "memory_store.hpp"
#include "core_types.h"

namespace jet_community {

namespace ordering {
    // define internal types
    using ordinal_t = typename matrix_t::ordinal_type;
    using mem_t = memory_store;
    constexpr ordinal_t MID_CUTOFF = 32;
    constexpr ordinal_t LARGE_CUTOFF = 128;
    constexpr ordinal_t MASSIVE_CUTOFF = 15000;

    void generate_orderings(mem_t& mem, const matrix_t& g);
}

}