// ***********************************************************************
// 
// Jet: Multilevel Graph Partitioning
//
// Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). 
// 
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************
#pragma once
#include <list>
#include <limits>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_HashmapAccumulator.hpp"
#include "KokkosKernels_Uniform_Initialized_MemoryPool.hpp"
#include "ExperimentLoggerUtil.hpp"
#include "memory_store.hpp"

namespace jet_community {

template<typename ordinal_t>
KOKKOS_INLINE_FUNCTION ordinal_t xorshiftHash(ordinal_t key) {
  ordinal_t x = key;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

template<class crsMat> //typename ordinal_t, typename edge_offset_t, typename scalar_t, class Device>
class contracter {
public:

    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using dyn_policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using dyn_team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using member = typename team_policy_t::member_type;
    using mem_t = memory_store<matrix_t, part_t>;
    static constexpr ordinal_t get_null_val() {
        // this value must line up with the null value used by the hashmap
        // accumulator
        if (std::is_signed<ordinal_t>::value) {
            return -1;
        } else {
            return std::numeric_limits<ordinal_t>::max();
        }
    }
    static constexpr ordinal_t ORD_MAX  = get_null_val();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;
    // contains matrix and vertex weights corresponding to current level
    // interp matrix maps previous level to this level
    struct coarse_level_triple {
        matrix_t mtx;
        wgt_view_t wdeg;
    };

    // define behavior-controlling enums
    enum Heuristic { HECv1, HECv2, HECv3, Match, MtMetis };

    // internal parameters and data
    // default heuristic is MtMetis
    Heuristic h = MtMetis;
    ordinal_t coarse_vtx_cutoff = 1000;
    ordinal_t min_allowed_vtx = 250;
    unsigned int max_levels = 200;

bool should_use_dyn(const ordinal_t n, const Kokkos::View<const edge_offset_t*, Device> work, int t_count){
    bool use_dyn = false;
    edge_offset_t max = 0;
    edge_offset_t min = std::numeric_limits<edge_offset_t>::max();
    if(is_host_space){
        ordinal_t static_size = (n + t_count) / t_count;
        for(ordinal_t i = 0; i < t_count; i++){
            ordinal_t start = i * static_size;
            ordinal_t end = start + static_size;
            if(start > n) start = n;
            if(end > n) end = n;
            edge_offset_t size = work(end) - work(start);
            if(size > max){
                max = size;
            }
            if(size < min) {
                min = size;
            }
        }
        //printf("min size: %i, max size: %i\n", min, max);
        if(n > 500000 && max > 5*min){
            use_dyn = true;
        }
    }
    return use_dyn;
}

struct countingFunctor {

    matrix_t g;
    vtx_view_t vcmap;
    edge_view_t degree_initial;
    ordinal_t workLength;

    countingFunctor(matrix_t _g,
            vtx_view_t _vcmap,
            edge_view_t _degree_initial) :
        g(_g),
        vcmap(_vcmap),
        degree_initial(_degree_initial),
        workLength(_g.numRows()) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t& i) const 
    {
        ordinal_t u = vcmap(i);
        edge_offset_t start = g.graph.row_map(i);
        edge_offset_t end = g.graph.row_map(i + 1);
        ordinal_t nonLoopEdgesTotal = end - start;
        Kokkos::atomic_add(&degree_initial(u), nonLoopEdgesTotal);
    }
};

struct combineAndDedupe {
    matrix_t g;
    vtx_view_t vcmap;
    vtx_view_t htable;
    wgt_view_t hvals;
    edge_view_t hrow_map;
    edge_view_t counts;
    vtx_view_t vtx;

    combineAndDedupe(matrix_t _g,
            vtx_view_t _vcmap,
            vtx_view_t _htable,
            wgt_view_t _hvals,
            edge_view_t _hrow_map,
            edge_view_t _counts,
            vtx_view_t _vtx) :
            g(_g),
            vcmap(_vcmap),
            htable(_htable),
            hvals(_hvals),
            hrow_map(_hrow_map),
            counts(_counts),
            vtx(_vtx) {}

    KOKKOS_INLINE_FUNCTION
        edge_offset_t insert(const edge_offset_t& hash_start, const edge_offset_t& size, const ordinal_t& u, const ordinal_t& i) const {
            edge_offset_t offset = abs(xorshiftHash<ordinal_t>(u)) % size;
            while(true){
                if(htable(hash_start + offset) == -1){
                    if(Kokkos::atomic_compare_exchange(&htable(hash_start + offset), -1, u) == -1){
                        Kokkos::atomic_add(&counts(i), 1);
                    }
                }
                if(htable(hash_start + offset) == u){
                    return offset;
                } else {
                    offset++;
                    if(offset >= size) offset -= size;
                }
            }
        }

    KOKKOS_INLINE_FUNCTION
        void operator()(const member& thread) const
    {
        const ordinal_t x = vtx(thread.league_rank());
        const ordinal_t i = vcmap(x);
        const edge_offset_t start = g.graph.row_map(x);
        const edge_offset_t end = g.graph.row_map(x + 1);
        const edge_offset_t hash_start = hrow_map(i);
        const edge_offset_t size = hrow_map(i + 1) - hash_start;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
            ordinal_t u = vcmap(g.graph.entries(j));
            if(i != u){
                edge_offset_t offset = insert(hash_start, size, u, i);
                Kokkos::atomic_add(&hvals(hash_start + offset), g.values(j));
            }
        });
    }

    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t& xx) const
    {
        const ordinal_t x = vtx(xx);
        const ordinal_t i = vcmap(x);
        const edge_offset_t start = g.graph.row_map(x);
        const edge_offset_t end = g.graph.row_map(x + 1);
        const edge_offset_t hash_start = hrow_map(i);
        const edge_offset_t size = hrow_map(i + 1) - hash_start;
        for(edge_offset_t j = start; j < end; j++){
            ordinal_t u = vcmap(g.graph.entries(j));
            if(i != u){
                edge_offset_t offset = insert(hash_start, size, u, i);
                Kokkos::atomic_add(&hvals(hash_start + offset), g.values(j));
            }
        }
    }
};

struct consolidateUnique {
    vtx_view_t htable, entries_coarse;
    wgt_view_t hvals, wgts_coarse;

    consolidateUnique(vtx_view_t _htable,
            vtx_view_t _entries_coarse,
            wgt_view_t _hvals,
            wgt_view_t _wgts_coarse) :
            htable(_htable),
            entries_coarse(_entries_coarse),
            hvals(_hvals),
            wgts_coarse(_wgts_coarse) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const edge_offset_t j, edge_offset_t& insert, const bool final) const
    {
        if(htable(j) > -1){
            if(final){
                entries_coarse(insert) = htable(j);
                wgts_coarse(insert) = hvals(j);
            }
            insert++;
        }
    }
};

coarse_level_triple build_coarse_graph(const coarse_level_triple level,
    const vtx_view_t vcmap,
    const ordinal_t nc,
    mem_t& mem,
    ExperimentLoggerUtil<scalar_t>& experiment) {

    matrix_t g = level.mtx;
    ordinal_t n = g.numRows();

    Kokkos::Timer timer;
    edge_view_t hrow_map = Kokkos::subview(mem.cd_mem.conn_offsets, std::make_pair((ordinal_t)0, nc + 1));
    Kokkos::deep_copy(exec_space(), hrow_map, 0);
    countingFunctor countF(g, vcmap, hrow_map);
    Kokkos::parallel_for("count edges per coarse vertex (also compute coarse vertex weights)", policy_t(0, n), countF);
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Count, timer.seconds());
    timer.reset();
    edge_offset_t hash_size = 0;
    //exclusive prefix sum
    Kokkos::parallel_scan("scan offsets", policy_t(0, nc + 1), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        edge_offset_t val = hrow_map(i);
        if(final){
            hrow_map(i) = update;
        }
        update += val;
    }, hash_size);
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Prefix, timer.seconds());
    timer.reset();
    vtx_view_t htable = Kokkos::subview(mem.cd_mem.conn_entries, std::make_pair((edge_offset_t)0, hash_size));
    Kokkos::deep_copy(exec_space(), htable, -1);
    wgt_view_t hvals = Kokkos::subview(mem.cd_mem.conn_vals, std::make_pair((edge_offset_t)0, hash_size));
    Kokkos::deep_copy(exec_space(), hvals, 0);
    //insert each coarse vertex into a bucket determined by a hash
    //use linear probing to resolve conflicts
    //combine weights using atomic addition
    edge_view_t coarse_row_map_f("edges_per_source", nc + 1);
    ordinal_t low = 0, high = 0;
    ordinal_t limit = 32;
    vtx_view_t vtx_scratch = mem.s_mem.vtx1;
    Kokkos::parallel_scan("compact high degree", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t degree = g.graph.row_map(i+1) - g.graph.row_map(i);
        if(degree >= limit){
            if(final){
                vtx_scratch(update) = i;
            }
            update++;
        }
    }, high);
    Kokkos::parallel_scan("compact low degree", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t degree = g.graph.row_map(i+1) - g.graph.row_map(i);
        if(degree < limit){
            if(final){
                vtx_scratch(high + update) = i;
            }
            update++;
        }
    }, low);
    combineAndDedupe cnd(g, vcmap, htable, hvals, hrow_map, coarse_row_map_f, vtx_scratch);
    Kokkos::parallel_for("deduplicate", team_policy_t(high, Kokkos::AUTO), cnd);
    Kokkos::parallel_for("deduplicate", policy_t(high, high + low), cnd);
    Kokkos::fence();
    experiment.addMeasurement(Measurement::Dedupe, timer.seconds());
    timer.reset();
    edge_offset_t old_size = hash_size;
    Kokkos::parallel_scan("scan offsets", policy_t(0, nc + 1), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        edge_offset_t val = coarse_row_map_f(i);
        if(final){
            coarse_row_map_f(i) = update;
        }
        update += val;
    }, hash_size);
    vtx_view_t entries_coarse(Kokkos::ViewAllocateWithoutInitializing("coarse entries"), hash_size);
    wgt_view_t wgts_coarse(Kokkos::ViewAllocateWithoutInitializing("coarse weights"), hash_size);
    consolidateUnique consolidate(htable, entries_coarse, hvals, wgts_coarse);
    Kokkos::parallel_scan("consolidate", policy_t(0, old_size), consolidate);
    graph_type gc_graph(entries_coarse, coarse_row_map_f);
    matrix_t gc("gc", nc, wgts_coarse, gc_graph);
    coarse_level_triple next_level;
    next_level.mtx = gc;
    Kokkos::fence();
    experiment.addMeasurement(Measurement::WriteGraph, timer.seconds());
    timer.reset();
    return next_level;
}

void set_heuristic(Heuristic _h) {
    this->h = _h;
}

void set_coarse_vtx_cutoff(ordinal_t _coarse_vtx_cutoff) {
    this->coarse_vtx_cutoff = _coarse_vtx_cutoff;
}

void set_min_allowed_vtx(ordinal_t _min_allowed_vtx) {
    this->min_allowed_vtx = _min_allowed_vtx;
}

void set_max_levels(unsigned int _max_levels) {
    this->max_levels = _max_levels;
}

};

}
