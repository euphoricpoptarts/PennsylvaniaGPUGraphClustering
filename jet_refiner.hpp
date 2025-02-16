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
#include <type_traits>
#include <limits>
#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>
#include "Kokkos_Bitset.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "ExperimentLoggerUtil.hpp"
#include "part_stat.hpp"

namespace jet_partitioner {

template<class crsMat, typename part_t>
class jet_refiner {
public:

    //helper for getting gain_t
    template<typename T>
    struct type_identity {
        typedef T type;
    };

    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    // need some trickery because make_signed is undefined for floating point types
    using gain_t = typename std::conditional_t<std::is_signed_v<scalar_t>, type_identity<scalar_t>, std::make_signed<scalar_t>>::type;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using vtx_svt = Kokkos::View<ordinal_t, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using gain_vt = Kokkos::View<gain_t*, Device>;
    using gain_svt = Kokkos::View<gain_t, Device>;
    using vtx_pin_st = Kokkos::View<ordinal_t, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_vt = Kokkos::View<gain_t*, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_st = Kokkos::View<gain_t, Kokkos::SharedHostPinnedSpace>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using part_svt = Kokkos::View<part_t, Device>;
    using obj_vt = Kokkos::View<float*, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using dyn_policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using dyn_team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using member = typename team_policy_t::member_type;
    using stat = part_stat<matrix_t, part_t>;
    using refine_data = typename stat::refine_data;
    using argmax_reducer_t = Kokkos::MaxFirstLoc<float, edge_offset_t, Device>;
    using argmax_t = typename argmax_reducer_t::value_type;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    static constexpr gain_t GAIN_MIN = std::numeric_limits<gain_t>::lowest();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;
    static constexpr part_t NULL_PART = -1;
    static constexpr part_t HASH_RECLAIM = -2;

    static const ordinal_t max_sections = 128;
    static const int max_buckets = 50;
    static const int mid_bucket = 25;

    static KOKKOS_INLINE_FUNCTION uint32_t hash(uint32_t x) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

struct problem {
    matrix_t g;
    wgt_view_t vtx_w;
    wgt_view_t wdeg;
    wgt_view_t nb_self_loops;
    part_vt constraint;
    ordinal_t opt;
    ordinal_t size_max;
    bool use_team = true;
};

//vertex-part connectivity data
struct conn_data {
    gain_vt conn_vals;
    gain_vt pvals, bvals;
    edge_view_t conn_offsets;
    vtx_view_t lock_bit;
    part_vt dest_cache;
    part_vt conn_entries;
    part_vt conn_table_sizes;
};

//this struct contains all the scratch memory used by the refinement iterations
struct scratch_mem {
    gain_vt gain1;
    obj_vt obj1, gain_persistent;
    vtx_view_t vtx1, vtx2, vtx3, zeros1;
    part_vt dest_part, undersized;
    vtx_pin_st scan_host;
    gain_pin_st cut_change1, cut_change2, max_part;
    gain_pin_vt reduce_locs;
    typename gain_vt::HostMirror reduce_copy;

    scratch_mem(const ordinal_t n) {
        gain1 = gain_vt(Kokkos::ViewAllocateWithoutInitializing("gain scratch 1"), n);
        obj1 = obj_vt(Kokkos::ViewAllocateWithoutInitializing("obj scratch 1"), n);
        gain_persistent = obj_vt(Kokkos::ViewAllocateWithoutInitializing("gain persistent"), n);
        vtx1 = vtx_view_t(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 1"), n);
        vtx2 = vtx_view_t(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 2"), n);
        vtx3 = vtx_view_t(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 3"), n);
        dest_part = part_vt(Kokkos::ViewAllocateWithoutInitializing("destination scratch"), n);
        zeros1 = vtx_view_t("zeros 1", n);
        scan_host = vtx_pin_st("scan host");
        reduce_locs = gain_pin_vt("reduce to here", 3);
        reduce_copy = Kokkos::create_mirror_view(reduce_locs);
        cut_change1 = Kokkos::subview(reduce_locs, 0);
        cut_change2 = Kokkos::subview(reduce_locs, 1);
        max_part = Kokkos::subview(reduce_locs, 2);
    }
};

    scratch_mem perm_scratch;
    conn_data perm_cdata;

    //find maximum size for conn_entries and conn_vals
    edge_offset_t count_gain_size(const matrix_t largest){
        edge_offset_t gain_size = 0;
        Kokkos::parallel_reduce("comp offsets", policy_t(0, largest.numRows()), KOKKOS_LAMBDA(const ordinal_t& i, edge_offset_t& update){
            ordinal_t degree = largest.graph.row_map(i + 1) - largest.graph.row_map(i);
            update += degree;
        }, gain_size);
        return gain_size;
    }

    jet_refiner(const matrix_t largest) :
        perm_scratch(largest.numRows()) {
        ordinal_t n = largest.numRows();
        edge_view_t conn_offsets("gain offsets", n + 1);
        edge_offset_t gain_size = count_gain_size(largest);
        perm_cdata.conn_vals = gain_vt(Kokkos::ViewAllocateWithoutInitializing("conn vals"), gain_size);
        perm_cdata.pvals = gain_vt(Kokkos::ViewAllocateWithoutInitializing("p vals"), n);
        perm_cdata.bvals = gain_vt(Kokkos::ViewAllocateWithoutInitializing("b vals"), n);
        perm_cdata.conn_entries = part_vt(Kokkos::ViewAllocateWithoutInitializing("conn entries"), gain_size);
        perm_cdata.conn_offsets = conn_offsets;
        perm_cdata.dest_cache = part_vt(Kokkos::ViewAllocateWithoutInitializing("best connected part for each vertex"), n);
        perm_cdata.conn_table_sizes = part_vt(Kokkos::ViewAllocateWithoutInitializing("map size"), n);
        perm_cdata.lock_bit = vtx_view_t("lock bit", n);
    }

void copy_refine_data(refine_data& lhs, refine_data& rhs){
    Kokkos::deep_copy(exec_space(), lhs.in_deg, rhs.in_deg);
    Kokkos::deep_copy(exec_space(), lhs.total_deg, rhs.total_deg);
    lhs.g_deg = rhs.g_deg;
    lhs.cut = rhs.cut;
    lhs.init = rhs.init;
    lhs.mod = rhs.mod;
    lhs.label_count = rhs.label_count;
}

refine_data clone_refine_data(refine_data& rhs){
    refine_data clone;
    clone.in_deg = gain_vt(Kokkos::ViewAllocateWithoutInitializing("internal degree of clusters"), rhs.in_deg.extent(0));
    clone.total_deg = gain_vt(Kokkos::ViewAllocateWithoutInitializing("total degree of clusters"), rhs.total_deg.extent(0));
    copy_refine_data(clone, rhs);
    return clone;
}

//determines which vertices (if any) should be moved to another part to decrease cutsize
//8 kernels, 2 device-host syncs
vtx_view_t jet_lp(const problem& prob, const part_vt& part, const refine_data& rfd, const conn_data& cdata, scratch_mem& scratch, float filter_ratio, bool initial){
    const matrix_t& g = prob.g;
    ordinal_t n = g.numRows();
    ordinal_t num_pos = 0;
    part_vt dest_part = scratch.dest_part;
    part_vt conn_entries = cdata.conn_entries;
    edge_view_t conn_offsets = cdata.conn_offsets;
    gain_vt conn_vals = cdata.conn_vals;
    obj_vt save_gains = scratch.gain_persistent;
    vtx_view_t lock_bit = cdata.lock_bit;
    gain_vt total_deg = rfd.total_deg;
    gain_vt wdeg = prob.wdeg;
    float inv_2m = 1.0 / static_cast<float>(rfd.g_deg);
    ordinal_t cutoff = 128;
    Kokkos::parallel_scan("filter out locked and find large tables", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        part_t cache = cdata.dest_cache(i);
        if(cache == NULL_PART && lock_bit(i) == 0 && cdata.conn_table_sizes(i) > cutoff){
            if(final){
                scratch.vtx2(update) = i;
            }
            update++;
        } else if(final){
            if(lock_bit(i)) {
                dest_part(i) = part(i);
            } else if(cache != NULL_PART) {
                dest_part(i) = cache;
            }
        }
    }, num_pos);
    vtx_view_t large_tables = Kokkos::subview(scratch.vtx2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    Kokkos::parallel_for("select destination part (small tables)", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        if(!(cdata.dest_cache(i) == NULL_PART && lock_bit(i) == 0) || cdata.conn_table_sizes(i) > cutoff){
            return;
        }
        part_t p = part(i);
        part_t best = p;
        float wd = wdeg(i);
        float b_conn = -100000.0;
        gain_t bval = 0;
        float multi = wd*inv_2m;
        edge_offset_t start = conn_offsets(i);
        part_t size = cdata.conn_table_sizes(i);
        edge_offset_t end = start + size;
        //finds potential destination as most connected part excluding p
        for(edge_offset_t j = start; j < end; j++){
            float j_conn = conn_vals(j);
            if(j_conn > 0){
                part_t px = conn_entries(j);
                j_conn -= static_cast<float>(total_deg(px))*multi;
                if(j_conn > b_conn){
                    b_conn = j_conn;
                    bval = conn_vals(j);
                    best = px;
                }
            }
        }
        save_gains(i) = 0;
        if(best != p){
            float p_conn = cdata.pvals(i) - (total_deg(p) - wd)*multi;
            float limit = p_conn - filter_ratio*(p_conn);
            // vertices must pass this filter in order to be considered further
            // b_conn >= p_conn may seem redundant but it is important
            // to address an edge case where floor(filter_ratio*p_conn) rounds to zero
            if(b_conn >= p_conn || (b_conn >= limit)){
                save_gains(i) = b_conn - p_conn;
                cdata.bvals(i) = bval;
            } else {
                best = p;
            }
        }
        cdata.dest_cache(i) = best;
        //a vertex is not considered further if best == p
        dest_part(i) = best;
    });
    Kokkos::parallel_for("select destination part (large tables)", team_policy_t(num_pos, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = large_tables(t.league_rank());
        float wd = wdeg(i);
        float multi = wd*inv_2m;
        edge_offset_t start = conn_offsets(i);
        part_t size = cdata.conn_table_sizes(i);
        edge_offset_t end = start + size;
        argmax_t am{-100000.0, end};
        //finds potential destination as most connected part excluding p
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, start, end), [=](const edge_offset_t j, argmax_t& local){
            float j_conn = conn_vals(j);
            if(j_conn > 0){
                part_t px = conn_entries(j);
                j_conn -= static_cast<float>(total_deg(px))*multi;
                if(j_conn > local.val){
                    local.val = j_conn;
                    local.loc = j;
                }
            }
        }, argmax_reducer_t(am));
        t.team_barrier();
        Kokkos::single(Kokkos::PerTeam(t), [=](){
            save_gains(i) = 0;
            part_t p = part(i);
            part_t best = p;
            if(am.loc >= start && am.loc < end){
                float b_conn = am.val;
                best = conn_entries(am.loc);
                float p_conn = cdata.pvals(i) - (total_deg(p) - wd)*multi;
                float limit = p_conn - filter_ratio*(p_conn);
                // vertices must pass this filter in order to be considered further
                // b_conn >= p_conn may seem redundant but it is important
                // to address an edge case where floor(filter_ratio*p_conn) rounds to zero
                if(b_conn >= p_conn || (b_conn >= limit)){
                    save_gains(i) = b_conn - p_conn;
                    cdata.bvals(i) = conn_vals(am.loc);
                } else {
                    best = p;
                }
            }
            cdata.dest_cache(i) = best;
            //a vertex is not considered further if best == p
            dest_part(i) = best;
        });
    });
    //need to store the pre-afterburn gains into a separate view
    //than savegains, because we write new values into it that may not be overwritten
    //if a vertex has its best neighbor cached
    obj_vt pregain = scratch.obj1;
    //write all unlocked vertices that passed the above filter into an unordered list
    //output count of such vertices into num_pos
    vtx_view_t swap_scratch = scratch.vtx3;
    Kokkos::parallel_scan("filter potentially viable moves", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        part_t p = part(i);
        part_t best = dest_part(i);
        if(p != best && lock_bit(i) == 0){
            if(final){
                swap_scratch(update) = i;
                pregain(i) = save_gains(i);
            }
            update++;
        } else if(final){
            pregain(i) = -100000.0;
            lock_bit(i) = 0;
        }
    }, scratch.scan_host);
    exec_space().fence();
    num_pos = scratch.scan_host();
    //truncate scratch views by num_pos
    vtx_view_t pos_moves = Kokkos::subview(swap_scratch, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    //in this kernel every potential move from the previous filters
    //is reevaluated by considering the effect of the other potential moves
    //a move is considered to occur before another according to their potential gains
    //and the vertex ids
    ordinal_t big = 0;
    cutoff = 128;
    Kokkos::parallel_scan("filter out locked and find large tables", policy_t(0, num_pos), KOKKOS_LAMBDA(const ordinal_t x, ordinal_t& update, const bool final){
        ordinal_t i = pos_moves(x);
        if(g.graph.row_map(i+1) - g.graph.row_map(i) > cutoff){
            if(final){
                scratch.vtx1(update) = i;
            }
            update++;
        } else if(final){
            scratch.vtx2(x - update) = i;
        }
    }, big);
    vtx_view_t big_rows = Kokkos::subview(scratch.vtx1, std::make_pair(static_cast<ordinal_t>(0), big));
    vtx_view_t small_rows = Kokkos::subview(scratch.vtx2, std::make_pair(static_cast<ordinal_t>(0), num_pos - big));
    Kokkos::parallel_for("afterburner heuristic", team_policy_t(big, 256), KOKKOS_LAMBDA(const member& t){
        float change = 0;
        ordinal_t i = big_rows(t.league_rank());
        part_t best = dest_part(i);
        part_t p = part(i);
        float wd = prob.wdeg(i);
        float multi = wd*inv_2m;
        float igain = pregain(i);
        ordinal_t hi = hash(i);
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&](const edge_offset_t j, float& update){
            ordinal_t v = g.graph.entries(j);
            float vgain = pregain(v);
            //adjust local gain if v has higher priority than i
            if((vgain - igain) >= 0.1 || (abs(vgain - igain) < 0.1 && static_cast<ordinal_t>(hash(v)) < hi)){
                part_t vpart = dest_part(v);
                scalar_t wgt = g.values(j);
                float q = static_cast<float>(wgt) - multi*prob.wdeg(v);
                update -= (vpart == p) ? q : 0;
                update += (vpart == best) ? q : 0;
                vpart = part(v);
                update += (vpart == p) ? q : 0;
                update -= (vpart == best) ? q : 0;
            }
        }, change);
        t.team_barrier();
        Kokkos::single(Kokkos::PerTeam(t), [&](){
            if(igain + change > 0){
                lock_bit(i) = 1;
            }
        });
    });
    Kokkos::parallel_for("afterburner heuristic", policy_t(0, num_pos - big), KOKKOS_LAMBDA(const ordinal_t& x){
        float change = 0;
        ordinal_t i = small_rows(x);
        part_t best = dest_part(i);
        part_t p = part(i);
        float wd = prob.wdeg(i);
        float multi = wd*inv_2m;
        float igain = pregain(i);
        ordinal_t hi = hash(i);
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++){
            ordinal_t v = g.graph.entries(j);
            float vgain = pregain(v);
            //adjust local gain if v has higher priority than i
            if((vgain - igain) >= 0.1 || (abs(vgain - igain) < 0.1 && static_cast<ordinal_t>(hash(v)) < hi)){
                part_t vpart = dest_part(v);
                scalar_t wgt = g.values(j);
                float q = static_cast<float>(wgt) - multi*prob.wdeg(v);
                change -= (vpart == p) ? q : 0;
                change += (vpart == best) ? q : 0;
                vpart = part(v);
                change += (vpart == p) ? q : 0;
                change -= (vpart == best) ? q : 0;
            }
        }
        if(igain + change >= 0){
            lock_bit(i) = 1;
        }
    });
    vtx_view_t swaps2 = Kokkos::subview(scratch.vtx2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    //scan all vertices that passed the post filter
    Kokkos::parallel_scan("filter beneficial moves", policy_t(0, num_pos), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(lock_bit(pos_moves(i))){
            if(final){
                swaps2(update) = pos_moves(i);
                // don't maintain locks in coarsening phase
                if(initial) lock_bit(pos_moves(i)) = 0;
            }
            update++;
        }
    }, scratch.scan_host);
    exec_space().fence();
    num_pos = scratch.scan_host();
    pos_moves = Kokkos::subview(swaps2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    return pos_moves;
}

//updates datastructures assuming a "large" number of vertices are moved
//2 kernels, 0 device-host syncs
void update_large(const problem& prob, part_vt part, const vtx_view_t swaps, scratch_mem& scratch, conn_data& cdata){
    const matrix_t& g = prob.g;
    const part_vt& constraint = prob.constraint;
    ordinal_t total_moves = swaps.extent(0);
    vtx_view_t swap_bit = scratch.zeros1;
    Kokkos::parallel_for("mark adjacent", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        swap_bit(i) = 1;
        cdata.dest_cache(i) = NULL_PART;
        //mark adjacent vertices
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            cdata.dest_cache(v) = NULL_PART;
            swap_bit(v) = 1;
        });
    });
    ordinal_t total = 0;
    Kokkos::parallel_scan("find swapped", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(swap_bit(i)){
            if(final){
                scratch.vtx1(update) = i;
                // reset to zero for next iteration
                swap_bit(i) = 0;
            }
            update++;
        }
    }, total);
    vtx_view_t affected = Kokkos::subview(scratch.vtx1, std::make_pair(static_cast<ordinal_t>(0), total));
    //recompute conn tables for each vertex adjacent to a moved vertex
    Kokkos::parallel_for("reset conn DS", team_policy_t(total, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(4*sizeof(part_t))), KOKKOS_LAMBDA(const member& t){
        const ordinal_t i = affected(t.league_rank());
        edge_offset_t g_start = cdata.conn_offsets(i);
        edge_offset_t g_end = cdata.conn_offsets(i + 1);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g_start, g_end), [&] (const edge_offset_t& j) {
            cdata.conn_entries(j) = NULL_PART;
            cdata.conn_vals(j) = 0;
        });
        part_t size = g_end - g_start;
        part_t* used_cap = (part_t*) t.team_shmem().get_shmem(sizeof(part_t));
        *used_cap = 0;
        part_t* s_conn_entries = cdata.conn_entries.data() + g_start;
        gain_t* s_conn_vals = cdata.conn_vals.data() + g_start;
        cdata.pvals(i) = 0;
        part_t c_i = constraint(i);
        part_t p_i = part(i);
        t.team_barrier();
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j, gain_t& update){
            ordinal_t v = g.graph.entries(j);
            if(c_i != constraint(v)) return;
            gain_t wgt = g.values(j);
            part_t p = part(v);
            if(p == p_i){
                update += wgt;
                return;
            }
            part_t p_o = hash(p) % static_cast<uint32_t>(size);
            bool success = false;
            while(!success){
                part_t px = s_conn_entries[p_o];
                while(px != p && px != NULL_PART){
                    p_o = (p_o + 1) % size;
                    px = s_conn_entries[p_o];
                }
                if(px == p){
                    success = true;
                } else {
                    px = Kokkos::atomic_compare_exchange(s_conn_entries + p_o, NULL_PART, p);
                    if(px == NULL_PART) Kokkos::atomic_add(used_cap, 1);
                    if(px == p || px == NULL_PART){
                        success = true;
                    } else {
                        p_o = (p_o + 1) % size;
                    }
                }
            }
            Kokkos::atomic_add(s_conn_vals + p_o, wgt);
        }, cdata.pvals(i));
        cdata.conn_table_sizes(i) = size;
    });
}

//update datastructures assuming a "small" number of vertices are moved
//2 kernels, 0 device-host syncs
void update_small(const problem& prob, const part_vt part, const vtx_view_t swaps, const part_vt dest_part, conn_data& cdata){
    const matrix_t& g = prob.g;
    const part_vt& constraint = prob.constraint;
    ordinal_t total_moves = swaps.extent(0);
    Kokkos::parallel_for("update conns (subtract) (high degree)", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        part_t p = part(i);
        //subtract i's contribution to p connectivity for adjacent vertices
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            if(constraint(i) != constraint(v)) return;
            gain_t wgt = g.values(j);
            if(p == part(v)){
                Kokkos::atomic_add(&cdata.pvals(v), -wgt);
                return;
            }
            edge_offset_t v_start = cdata.conn_offsets(v);
            part_t v_size = cdata.conn_table_sizes(v);
            part_t p_o = hash(p) % static_cast<uint32_t>(v_size);
            //v is always adjacent to p because it is adjacent to i which is in p
            while(cdata.conn_entries(v_start + p_o) != p){
                p_o = (p_o + 1) % v_size;
            }
            //DO NOT USE ATOMIC_ADD_FETCH HERE IT IS WAY SLOWER
            gain_t x = Kokkos::atomic_fetch_add(&cdata.conn_vals(v_start + p_o), -wgt);
            //parts have locked locations if v_size == k (even when not originally allocated to size k)
            if(x == wgt){
                //free this gain slot
                cdata.conn_entries(v_start + p_o) = HASH_RECLAIM;
            }
        });
    });

    // remove the new cluster id from hashmap, and insert old cluster id into hashmap
    Kokkos::parallel_for("swap pval and bval", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = swaps(x);
        // remove best (new cluster) from the hashmap
        part_t best = dest_part(i);
        // insert p (old cluster) into the hashmap
        part_t p = part(i);
        part(i) = best;
        dest_part(i) = p;
        part_t size = cdata.conn_table_sizes(i);
        edge_offset_t offset = cdata.conn_offsets(i);
        part_t b_hash = hash(best) % static_cast<uint32_t>(size);
        gain_t old_val = cdata.pvals(i);
        cdata.pvals(i) = 0;
        // find new cluster's connection strength, and set into pval
        for(part_t q = 0; q < size; q++){
            part_t p_i = (b_hash + q) % size;
            if(cdata.conn_entries(offset + p_i) == best){
                cdata.pvals(i) = cdata.conn_vals(offset + p_i);
                // delete from hashmap
                cdata.conn_vals(offset + p_i) = 0;
                cdata.conn_entries(offset + p_i) = HASH_RECLAIM;
                break;
            } else if(cdata.conn_entries(offset + p_i) == NULL_PART){
                break;
            }
        }
        // DO NOT insert old cluster into hashmap if it has no connection to vertex
        if(old_val == 0) return;
        part_t count = 0;
        bool success = false;
        part_t p_o = hash(p) % static_cast<uint32_t>(size);
        //insert p into conn table
        //needs to find either HASH_RECLAIM or NULL_PART to make insertion
        while(!success && count < size){
            part_t px = cdata.conn_entries(offset + p_o);
            while(px > NULL_PART && count++ < size){
                p_o = (p_o + 1) % size;
                px = cdata.conn_entries(offset + p_o);
            }
            if(px <= NULL_PART) {
                cdata.conn_entries(offset + p_o) = p;
                success = true;
            }
        }
        //if we run out of space, start densely allocating after end of current hash table
        if(!success){
            p_o = size;
            while(!success){
                part_t px = cdata.conn_entries(offset + p_o);
                while(px > NULL_PART && count++ < size){
                    p_o++;
                    px = cdata.conn_entries(offset + p_o);
                }
                if(px <= NULL_PART){
                    cdata.conn_entries(offset + p_o) = p;
                    cdata.conn_table_sizes(i)++;
                    success = true;
                }
            }
        }
        cdata.conn_vals(offset + p_o) = old_val;
    });

    Kokkos::parallel_for("update conns (add) (high degree)", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        //part contains new part at this point
        part_t best = part(i);
        //add i's contribution to best connectivity for adjacent vertices
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
                if(constraint(i) != constraint(v)) return;
            gain_t wgt = g.values(j);
            cdata.dest_cache(v) = NULL_PART;
            if(best == part(v)){
                Kokkos::atomic_add(&cdata.pvals(v), wgt);
                return;
            }
            edge_offset_t v_start = cdata.conn_offsets(v);
            part_t v_size = cdata.conn_table_sizes(v);
            part_t p_o = hash(best) % static_cast<uint32_t>(v_size);
            bool success = false;
            //check if best in conn table
            //can only determine best is absent if NULL_PART is found or v_size reached
            for(part_t q = 0; q < v_size; q++){
                part_t p_i = (p_o + q) % v_size;
                part_t px = cdata.conn_entries(v_start + p_i);
                if(px == best){
                    success = true;
                    p_o = p_i;
                    break;
                } else if(px == NULL_PART){
                    break;
                }
            }
            part_t count = 0;
            //insert best into conn table
            //needs to find either HASH_RECLAIM or NULL_PART to make insertion
            while(!success && count < v_size){
                part_t px = cdata.conn_entries(v_start + p_o);
                while(px != best && px > NULL_PART && count++ < v_size){
                    p_o = (p_o + 1) % v_size;
                    px = cdata.conn_entries(v_start + p_o);
                }
                if(px == best){
                    success = true;
                } else {
                    part_t orig = NULL_PART;
                    if(cdata.conn_entries(v_start + p_o) == HASH_RECLAIM) orig = HASH_RECLAIM;
                    //don't care if this thread succeeds if another thread succeeds with the same value
                    Kokkos::atomic_compare_exchange(&cdata.conn_entries(v_start + p_o), orig, best);
                    if(cdata.conn_entries(v_start + p_o) == best){
                        success = true;
                    } else {
                        p_o = (p_o + 1) % v_size;
                        count++;
                    }
                }
            }
            //if we run out of space, start densely allocating after end of current hash table
            if(!success){
                p_o = v_size;
                while(!success){
                    part_t px = cdata.conn_entries(v_start + p_o);
                    while(px != best && px > NULL_PART && count++ < v_size){
                        p_o++;
                        px = cdata.conn_entries(v_start + p_o);
                    }
                    if(px == best){
                        success = true;
                    } else {
                        part_t orig = NULL_PART;
                        if(cdata.conn_entries(v_start + p_o) == HASH_RECLAIM) orig = HASH_RECLAIM;
                        if(Kokkos::atomic_compare_exchange(&cdata.conn_entries(v_start + p_o), orig, best) == orig){
                            Kokkos::atomic_add(&cdata.conn_table_sizes(v), 1);
                        }
                        //don't care if this thread succeeded if another thread succeeded with the same value
                        if(cdata.conn_entries(v_start + p_o) == best){
                            success = true;
                        } else {
                            p_o++;
                        }
                    }
                }
            }
            Kokkos::atomic_add(&cdata.conn_vals(v_start + p_o), wgt);
        });
    });
}

KOKKOS_INLINE_FUNCTION
static gain_t lookup(const part_t* keys, const gain_t* vals, const part_t& target, const part_t& size){
    part_t start = hash(target) % static_cast<uint32_t>(size);
    for(part_t q = 0; q < size; q++){
        part_t p_i = (start + q) % size;
        if(keys[p_i] == target){
            return vals[p_i];
        } else if(keys[p_i] == NULL_PART){
            return 0;
        }
    }
    return 0;
}

//perform swaps, update gains, and compute change to cut and imbalance
//4 kernels, 1 device-host syncs
void perform_moves(const problem& prob, part_vt part, const vtx_view_t swaps, const part_vt dest_part, scratch_mem& scratch, conn_data cdata, refine_data& curr_state, bool use_big){
    const wgt_view_t& wdeg = prob.wdeg;
    const wgt_view_t& nb_self_loops = prob.nb_self_loops;
    ordinal_t total_moves = swaps.extent(0);
    //total change in cutsize = (sum over all moves) -((new_b_con - new_p_con) + (old_b_con - old_p_con))
    Kokkos::parallel_reduce("count cutsize change part1", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t& x, gain_t& gain_update){
        ordinal_t i = swaps(x);
        part_t best = dest_part(i);
        part_t p = part(i);
        edge_offset_t start = cdata.conn_offsets(i);
        part_t size = cdata.conn_table_sizes(i);
        // edges of this vertex in p before moving
        gain_t p_con = cdata.pvals(i);
        // edges of this vertex in best before moving
        // this is stored by an earlier lookup because the lookup is very expensive for the initial pass (the hashtables don't function as hashtables in the initial pass)
        gain_t b_con = cdata.bvals(i);
        Kokkos::atomic_add(&curr_state.in_deg(p), -p_con);
        Kokkos::atomic_add(&curr_state.in_deg(best), b_con);
        gain_update += b_con - p_con;
    }, scratch.cut_change1);
    //change part assignments and update part sizes
    if(use_big || total_moves >= prob.g.numRows() * 0.2){
        Kokkos::parallel_for("perform moves", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
            ordinal_t i = swaps(x);
            part_t p = part(i);
            part_t best = dest_part(i);
            cdata.dest_cache(i) = NULL_PART;
            part(i) = best;
            dest_part(i) = p;
            Kokkos::atomic_add(&curr_state.total_deg(p), -wdeg(i));
            Kokkos::atomic_add(&curr_state.total_deg(best), wdeg(i));
            Kokkos::atomic_add(&curr_state.in_deg(p), -nb_self_loops(i));
            Kokkos::atomic_add(&curr_state.in_deg(best), nb_self_loops(i));
        });
        update_large(prob, part, swaps, scratch, cdata);
    } else {
        Kokkos::parallel_for("perform moves", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
            ordinal_t i = swaps(x);
            part_t p = part(i);
            part_t best = dest_part(i);
            cdata.dest_cache(i) = NULL_PART;
            Kokkos::atomic_add(&curr_state.total_deg(p), -wdeg(i));
            Kokkos::atomic_add(&curr_state.total_deg(best), wdeg(i));
            Kokkos::atomic_add(&curr_state.in_deg(p), -nb_self_loops(i));
            Kokkos::atomic_add(&curr_state.in_deg(best), nb_self_loops(i));
        });
        update_small(prob, part, swaps, dest_part, cdata);
    }
    Kokkos::parallel_reduce("count cutsize change part2", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t& x, gain_t& gain_update){
        ordinal_t i = swaps(x);
        part_t p = dest_part(i);
        part_t best = part(i);
        edge_offset_t start = cdata.conn_offsets(i);
        part_t size = cdata.conn_table_sizes(i);
        //edges of other vertices in p connecting to this vertex after moving
        // this lookup is cheaper now since the hashtables should be less full
        gain_t p_con = lookup(cdata.conn_entries.data() + start, cdata.conn_vals.data() + start, p, size);
        //edges of other vertices in best connecting to this vertex after moving
        gain_t b_con = cdata.pvals(i);
        Kokkos::atomic_add(&curr_state.in_deg(p), -p_con);
        Kokkos::atomic_add(&curr_state.in_deg(best), b_con);
        gain_update += b_con - p_con;
    }, scratch.cut_change2);
    exec_space().fence();
    int64_t cut_change = scratch.cut_change2() + scratch.cut_change1();
    curr_state.cut -= cut_change;
} 

//initializes datastructures
conn_data init_conn_data(const conn_data& scratch_cdata, const matrix_t& g, const part_vt& part, const part_vt& constraint, bool is_initial){
    ordinal_t n = g.numRows();
    conn_data cdata;
    cdata.conn_offsets = Kokkos::subview(scratch_cdata.conn_offsets, std::make_pair(static_cast<ordinal_t>(0), n + 1));
    Kokkos::parallel_for("comp conn row size", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t& i){
        ordinal_t degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        cdata.conn_offsets(i + 1) = degree;
    });
    edge_offset_t gain_size = 0;
    Kokkos::parallel_scan("comp conn offsets", policy_t(0, n + 1), KOKKOS_LAMBDA(const ordinal_t& i, edge_offset_t& update, const bool final){
        update += cdata.conn_offsets(i);
        if(final){
            cdata.conn_offsets(i) = update;
        }
    }, gain_size);
    cdata.conn_vals = Kokkos::subview(scratch_cdata.conn_vals, std::make_pair(static_cast<edge_offset_t>(0), gain_size));
    cdata.pvals = Kokkos::subview(scratch_cdata.pvals, std::make_pair(static_cast<ordinal_t>(0), n));
    cdata.bvals = Kokkos::subview(scratch_cdata.bvals, std::make_pair(static_cast<ordinal_t>(0), n));
    cdata.conn_entries = Kokkos::subview(scratch_cdata.conn_entries, std::make_pair(static_cast<edge_offset_t>(0), gain_size));
    cdata.dest_cache = Kokkos::subview(scratch_cdata.dest_cache, std::make_pair(static_cast<ordinal_t>(0), n));
    cdata.conn_table_sizes = Kokkos::subview(scratch_cdata.conn_table_sizes, std::make_pair(static_cast<ordinal_t>(0), n));
    cdata.lock_bit = Kokkos::subview(scratch_cdata.lock_bit, std::make_pair(static_cast<ordinal_t>(0), n));
    Kokkos::deep_copy(exec_space(), cdata.conn_vals, 0);
    Kokkos::deep_copy(exec_space(), cdata.pvals, 0);
    Kokkos::deep_copy(exec_space(), cdata.conn_entries, NULL_PART);
    Kokkos::deep_copy(exec_space(), cdata.dest_cache, NULL_PART);
    Kokkos::deep_copy(exec_space(), cdata.lock_bit, 0);
    //initialize conn tables for each vertex
    //conn tables are resized to be small so that traversal is faster, but large enough so that updates have few collisions
    if(!is_initial){
        //low degree version
        Kokkos::parallel_for("init conn DS", team_policy_t(g.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
            ordinal_t i = t.league_rank();
            edge_offset_t g_start = cdata.conn_offsets(i);
            edge_offset_t g_end = cdata.conn_offsets(i + 1);
            part_t size = g_end - g_start;
            part_t* s_conn_entries = cdata.conn_entries.data() + g_start;
            gain_t* s_conn_vals = cdata.conn_vals.data() + g_start;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j, gain_t& update){
                ordinal_t v = g.graph.entries(j);
                if(constraint(i) != constraint(v)) return;
                gain_t wgt = g.values(j);
                part_t p = part(v);
                if(p == part(i)){
                    update += wgt;
                    return;
                }
                part_t p_o = hash(p) % static_cast<uint32_t>(size);
                bool success = false;
                while(!success){
                    part_t px = s_conn_entries[p_o];
                    while(px != p && px != NULL_PART){
                        p_o = (p_o + 1) % size;
                        px = s_conn_entries[p_o];
                    }
                    if(px == p){
                        success = true;
                    } else {
                        Kokkos::atomic_compare_exchange(s_conn_entries + p_o, NULL_PART, p);
                        if(s_conn_entries[p_o] == p){
                            success = true;
                        } else {
                            p_o = (p_o + 1) % size;
                        }
                    }
                }
                Kokkos::atomic_add(s_conn_vals + p_o, wgt);
            }, cdata.pvals(i));
            cdata.conn_table_sizes(i) = size;
        });
    } else {
        // the initial pass has every vertex in a singleton cluster
        // so the cluster adjacency structure is essentially the same as the input graph
        // however, we may still need to ignore certain adjacencies given the constraint
        Kokkos::parallel_for("init conn DS", team_policy_t(g.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
            ordinal_t i = t.league_rank();
            edge_offset_t g_start = cdata.conn_offsets(i);
            edge_offset_t g_end = cdata.conn_offsets(i + 1);
            part_t size = g_end - g_start;
            cdata.conn_table_sizes(i) = size;
            part_t ci = constraint(i);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j){
                ordinal_t v = g.graph.entries(j);
                if(ci != constraint(v)){
                    cdata.conn_entries(j) = HASH_RECLAIM;
                    return;
                };
                gain_t wgt = g.values(j);
                cdata.conn_entries(j) = v;
                cdata.conn_vals(j) = wgt;
            });
        });
    }
    return cdata;
}

void jet_refine(const matrix_t g, wgt_view_t wdeg, wgt_view_t nb_self_loops, part_vt best_part, part_vt constraint, refine_data& best_state, bool is_initial, ExperimentLoggerUtil<scalar_t>& experiment){
    Kokkos::Timer y;
    //contains several scratch views that are reused in each iteration
    //reallocating in each iteration would be expensive (GPU memory is often slow to deallocate)
    scratch_mem& scratch = perm_scratch;
    //initialize metadata if this is first level being refined
    //ie. if this is the coarsest level
    if(!best_state.init){
        best_state.mod = -1.0;
        best_state.in_deg = gain_vt("internal degree of clusters", g.numRows());
        best_state.total_deg = gain_vt("total degree of clusters", g.numRows());
        best_state.g_deg = stat::sum(wdeg);
        best_state.cut = best_state.g_deg;
        Kokkos::deep_copy(best_state.in_deg, nb_self_loops);
        Kokkos::deep_copy(best_state.total_deg, wdeg);
        best_state.init = true;
    }
    problem prob;
    prob.g = g;
    prob.wdeg = wdeg;
    prob.nb_self_loops = nb_self_loops;
    prob.constraint = constraint;
    refine_data curr_state = clone_refine_data(best_state);
    part_vt part(Kokkos::ViewAllocateWithoutInitializing("current partition"), g.numRows());
    Kokkos::deep_copy(exec_space(), part, best_part);
    conn_data cdata = init_conn_data(perm_cdata, g, part, constraint, is_initial);
    int iter_count = 0;
    Kokkos::fence();
    Kokkos::Timer iter_t;
    double tol = 1;//0.999;
#ifdef FOUR9
    tol = 0.9999;
#elif defined TWO9
    tol = 0.99;
#endif
    //repeat until 12 phases since a significant
    //improvement in cut or balance
    //this accounts for at least 3 full lp+rebalancing cycles
    float filter_ratio = 0.5;
    if(best_state.g_deg == g.nnz()) filter_ratio = 0.5;
    bool use_big = true;
//    for(filter_ratio = 0.95; filter_ratio >= 0; filter_ratio -= 0.05){
        int count = 0;
        while(count++ <= 5){
            iter_count++;
            if(iter_count > 3) use_big = false;
            vtx_view_t moves;
            moves = jet_lp(prob, part, curr_state, cdata, scratch, filter_ratio, true);
            if(moves.extent(0) == 0) return;
            perform_moves(prob, part, moves, scratch.dest_part, scratch, cdata, curr_state, use_big);
            curr_state.mod = stat::modularity(curr_state.g_deg, curr_state.in_deg, curr_state.total_deg);
            std::cout << "Cut: " << curr_state.cut << "; Modularity: " << std::setprecision(6) << curr_state.mod << "; Labels: " << stat::total_labels(curr_state.total_deg) << std::endl;
            //copy current partition and relevant data to output partition if following conditions pass
            if(curr_state.mod > best_state.mod){
                //do not reset counter if cut improvement is too small
                if(curr_state.mod > tol*best_state.mod){
                    // count = 0;
                }
                copy_refine_data(best_state, curr_state);
                Kokkos::deep_copy(exec_space(), best_part, part);
            }
        }
    // }
    Kokkos::fence();
    //divide cut by 2 because each cut edge is counted from both sides
    typename ExperimentLoggerUtil<scalar_t>::CoarseLevel cl(best_state.cut / 2, 0, g.nnz(), g.numRows(), y.seconds(), iter_t.seconds(), iter_count, iter_count);
    std::cout << "Avg iteration time: " << (iter_t.seconds() / iter_count) << std::endl;
    experiment.addCoarseLevel(cl);
    y.reset();
    iter_t.reset();
}
};

}
